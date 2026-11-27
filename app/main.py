import asyncio
import base64
import hmac
import json
import os
import re
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set

import cv2
from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from app.auth import (
    create_access_token,
    get_current_user,
    get_ws_user,
    hash_password,
    require_role,
    verify_password,
)
from app.clinical_engine import DripFlowEngine, EngineConfig, OnnxDropDetector
from app.clinical_rules import (
    evaluate_infusion_order,
    explain_telemetry_decision,
    validate_lifecycle_transition,
)
from app.database import (
    BED_VIDEO_MAP,
    VIDEO_BED_MAP,
    Alert,
    AuditLog,
    InfusionBed,
    InfusionSession,
    Patient,
    SessionLocal,
    TelemetryLog,
    User,
    atomic_bed_update,
    get_db,
    init_db,
    record_audit,
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best.onnx"
DEFAULTS_PATH = BASE_DIR / "engine_defaults.json"
VIDEOS_DIR = (BASE_DIR / "sample_videos").resolve()
STATIC_DIR = (BASE_DIR / "static").resolve()
ARCHIVES_DIR = (BASE_DIR / "archives").resolve()

VIDEOS_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)
ARCHIVES_DIR.mkdir(exist_ok=True)

CLINICAL_SECURITY_CODE = os.getenv("DRIPGUARD_SECURITY_CODE", "DG-SEC-2026")

DEFAULTS = json.loads(DEFAULTS_PATH.read_text(encoding="utf-8")) if DEFAULTS_PATH.exists() else {}
DEFAULT_CONF = float(DEFAULTS.get("conf", 0.22))
DEFAULT_IOU = float(DEFAULTS.get("iou", 0.60))
DEFAULT_SIZE = int(DEFAULTS.get("input_size", 640))

# Wide adaptive ROI presets so swaying saline chambers never leave the scan field
VIDEO_PRESETS = {
    "12076506": {"roi": [0.15, 0.12, 0.85, 0.88], "corridor": [0.25, 0.15, 0.75, 0.82]},
    "15011120": {"roi": [0.20, 0.20, 0.85, 0.85], "corridor": [0.38, 0.30, 0.85, 0.78]},
    "6248839":  {"roi": [0.15, 0.12, 0.85, 0.88], "corridor": [0.25, 0.10, 0.75, 0.85]},
    "13187829": {"roi": [0.12, 0.12, 0.88, 0.88], "corridor": [0.22, 0.12, 0.78, 0.86]},
    "12080909": {"roi": [0.15, 0.12, 0.85, 0.88], "corridor": [0.25, 0.15, 0.75, 0.82]},
    "15765899": {"roi": [0.15, 0.12, 0.85, 0.88], "corridor": [0.25, 0.12, 0.75, 0.85]},
}


def _get_preset_for_video(video_name: str) -> dict:
    for key, preset in VIDEO_PRESETS.items():
        if key in video_name:
            return preset
    return {"roi": [0.12, 0.12, 0.88, 0.88], "corridor": [0.22, 0.12, 0.78, 0.86]}


def _verify_security_code(code: Optional[str]) -> None:
    if not code or not hmac.compare_digest(code.strip(), CLINICAL_SECURITY_CODE):
        raise HTTPException(
            status_code=403,
            detail="Invalid Clinical Security Code. Action rejected (only authorized clinicians hold the code).",
        )


def _generate_discharge_pdf(db: Session, bed: InfusionBed, actor: str, disposition: str) -> str:
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    ts_slug = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    patient_name = bed.patient.full_name if bed.patient else "Unassigned"
    patient_mrn = bed.patient.mrn if bed.patient else "N/A"
    patient_age = f"{bed.patient.age_years} yrs" if bed.patient else "N/A"
    patient_wt = f"{bed.patient.weight_kg} kg" if bed.patient else "N/A"

    latest_log = (
        db.query(TelemetryLog)
        .filter(TelemetryLog.bed_id == bed.id)
        .order_by(TelemetryLog.id.desc())
        .first()
    )
    total_logs = db.query(func.count(TelemetryLog.id)).filter(TelemetryLog.bed_id == bed.id).scalar() or 0
    total_alerts = db.query(func.count(Alert.id)).filter(Alert.bed_id == bed.id).scalar() or 0
    unacked_alerts = (
        db.query(func.count(Alert.id))
        .filter(Alert.bed_id == bed.id, Alert.acknowledged == False)
        .scalar()
        or 0
    )

    final_drops = latest_log.total_drops if latest_log else 0
    final_dpm = f"{latest_log.dpm:.1f} DPM" if (latest_log and latest_log.dpm is not None) else "N/A"
    final_ml_hr = f"{latest_log.ml_hr:.1f} mL/hr" if (latest_log and latest_log.ml_hr is not None) else "N/A"
    final_infused = f"{latest_log.ml_infused:.2f} mL" if latest_log else "0.00 mL"
    final_status = latest_log.status if latest_log else bed.lifecycle_state

    lines = [
        "DRIPGUARD AI - CLINICAL INFUSION DISCHARGE & AUDIT RECORD",
        "========================================================================",
        f"Archive Timestamp : {now_str}",
        f"Disposition       : {disposition}",
        f"Authorized By     : {actor}",
        "",
        "1. PATIENT DEMOGRAPHICS",
        "------------------------------------------------------------------------",
        f"Patient Full Name : {patient_name}",
        f"Medical Record #  : {patient_mrn}",
        f"Age / Weight      : {patient_age} / {patient_wt}",
        "",
        "2. BED & OPTICAL FEED ASSIGNMENT",
        "------------------------------------------------------------------------",
        f"Bed Code / Ward   : {bed.bed_code} ({bed.ward_name})",
        f"Optical Video Src : {bed.video_source}",
        f"Chamber ROI       : {bed.roi_json}",
        f"Nozzle Corridor   : {bed.corridor_json}",
        "",
        "3. PRESCRIBED IV INFUSION ORDER",
        "------------------------------------------------------------------------",
        f"IV Fluid Type     : {bed.fluid_type}",
        f"Prescribed Rate   : {bed.prescribed_dpm if bed.prescribed_dpm is not None else 'Observation'} DPM",
        f"Tubing Drop Factor: {bed.drop_factor} gtt/mL",
        f"Target Bag Volume : {bed.target_volume_ml} mL",
        f"Target Duration   : {bed.target_duration_min} min",
        "",
        "4. FINAL TELEMETRY & SAFETY SUMMARY",
        "------------------------------------------------------------------------",
        f"Final Flow Status : {final_status}",
        f"Total Drop Events : {final_drops} drops",
        f"Last Measured Rate: {final_dpm} ({final_ml_hr})",
        f"Total Infused Vol : {final_infused}",
        f"Telemetry Samples : {total_logs} logged intervals",
        f"Clinical Alerts   : {total_alerts} total ({unacked_alerts} unacknowledged)",
        "========================================================================",
        "End of Certified Electronic Medical Archive Record - DripGuard AI",
    ]

    def pdf_escape(s: str) -> str:
        clean = s.encode("latin-1", "replace").decode("latin-1")
        return clean.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    content_ops = ["BT", "/F1 10 Tf", "45 785 Td", "14 TL"]
    for idx, line in enumerate(lines):
        escaped = pdf_escape(line)
        if idx == 0:
            content_ops.append(f"({escaped}) Tj")
        else:
            content_ops.append(f"T* ({escaped}) Tj")
    content_ops.append("ET")
    stream_bytes = "\n".join(content_ops).encode("latin-1")

    objects = [
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n",
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 842] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n",
        f"4 0 obj\n<< /Length {len(stream_bytes)} >>\nstream\n".encode("latin-1") + stream_bytes + b"\nendstream\nendobj\n",
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>\nendobj\n",
    ]

    pdf_buf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for obj in objects:
        offsets.append(len(pdf_buf))
        pdf_buf.extend(obj)

    xref_pos = len(pdf_buf)
    pdf_buf.extend(f"xref\n0 {len(offsets)}\n".encode("latin-1"))
    pdf_buf.extend(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        pdf_buf.extend(f"{off:010d} 00000 n \n".encode("latin-1"))

    pdf_buf.extend(
        f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode("latin-1")
    )

    safe_mrn = re.sub(r"[^a-zA-Z0-9_-]", "_", patient_mrn)
    safe_bed = re.sub(r"[^a-zA-Z0-9_-]", "_", bed.bed_code)
    pdf_filename = f"Discharge_{safe_bed}_{safe_mrn}_{ts_slug}.pdf"
    (ARCHIVES_DIR / pdf_filename).write_bytes(bytes(pdf_buf))
    return pdf_filename


def seed_initial_data() -> None:
    init_db()
    db = SessionLocal()
    try:
        seed_pw = os.getenv("DRIPGUARD_DEMO_PASSWORD", "ChangeMe@2026")
        if db.query(User).count() == 0:
            db.add(User(username="nurse1", full_name="Senior ICU Nurse", hashed_password=hash_password(seed_pw), role="nurse"))
            db.add(User(username="doctor1", full_name="Attending Physician", hashed_password=hash_password(seed_pw), role="doctor"))
            db.add(User(username="admin", full_name="Clinical Engineer", hashed_password=hash_password(seed_pw), role="admin"))
            db.commit()

        if db.query(Patient).count() == 0:
            patients_data = [
                Patient(mrn="MRN-1001", full_name="Rahim Uddin", age_years=54, weight_kg=68.0, is_pediatric=False, ward_notes="Post-op hydration"),
                Patient(mrn="MRN-1002", full_name="Baby Ayaan", age_years=3, weight_kg=14.0, is_pediatric=True, ward_notes="Pediatric micro-drip protocol"),
                Patient(mrn="MRN-1003", full_name="Fatema Begum", age_years=42, weight_kg=58.0, is_pediatric=False, ward_notes="Electrolyte replacement therapy"),
                Patient(mrn="MRN-1004", full_name="Kamrul Hasan", age_years=61, weight_kg=74.0, is_pediatric=False, ward_notes="Cardiac step-down maintenance IV"),
                Patient(mrn="MRN-1005", full_name="Nusrat Jahan", age_years=29, weight_kg=52.0, is_pediatric=False, ward_notes="Sepsis antibiotic infusion"),
                Patient(mrn="MRN-1006", full_name="Baby Zayan", age_years=5, weight_kg=18.5, is_pediatric=True, ward_notes="Pediatric dextrose maintenance"),
            ]
            db.add_all(patients_data)
            db.commit()
            for p in patients_data:
                db.refresh(p)

            available_vids = sorted([p.name for p in VIDEOS_DIR.glob("*.mp4")])

            def resolve_vid(bed_code: str, fallback_idx: int) -> str:
                mapped = BED_VIDEO_MAP.get(bed_code)
                if mapped and (VIDEOS_DIR / mapped).exists():
                    return mapped
                if available_vids:
                    return available_vids[fallback_idx % len(available_vids)]
                return "webcam:0"

            bed_specs = [
                ("BED-01", "ICU-A", patients_data[0].id, "Normal Saline 0.9% (500 mL)", 500.0, 165.0, 20.0, 60.0),
                ("BED-02", "PICU-B", patients_data[1].id, "Dextrose 5% in 0.45% NaCl", 250.0, 300.0, 60.0, 50.0),
                ("BED-03", "ICU-A", patients_data[2].id, "Ringer's Lactate (1000 mL)", 1000.0, 360.0, 20.0, 55.0),
                ("BED-04", "CCU-1", patients_data[3].id, "Normal Saline 0.9% (500 mL)", 500.0, 250.0, 20.0, 40.0),
                ("BED-05", "MED-W2", patients_data[4].id, "Dextrose 5% Water (500 mL)", 500.0, 200.0, 20.0, 50.0),
                ("BED-06", "PICU-B", patients_data[5].id, "Pediatric 0.45% NaCl (250 mL)", 250.0, 330.0, 60.0, 45.0),
            ]

            beds_to_add = []
            for idx, (b_code, ward, pid, fluid, vol, dur, df, dpm) in enumerate(bed_specs):
                v_src = resolve_vid(b_code, idx)
                preset = _get_preset_for_video(v_src)
                beds_to_add.append(
                    InfusionBed(
                        bed_code=b_code,
                        ward_name=ward,
                        patient_id=pid,
                        fluid_type=fluid,
                        target_volume_ml=vol,
                        target_duration_min=dur,
                        drop_factor=df,
                        prescribed_dpm=dpm,
                        video_source=v_src,
                        roi_json=json.dumps(preset["roi"]),
                        corridor_json=json.dumps(preset["corridor"]),
                        lifecycle_state="RUNNING",
                    )
                )

            db.add_all(beds_to_add)
            db.commit()
    finally:
        db.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    seed_initial_data()
    yield
    await sync_hub.shutdown_all()


app = FastAPI(title="DripGuard AI Clinical Mission Control", version="2.7.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

detector: Optional[OnnxDropDetector] = None


def get_detector() -> OnnxDropDetector:
    global detector
    if detector is None:
        if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size == 0:
            raise HTTPException(
                status_code=503,
                detail=f"ONNX model file is missing or empty at {MODEL_PATH}. Place valid best.onnx in models/.",
            )
        detector = OnnxDropDetector(
            str(MODEL_PATH),
            conf=max(DEFAULT_CONF, 0.20),
            iou=DEFAULT_IOU,
            size=DEFAULT_SIZE,
            providers=("CPUExecutionProvider",),
        )
    return detector


def validate_safe_video_source(src: str) -> None:
    if src.startswith("webcam:"):
        idx_str = src.split(":", 1)[1]
        if not idx_str.isdigit() or int(idx_str) > 8:
            raise HTTPException(status_code=422, detail="Invalid webcam index (must be webcam:0 .. webcam:8)")
        return
    if "/" in src or "\\" in src or ".." in src:
        raise HTTPException(status_code=422, detail="Invalid video_source path (path traversal blocked)")
    candidate = (VIDEOS_DIR / src).resolve()
    if candidate.parent != VIDEOS_DIR or not candidate.is_file():
        raise HTTPException(status_code=404, detail=f"Sample video not found: {src}")


class BedSyncHub:
    def __init__(self) -> None:
        self.subscribers: Dict[int, Set[WebSocket]] = {}
        self.live_configs: Dict[int, dict] = {}
        self.reset_flags: Set[int] = set()
        self.source_reload_flags: Set[int] = set()
        self.workers: Dict[int, asyncio.Task] = {}

    async def broadcast_bed_event(self, bed_id: int, event: dict) -> None:
        for ws in list(self.subscribers.get(bed_id, set())):
            try:
                await ws.send_json(event)
            except Exception:
                self.subscribers.get(bed_id, set()).discard(ws)

    def ensure_worker(self, bed_id: int, started_by: str) -> None:
        task = self.workers.get(bed_id)
        if task is None or task.done():
            self.workers[bed_id] = asyncio.create_task(self._run_bed_worker(bed_id, started_by))

    async def stop_bed_worker(self, bed_id: int) -> None:
        task = self.workers.pop(bed_id, None)
        if task and not task.done():
            task.cancel()
        for ws in list(self.subscribers.pop(bed_id, set())):
            try:
                await ws.close(code=1000, reason="Bed closed or deleted")
            except Exception:
                pass

    async def shutdown_all(self) -> None:
        for task in list(self.workers.values()):
            task.cancel()
        self.workers.clear()

    async def _run_bed_worker(self, bed_id: int, started_by: str) -> None:
        try:
            det = get_detector()
        except HTTPException as exc:
            await self.broadcast_bed_event(bed_id, {"error": exc.detail})
            return

        db = SessionLocal()
        bed = db.query(InfusionBed).filter(InfusionBed.id == bed_id).first()
        if not bed:
            db.close()
            await self.broadcast_bed_event(bed_id, {"error": "Bed not found"})
            return

        if bed.lifecycle_state == "VACANT":
            db.close()
            await self.broadcast_bed_event(bed_id, {"type": "vacant", "message": "Bed is currently VACANT. Assign a new patient & video to start monitoring."})
            return

        src = bed.video_source
        roi = json.loads(bed.roi_json)
        corridor = tuple(json.loads(bed.corridor_json))
        cfg = EngineConfig(
            drop_factor=bed.drop_factor,
            prescribed_dpm=bed.prescribed_dpm,
            tolerance=bed.tolerance,
            corridor=corridor,
            auto_track=True,
            min_conf=0.22,
        )
        session_rec = InfusionSession(bed_id=bed_id, started_by=started_by, state="RUNNING")
        db.add(session_rec)
        db.commit()
        db.refresh(session_rec)
        session_id = session_rec.id
        db.close()

        engine = DripFlowEngine(cfg)

        def open_capture(source_str: str):
            if source_str.startswith("webcam:"):
                return cv2.VideoCapture(int(source_str.split(":")[1])), False
            v_path = (VIDEOS_DIR / source_str).resolve()
            if v_path.parent != VIDEOS_DIR or not v_path.exists():
                return None, True
            return cv2.VideoCapture(str(v_path)), True

        cap, is_file = open_capture(src)
        if cap is None:
            await self.broadcast_bed_event(bed_id, {"error": f"Video not found: {src}"})
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if fps <= 1 or fps > 120:
            fps = 30.0
        frame_delay = 1.0 / min(fps, 30.0)

        frame_idx = 0
        last_db_log_idx = -999

        try:
            while self.subscribers.get(bed_id):
                pushed = self.live_configs.pop(bed_id, None)
                if pushed:
                    engine.cfg.prescribed_dpm = pushed["prescribed_dpm"]
                    engine.cfg.drop_factor = float(pushed["drop_factor"])
                    engine.cfg.tolerance = float(pushed.get("tolerance", engine.cfg.tolerance))
                    if "auto_track" in pushed:
                        engine.cfg.auto_track = bool(pushed["auto_track"])
                    roi = pushed["roi"]
                    corridor = tuple(pushed["corridor"])
                    engine.sync_base_corridor(corridor)
                    if bed_id in self.source_reload_flags:
                        self.source_reload_flags.discard(bed_id)
                        cap.release()
                        cap, is_file = open_capture(pushed["video_source"])
                        if cap is None:
                            await self.broadcast_bed_event(bed_id, {"error": f"Video not found: {pushed['video_source']}"})
                            break
                        engine.reset()
                        frame_idx = 0
                        last_db_log_idx = -999

                if bed_id in self.reset_flags:
                    self.reset_flags.discard(bed_id)
                    engine.reset()
                    frame_idx = 0
                    last_db_log_idx = -999
                    if is_file:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

                ok, fr = cap.read()
                if not ok:
                    if is_file:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        engine.reset()
                        frame_idx = 0
                        last_db_log_idx = -999
                        continue
                    break

                H, W = fr.shape[:2]
                t = frame_idx / fps

                rx0 = max(0, min(W - 10, int(roi[0] * W)))
                ry0 = max(0, min(H - 10, int(roi[1] * H)))
                rx1 = max(rx0 + 10, min(W, int(roi[2] * W)))
                ry1 = max(ry0 + 10, min(H, int(roi[3] * H)))
                rw, rh = rx1 - rx0, ry1 - ry0

                roi_crop = fr[ry0:ry1, rx0:rx1]
                boxes_roi = await asyncio.to_thread(det.detect, roi_crop)
                st = engine.update(t, boxes_roi)

                xai = explain_telemetry_decision(
                    status=st.status,
                    dpm=st.dpm,
                    prescribed_dpm=engine.cfg.prescribed_dpm,
                    tolerance=engine.cfg.tolerance,
                    ml_hr=st.ml_hr,
                    drop_factor=engine.cfg.drop_factor,
                    raw_reason=st.reason,
                )

                out_w = 440
                out_h = int(H * (out_w / W))
                vis = cv2.resize(fr, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
                sx, sy = out_w / W, out_h / H

                # Draw outer Search ROI (Yellow)
                cv2.rectangle(vis, (int(rx0 * sx), int(ry0 * sy)), (int(rx1 * sx), int(ry1 * sy)), (220, 190, 80), 1)

                # Draw Dynamic Self-Centering Nozzle Corridor (Cyan/Blue)
                dyn_corr = st.active_corridor or corridor
                cx1 = int((rx0 + dyn_corr[0] * rw) * sx)
                cy1 = int((ry0 + dyn_corr[1] * rh) * sy)
                cx2 = int((rx0 + dyn_corr[2] * rw) * sx)
                cy2 = int((ry0 + dyn_corr[3] * rh) * sy)
                cv2.rectangle(vis, (cx1, cy1), (cx2, cy2), (56, 189, 248), 2)
                if engine.cfg.auto_track:
                    cv2.putText(vis, "AUTO-LOCK", (cx1 + 4, max(14, cy1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (56, 189, 248), 1, cv2.LINE_AA)

                # Draw Tracked Fluid Drops (Neon Green)
                for x1, y1, x2, y2, cf in st.boxes:
                    bx1 = int((rx0 + x1 * rw) * sx)
                    by1 = int((ry0 + y1 * rh) * sy)
                    bx2 = int((rx0 + x2 * rw) * sx)
                    by2 = int((ry0 + y2 * rh) * sy)
                    cv2.rectangle(vis, (bx1, by1), (bx2, by2), (102, 255, 0), 2)
                    cv2.putText(vis, f"{cf:.2f}", (bx2 + 4, by1 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (102, 255, 0), 1, cv2.LINE_AA)

                _, jpg = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), 76])
                b64_img = base64.b64encode(jpg.tobytes()).decode("ascii")

                if frame_idx - last_db_log_idx >= int(fps * 2):
                    last_db_log_idx = frame_idx
                    log_db = SessionLocal()
                    try:
                        bed_check = log_db.query(InfusionBed).filter(InfusionBed.id == bed_id).first()
                        if bed_check and bed_check.lifecycle_state != "VACANT":
                            log_db.add(TelemetryLog(
                                bed_id=bed_id,
                                total_drops=st.total_drops,
                                dpm=st.dpm,
                                ml_hr=st.ml_hr,
                                ml_infused=st.ml_infused,
                                status=st.status,
                                confidence=st.confidence,
                                reason=st.reason,
                            ))
                            if st.status in ("FAST", "SLOW", "STOPPED_OR_OCCLUDED"):
                                existing_open = (
                                    log_db.query(Alert)
                                    .filter(
                                        Alert.bed_id == bed_id,
                                        Alert.alert_type == st.status,
                                        Alert.acknowledged == False,
                                    )
                                    .first()
                                )
                                if not existing_open:
                                    sev = "CRITICAL" if st.status in ("FAST", "STOPPED_OR_OCCLUDED") else "WARNING"
                                    log_db.add(Alert(
                                        bed_id=bed_id,
                                        severity=sev,
                                        alert_type=st.status,
                                        message=xai["action_required"],
                                    ))
                            log_db.commit()
                    finally:
                        log_db.close()

                await self.broadcast_bed_event(bed_id, {
                    "type": "telemetry",
                    "frame": b64_img,
                    "telemetry": {
                        "t": round(st.t, 2),
                        "total_drops": st.total_drops,
                        "dpm": st.dpm,
                        "ml_hr": st.ml_hr,
                        "ml_infused": st.ml_infused,
                        "status": st.status,
                        "reason": st.reason,
                        "confidence": st.confidence,
                        "new_event": st.new_event,
                        "xai": xai,
                    },
                })

                frame_idx += 1
                await asyncio.sleep(frame_delay)
        except asyncio.CancelledError:
            pass
        finally:
            if cap is not None:
                cap.release()
            self.workers.pop(bed_id, None)
            end_db = SessionLocal()
            try:
                s = end_db.query(InfusionSession).filter(InfusionSession.id == session_id).first()
                if s:
                    s.ended_at = datetime.now(timezone.utc)
                    s.state = "COMPLETED"
                    end_db.commit()
            finally:
                end_db.close()


sync_hub = BedSyncHub()


def _validate_box(vals: List[float], name: str) -> List[float]:
    if len(vals) != 4 or any(not (0.0 <= v <= 1.0) for v in vals):
        raise ValueError(f"{name} must contain 4 normalized coordinates between 0.0 and 1.0")
    if vals[2] <= vals[0] + 0.02 or vals[3] <= vals[1] + 0.02:
        raise ValueError(f"{name} must satisfy x1 > x0 + 0.02 and y1 > y0 + 0.02")
    return vals


class PatientCreate(BaseModel):
    mrn: str = Field(..., min_length=2, max_length=40)
    full_name: str = Field(..., min_length=2, max_length=100)
    age_years: int = Field(..., ge=0, le=120)
    weight_kg: float = Field(..., gt=0.5, le=350.0)
    is_pediatric: bool = False
    ward_notes: str = ""


class OrderCheckRequest(BaseModel):
    prescribed_dpm: Optional[float] = Field(default=None, gt=0, le=300)
    drop_factor: float = Field(default=20.0, gt=0)
    target_volume_ml: float = Field(default=500.0, gt=0, le=5000)
    target_duration_min: float = Field(default=240.0, gt=0, le=10080)
    patient_id: Optional[int] = None


class BedCreate(BaseModel):
    bed_code: str = Field(..., min_length=2, max_length=32)
    ward_name: str = "ICU-A"
    patient_id: Optional[int] = None
    fluid_type: str = "Normal Saline 0.9%"
    target_volume_ml: float = Field(default=500.0, gt=0, le=5000)
    target_duration_min: float = Field(default=240.0, gt=0, le=10080)
    drop_factor: float = Field(default=20.0, gt=0)
    prescribed_dpm: Optional[float] = Field(default=60.0, gt=0, le=300)
    tolerance: float = Field(default=0.25, gt=0.01, le=0.9)
    video_source: Optional[str] = None
    roi: Optional[List[float]] = None
    corridor: Optional[List[float]] = None

    @field_validator("roi")
    @classmethod
    def check_roi(cls, v):
        return _validate_box(v, "roi") if v is not None else v

    @field_validator("corridor")
    @classmethod
    def check_corridor(cls, v):
        return _validate_box(v, "corridor") if v is not None else v


class BedUpdate(BaseModel):
    ward_name: Optional[str] = None
    patient_id: Optional[int] = None
    fluid_type: Optional[str] = None
    target_volume_ml: Optional[float] = Field(default=None, gt=0, le=5000)
    target_duration_min: Optional[float] = Field(default=None, gt=0, le=10080)
    drop_factor: Optional[float] = Field(default=None, gt=0)
    prescribed_dpm: Optional[float] = Field(default=None, gt=0, le=300)
    clear_prescribed_dpm: bool = False
    tolerance: Optional[float] = Field(default=None, gt=0.01, le=0.9)
    video_source: Optional[str] = None
    roi: Optional[List[float]] = None
    corridor: Optional[List[float]] = None
    auto_track: Optional[bool] = True
    lifecycle_state: Optional[str] = None
    version_id: int

    @field_validator("roi")
    @classmethod
    def check_roi(cls, v):
        return _validate_box(v, "roi") if v is not None else v

    @field_validator("corridor")
    @classmethod
    def check_corridor(cls, v):
        return _validate_box(v, "corridor") if v is not None else v


@app.get("/")
def serve_ui():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "model_ready": MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 0,
        "defaults": DEFAULTS,
    }


@app.post("/api/auth/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    token = create_access_token({"sub": user.username, "role": user.role})
    return {
        "access_token": token,
        "token_type": "bearer",
        "username": user.username,
        "role": user.role,
        "full_name": user.full_name,
    }


@app.get("/api/auth/me")
def get_me(user: User = Depends(get_current_user)):
    return {"id": user.id, "username": user.username, "full_name": user.full_name, "role": user.role}


@app.get("/api/archives")
def list_archives(user: User = Depends(get_current_user)):
    files = sorted([p for p in ARCHIVES_DIR.glob("*.pdf")], key=lambda x: x.stat().st_mtime, reverse=True)
    return [
        {
            "filename": f.name,
            "size_kb": round(f.stat().st_size / 1024, 1),
            "created_at": datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc).isoformat(),
        }
        for f in files
    ]


@app.get("/api/archives/{filename}")
def download_archive_pdf(filename: str, user: User = Depends(get_current_user)):
    if "/" in filename or "\\" in filename or ".." in filename or not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=422, detail="Invalid archive filename")
    target = (ARCHIVES_DIR / filename).resolve()
    if target.parent != ARCHIVES_DIR or not target.is_file():
        raise HTTPException(status_code=404, detail="Archive PDF not found")
    return FileResponse(target, media_type="application/pdf", filename=filename)


@app.get("/api/videos")
def list_videos(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    vids = sorted([p.name for p in VIDEOS_DIR.glob("*.mp4")])
    dynamic_bed_video = dict(BED_VIDEO_MAP)
    dynamic_video_bed = dict(VIDEO_BED_MAP)
    for b in db.query(InfusionBed).all():
        if b.lifecycle_state != "VACANT":
            dynamic_bed_video[b.bed_code] = b.video_source
            dynamic_video_bed[b.video_source] = b.bed_code
    return {
        "videos": vids,
        "presets": VIDEO_PRESETS,
        "bed_video_map": dynamic_bed_video,
        "video_bed_map": dynamic_video_bed,
    }


@app.delete("/api/videos/{video_name}")
async def delete_video(
    video_name: str,
    x_security_code: Optional[str] = Header(default=None),
    db: Session = Depends(get_db),
    user: User = Depends(require_role("nurse", "doctor", "admin")),
):
    _verify_security_code(x_security_code)

    if "/" in video_name or "\\" in video_name or ".." in video_name or not video_name.lower().endswith(".mp4"):
        raise HTTPException(status_code=422, detail="Invalid video filename")

    target_path = (VIDEOS_DIR / video_name).resolve()
    if target_path.parent != VIDEOS_DIR:
        raise HTTPException(status_code=422, detail="Invalid file path")

    remaining_vids = sorted([p.name for p in VIDEOS_DIR.glob("*.mp4") if p.name != video_name])
    fallback_src = remaining_vids[0] if remaining_vids else "webcam:0"

    affected_beds = db.query(InfusionBed).filter(InfusionBed.video_source == video_name).all()
    for b in affected_beds:
        preset = _get_preset_for_video(fallback_src)
        b.video_source = fallback_src
        b.roi_json = json.dumps(preset["roi"])
        b.corridor_json = json.dumps(preset["corridor"])
        b.version_id += 1
        b.updated_at = datetime.now(timezone.utc)

    db.commit()

    if target_path.exists():
        target_path.unlink()

    record_audit(
        db,
        user.username,
        user.role,
        "DELETE_VIDEO",
        "VideoFile",
        0,
        f"Deleted video {video_name} (reassigned {len(affected_beds)} bed(s) to {fallback_src})",
    )

    for b in affected_beds:
        db.refresh(b)
        serialized = _serialize_bed(b)
        sync_hub.source_reload_flags.add(b.id)
        sync_hub.live_configs[b.id] = serialized
        await sync_hub.broadcast_bed_event(b.id, {"type": "bed_updated", "bed": serialized, "updated_by": user.username})

    return {"status": "deleted", "video": video_name, "reassigned_beds": len(affected_beds)}


@app.post("/api/clinical/validate-order")
def validate_order(payload: OrderCheckRequest, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    patient = db.query(Patient).filter(Patient.id == payload.patient_id).first() if payload.patient_id else None
    ev = evaluate_infusion_order(
        prescribed_dpm=payload.prescribed_dpm,
        drop_factor=payload.drop_factor,
        target_volume_ml=payload.target_volume_ml,
        target_duration_min=payload.target_duration_min,
        patient_weight_kg=patient.weight_kg if patient else None,
        is_pediatric=patient.is_pediatric if patient else False,
    )
    return {
        "feasible": ev.feasible,
        "risk_level": ev.risk_level,
        "computed_dpm": ev.computed_dpm,
        "computed_ml_hr": ev.computed_ml_hr,
        "explanation": ev.explanation,
        "violated_constraints": ev.violated_constraints,
        "recommended_adjustment": ev.recommended_adjustment,
    }


@app.get("/api/patients")
def list_patients(q: Optional[str] = None, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    query = db.query(Patient)
    if q:
        like = f"%{q.strip()}%"
        query = query.filter(or_(Patient.full_name.ilike(like), Patient.mrn.ilike(like)))
    return [
        {
            "id": p.id,
            "mrn": p.mrn,
            "full_name": p.full_name,
            "age_years": p.age_years,
            "weight_kg": p.weight_kg,
            "is_pediatric": p.is_pediatric,
            "ward_notes": p.ward_notes,
        }
        for p in query.order_by(Patient.id.asc()).all()
    ]


@app.post("/api/patients", status_code=201)
def create_patient(payload: PatientCreate, db: Session = Depends(get_db), user: User = Depends(require_role("nurse", "doctor", "admin"))):
    if db.query(Patient).filter(Patient.mrn == payload.mrn).first():
        raise HTTPException(status_code=400, detail="Patient MRN already exists")
    p = Patient(**payload.model_dump())
    db.add(p)
    db.commit()
    db.refresh(p)
    record_audit(db, user.username, user.role, "CREATE_PATIENT", "Patient", p.id, f"Registered {p.mrn}")
    return {"id": p.id, "mrn": p.mrn}


def _serialize_bed(b: InfusionBed) -> dict:
    return {
        "id": b.id,
        "bed_code": b.bed_code,
        "ward_name": b.ward_name,
        "patient_id": b.patient_id,
        "patient_name": b.patient.full_name if b.patient else "VACANT / Unassigned",
        "patient_mrn": b.patient.mrn if b.patient else None,
        "is_pediatric": b.patient.is_pediatric if b.patient else False,
        "fluid_type": b.fluid_type,
        "target_volume_ml": b.target_volume_ml,
        "target_duration_min": b.target_duration_min,
        "drop_factor": b.drop_factor,
        "prescribed_dpm": b.prescribed_dpm,
        "tolerance": b.tolerance,
        "video_source": b.video_source,
        "roi": json.loads(b.roi_json),
        "corridor": json.loads(b.corridor_json),
        "lifecycle_state": b.lifecycle_state,
        "version_id": b.version_id,
    }


@app.get("/api/beds")
def list_beds(
    search: Optional[str] = Query(default=None),
    ward: Optional[str] = Query(default=None),
    state: Optional[str] = Query(default=None),
    sort_by: str = Query(default="bed_code", pattern="^(bed_code|ward_name|updated_at|prescribed_dpm)$"),
    order: str = Query(default="asc", pattern="^(asc|desc)$"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    query = db.query(InfusionBed).outerjoin(Patient)
    if search:
        like = f"%{search.strip()}%"
        query = query.filter(
            or_(
                InfusionBed.bed_code.ilike(like),
                InfusionBed.fluid_type.ilike(like),
                Patient.full_name.ilike(like),
                Patient.mrn.ilike(like),
            )
        )
    if ward:
        query = query.filter(InfusionBed.ward_name == ward)
    if state:
        query = query.filter(InfusionBed.lifecycle_state == state)

    sort_col = getattr(InfusionBed, sort_by)
    query = query.order_by(sort_col.desc() if order == "desc" else sort_col.asc())
    return [_serialize_bed(b) for b in query.all()]


@app.get("/api/beds/by-video/{video_name}")
def get_bed_by_video(video_name: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    bed = db.query(InfusionBed).filter(InfusionBed.video_source == video_name, InfusionBed.lifecycle_state != "VACANT").first()
    if not bed:
        mapped_code = VIDEO_BED_MAP.get(video_name)
        if mapped_code:
            bed = db.query(InfusionBed).filter(InfusionBed.bed_code == mapped_code, InfusionBed.lifecycle_state != "VACANT").first()
    if not bed:
        raise HTTPException(status_code=404, detail=f"No active bed assigned to video: {video_name}")
    return _serialize_bed(bed)


@app.get("/api/ward/summary")
def ward_aggregation(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    total_beds = db.query(func.count(InfusionBed.id)).scalar() or 0
    running_beds = db.query(func.count(InfusionBed.id)).filter(InfusionBed.lifecycle_state == "RUNNING").scalar() or 0
    unacked_alerts = db.query(func.count(Alert.id)).filter(Alert.acknowledged == False).scalar() or 0
    total_drops = db.query(func.coalesce(func.sum(TelemetryLog.total_drops), 0)).scalar() or 0
    by_ward = db.query(InfusionBed.ward_name, func.count(InfusionBed.id)).group_by(InfusionBed.ward_name).all()
    return {
        "total_beds": total_beds,
        "running_beds": running_beds,
        "unacknowledged_alerts": unacked_alerts,
        "logged_drop_events": total_drops,
        "beds_by_ward": {w: c for w, c in by_ward},
    }


@app.post("/api/beds", status_code=201)
async def create_or_admit_bed(
    payload: BedCreate,
    db: Session = Depends(get_db),
    user: User = Depends(require_role("nurse", "doctor", "admin")),
):
    clean_code = payload.bed_code.strip().upper()
    existing_bed = db.query(InfusionBed).filter(InfusionBed.bed_code == clean_code).first()
    if existing_bed and existing_bed.lifecycle_state != "VACANT" and existing_bed.patient_id is not None:
        raise HTTPException(
            status_code=409,
            detail=f"Bed '{clean_code}' is currently occupied by {existing_bed.patient.full_name}. Discharge or vacate it first before admitting a new patient.",
        )

    available_vids = sorted([p.name for p in VIDEOS_DIR.glob("*.mp4")])
    fallback_vid = available_vids[0] if available_vids else "webcam:0"
    chosen_video = payload.video_source or BED_VIDEO_MAP.get(clean_code) or fallback_vid
    validate_safe_video_source(chosen_video)
    preset = _get_preset_for_video(chosen_video)
    chosen_roi = payload.roi if payload.roi is not None else preset["roi"]
    chosen_corridor = payload.corridor if payload.corridor is not None else preset["corridor"]

    patient = None
    if payload.patient_id is not None:
        patient = db.query(Patient).filter(Patient.id == payload.patient_id).first()
        if not patient:
            raise HTTPException(status_code=404, detail=f"Patient id {payload.patient_id} not found")

    ev = evaluate_infusion_order(
        prescribed_dpm=payload.prescribed_dpm,
        drop_factor=payload.drop_factor,
        target_volume_ml=payload.target_volume_ml,
        target_duration_min=payload.target_duration_min,
        patient_weight_kg=patient.weight_kg if patient else None,
        is_pediatric=patient.is_pediatric if patient else False,
    )
    if not ev.feasible:
        raise HTTPException(
            status_code=422,
            detail={
                "message": ev.explanation,
                "violated_constraints": ev.violated_constraints,
                "recommended_adjustment": ev.recommended_adjustment,
            },
        )

    if existing_bed:
        existing_bed.ward_name = payload.ward_name
        existing_bed.patient_id = payload.patient_id
        existing_bed.fluid_type = payload.fluid_type
        existing_bed.target_volume_ml = payload.target_volume_ml
        existing_bed.target_duration_min = payload.target_duration_min
        existing_bed.drop_factor = payload.drop_factor
        existing_bed.prescribed_dpm = payload.prescribed_dpm
        existing_bed.tolerance = payload.tolerance
        existing_bed.video_source = chosen_video
        existing_bed.roi_json = json.dumps(chosen_roi)
        existing_bed.corridor_json = json.dumps(chosen_corridor)
        existing_bed.lifecycle_state = "RUNNING"
        existing_bed.version_id += 1
        existing_bed.updated_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(existing_bed)
        record_audit(db, user.username, user.role, "ADMIT_VACANT_BED", "InfusionBed", existing_bed.id, f"Admitted new patient to vacant {existing_bed.bed_code}")
        serialized = _serialize_bed(existing_bed)
        sync_hub.source_reload_flags.add(existing_bed.id)
        sync_hub.live_configs[existing_bed.id] = serialized
        return serialized

    bed = InfusionBed(
        bed_code=clean_code,
        ward_name=payload.ward_name,
        patient_id=payload.patient_id,
        fluid_type=payload.fluid_type,
        target_volume_ml=payload.target_volume_ml,
        target_duration_min=payload.target_duration_min,
        drop_factor=payload.drop_factor,
        prescribed_dpm=payload.prescribed_dpm,
        tolerance=payload.tolerance,
        video_source=chosen_video,
        roi_json=json.dumps(chosen_roi),
        corridor_json=json.dumps(chosen_corridor),
        lifecycle_state="RUNNING",
    )
    db.add(bed)
    db.commit()
    db.refresh(bed)
    record_audit(db, user.username, user.role, "CREATE_BED", "InfusionBed", bed.id, f"Created {bed.bed_code} with {chosen_video}")
    return _serialize_bed(bed)


@app.post("/api/beds/{bed_id}/discharge")
async def discharge_and_vacate_bed(
    bed_id: int,
    x_security_code: Optional[str] = Header(default=None),
    db: Session = Depends(get_db),
    user: User = Depends(require_role("nurse", "doctor", "admin")),
):
    _verify_security_code(x_security_code)

    bed = db.query(InfusionBed).filter(InfusionBed.id == bed_id).first()
    if not bed:
        raise HTTPException(status_code=404, detail="Bed not found")
    if bed.lifecycle_state == "VACANT":
        raise HTTPException(status_code=400, detail=f"{bed.bed_code} is already VACANT.")

    pdf_file = _generate_discharge_pdf(db, bed, user.username, "DISCHARGED_TO_VACANT_BED")
    await sync_hub.stop_bed_worker(bed_id)

    db.query(TelemetryLog).filter(TelemetryLog.bed_id == bed_id).delete(synchronize_session=False)
    db.query(Alert).filter(Alert.bed_id == bed_id).delete(synchronize_session=False)

    old_patient = bed.patient.full_name if bed.patient else "Unassigned"
    bed.patient_id = None
    bed.prescribed_dpm = None
    bed.fluid_type = "None (Vacant Slot)"
    bed.lifecycle_state = "VACANT"
    bed.version_id += 1
    bed.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(bed)

    record_audit(
        db,
        user.username,
        user.role,
        "DISCHARGE_BED",
        "InfusionBed",
        bed.id,
        f"Discharged {old_patient} from {bed.bed_code}; archived PDF {pdf_file}",
    )
    return {
        "status": "discharged_and_vacant",
        "bed": _serialize_bed(bed),
        "archive_pdf": pdf_file,
    }


@app.post("/api/beds/{bed_id}/upload-video")
async def upload_bed_video(
    bed_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    user: User = Depends(require_role("nurse", "doctor", "admin")),
):
    bed = db.query(InfusionBed).filter(InfusionBed.id == bed_id).first()
    if not bed:
        raise HTTPException(status_code=404, detail="Bed not found")

    raw_name = Path(file.filename or "").name
    if not raw_name.lower().endswith(".mp4"):
        raise HTTPException(
            status_code=422,
            detail="Rejected: File must end with .mp4 extension (e.g. 12076506_1080_1920_60fps.mp4)",
        )

    safe_stem = re.sub(r"[^a-zA-Z0-9_() -]", "_", Path(raw_name).stem).strip()[:70]
    if not safe_stem:
        raise HTTPException(status_code=422, detail="Invalid video filename")
    final_filename = f"{safe_stem}.mp4"
    dest_path = (VIDEOS_DIR / final_filename).resolve()

    if dest_path.parent != VIDEOS_DIR:
        raise HTTPException(status_code=422, detail="Invalid target path")

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=422, detail="Uploaded video file is empty")
    if len(content) > 60 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Video exceeds 60 MB limit")

    dest_path.write_bytes(content)

    preset = _get_preset_for_video(final_filename)
    bed.video_source = final_filename
    bed.roi_json = json.dumps(preset["roi"])
    bed.corridor_json = json.dumps(preset["corridor"])
    bed.version_id += 1
    bed.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(bed)

    record_audit(db, user.username, user.role, "UPLOAD_BED_VIDEO", "InfusionBed", bed.id, f"Assigned {final_filename} to {bed.bed_code}")

    serialized = _serialize_bed(bed)
    sync_hub.source_reload_flags.add(bed_id)
    sync_hub.live_configs[bed_id] = serialized
    await sync_hub.broadcast_bed_event(bed_id, {"type": "bed_updated", "bed": serialized, "updated_by": user.username})
    return serialized


@app.patch("/api/beds/{bed_id}")
async def update_bed(bed_id: int, payload: BedUpdate, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    bed = db.query(InfusionBed).filter(InfusionBed.id == bed_id).first()
    if not bed:
        raise HTTPException(status_code=404, detail="Bed not found")

    provided = payload.model_dump(exclude_unset=True)
    if ("prescribed_dpm" in provided or payload.clear_prescribed_dpm) and user.role not in ("doctor", "admin"):
        raise HTTPException(status_code=403, detail="Only doctors or admins may modify prescribed DPM")

    updates = {}
    if "ward_name" in provided:
        updates["ward_name"] = payload.ward_name
    if "patient_id" in provided:
        if payload.patient_id is not None:
            target_patient = db.query(Patient).filter(Patient.id == payload.patient_id).first()
            if not target_patient:
                raise HTTPException(status_code=404, detail=f"Patient id {payload.patient_id} not found")
        updates["patient_id"] = payload.patient_id
    if "fluid_type" in provided:
        updates["fluid_type"] = payload.fluid_type
    if "target_volume_ml" in provided:
        updates["target_volume_ml"] = payload.target_volume_ml
    if "target_duration_min" in provided:
        updates["target_duration_min"] = payload.target_duration_min
    if "drop_factor" in provided:
        updates["drop_factor"] = payload.drop_factor
    if payload.clear_prescribed_dpm:
        updates["prescribed_dpm"] = None
    elif "prescribed_dpm" in provided:
        updates["prescribed_dpm"] = payload.prescribed_dpm
    if "tolerance" in provided:
        updates["tolerance"] = payload.tolerance

    if "lifecycle_state" in provided and payload.lifecycle_state:
        ok_tr, msg_tr = validate_lifecycle_transition(bed.lifecycle_state, payload.lifecycle_state)
        if not ok_tr:
            raise HTTPException(status_code=422, detail=msg_tr)
        updates["lifecycle_state"] = payload.lifecycle_state

    source_changed = False
    if "video_source" in provided and payload.video_source:
        validate_safe_video_source(payload.video_source)
        owner_bed = db.query(InfusionBed).filter(
            InfusionBed.video_source == payload.video_source,
            InfusionBed.lifecycle_state != "VACANT",
        ).first()
        if owner_bed and owner_bed.id != bed.id and len(provided) <= 2:
            return _serialize_bed(owner_bed)

        if payload.video_source != bed.video_source:
            source_changed = True
        updates["video_source"] = payload.video_source
        preset = _get_preset_for_video(payload.video_source)
        if "roi" not in provided:
            updates["roi_json"] = json.dumps(preset["roi"])
        if "corridor" not in provided:
            updates["corridor_json"] = json.dumps(preset["corridor"])

    if "roi" in provided and payload.roi is not None:
        updates["roi_json"] = json.dumps(payload.roi)
    if "corridor" in provided and payload.corridor is not None:
        updates["corridor_json"] = json.dumps(payload.corridor)

    eff_dpm = updates.get("prescribed_dpm", bed.prescribed_dpm)
    eff_df = updates.get("drop_factor", bed.drop_factor)
    eff_vol = updates.get("target_volume_ml", bed.target_volume_ml)
    eff_dur = updates.get("target_duration_min", bed.target_duration_min)
    eff_patient_id = updates.get("patient_id", bed.patient_id)
    patient = db.query(Patient).filter(Patient.id == eff_patient_id).first() if eff_patient_id else None

    if eff_dpm is not None:
        ev = evaluate_infusion_order(
            prescribed_dpm=eff_dpm,
            drop_factor=eff_df,
            target_volume_ml=eff_vol,
            target_duration_min=eff_dur,
            patient_weight_kg=patient.weight_kg if patient else None,
            is_pediatric=patient.is_pediatric if patient else False,
        )
        if not ev.feasible:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": ev.explanation,
                    "violated_constraints": ev.violated_constraints,
                    "recommended_adjustment": ev.recommended_adjustment,
                },
            )

    if not atomic_bed_update(db, bed_id, payload.version_id, updates):
        latest = db.query(InfusionBed).filter(InfusionBed.id == bed_id).first()
        raise HTTPException(
            status_code=409,
            detail={
                "message": f"Conflict: Bed modified concurrently (server v{latest.version_id} != your v{payload.version_id}). Auto-refreshing.",
                "latest_bed": _serialize_bed(latest),
            },
        )

    db.expire_all()
    updated_bed = db.query(InfusionBed).filter(InfusionBed.id == bed_id).first()
    record_audit(db, user.username, user.role, "UPDATE_BED", "InfusionBed", bed_id, f"Updated fields: {list(updates.keys())}")

    serialized = _serialize_bed(updated_bed)
    if "auto_track" in provided:
        serialized["auto_track"] = payload.auto_track
    if source_changed:
        sync_hub.source_reload_flags.add(bed_id)
    sync_hub.live_configs[bed_id] = serialized
    await sync_hub.broadcast_bed_event(bed_id, {"type": "bed_updated", "bed": serialized, "updated_by": user.username})
    return serialized


@app.delete("/api/beds/{bed_id}")
async def delete_bed(
    bed_id: int,
    x_security_code: Optional[str] = Header(default=None),
    db: Session = Depends(get_db),
    user: User = Depends(require_role("nurse", "doctor", "admin")),
):
    _verify_security_code(x_security_code)

    bed = db.query(InfusionBed).filter(InfusionBed.id == bed_id).first()
    if not bed:
        raise HTTPException(status_code=404, detail="Bed not found")

    pdf_file = _generate_discharge_pdf(db, bed, user.username, "PERMANENT_BED_CLOSURE_AND_DELETE")
    code = bed.bed_code
    await sync_hub.stop_bed_worker(bed_id)

    db.query(TelemetryLog).filter(TelemetryLog.bed_id == bed_id).delete(synchronize_session=False)
    db.query(Alert).filter(Alert.bed_id == bed_id).delete(synchronize_session=False)
    db.query(InfusionSession).filter(InfusionSession.bed_id == bed_id).delete(synchronize_session=False)
    db.delete(bed)
    db.commit()

    record_audit(db, user.username, user.role, "DELETE_BED", "InfusionBed", bed_id, f"Deleted bed {code}; archived PDF {pdf_file}")
    return {"status": "deleted", "bed_code": code, "archive_pdf": pdf_file}


@app.get("/api/alerts")
def list_alerts(
    acknowledged: Optional[bool] = None,
    bed_id: Optional[int] = None,
    limit: int = Query(default=30, ge=1, le=200),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    q = db.query(Alert)
    if acknowledged is not None:
        q = q.filter(Alert.acknowledged == acknowledged)
    if bed_id is not None:
        q = q.filter(Alert.bed_id == bed_id)
    rows = q.order_by(Alert.id.desc()).limit(limit).all()
    return [
        {
            "id": a.id,
            "bed_id": a.bed_id,
            "bed_code": a.bed.bed_code if a.bed else str(a.bed_id),
            "severity": a.severity,
            "alert_type": a.alert_type,
            "message": a.message,
            "acknowledged": a.acknowledged,
            "acknowledged_by": a.acknowledged_by,
            "created_at": a.created_at.isoformat(),
        }
        for a in rows
    ]


@app.post("/api/alerts/{alert_id}/ack")
def acknowledge_alert(alert_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    if alert.acknowledged:
        return {"status": "already_acknowledged", "id": alert.id, "acknowledged_by": alert.acknowledged_by}
    alert.acknowledged = True
    alert.acknowledged_by = user.username
    db.commit()
    record_audit(db, user.username, user.role, "ACK_ALERT", "Alert", alert_id, f"Acknowledged {alert.alert_type}")
    return {"status": "acknowledged", "id": alert.id}


@app.get("/api/beds/{bed_id}/logs")
def get_bed_logs(bed_id: int, limit: int = Query(default=30, ge=1, le=200), db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    rows = (
        db.query(TelemetryLog)
        .filter(TelemetryLog.bed_id == bed_id)
        .order_by(TelemetryLog.id.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "t": r.timestamp.isoformat(),
            "drops": r.total_drops,
            "dpm": r.dpm,
            "ml_hr": r.ml_hr,
            "ml_infused": r.ml_infused,
            "status": r.status,
            "confidence": r.confidence,
            "reason": r.reason,
        }
        for r in reversed(rows)
    ]


@app.get("/api/audit-logs")
def get_audit_logs(limit: int = Query(default=40, ge=1, le=200), db: Session = Depends(get_db), user: User = Depends(require_role("doctor", "admin"))):
    rows = db.query(AuditLog).order_by(AuditLog.id.desc()).limit(limit).all()
    return [
        {
            "id": r.id,
            "actor": r.actor,
            "role": r.role,
            "action": r.action,
            "target": f"{r.target_type}:{r.target_id}",
            "detail": r.detail,
            "timestamp": r.timestamp.isoformat(),
        }
        for r in rows
    ]


@app.websocket("/ws/stream/{bed_id}")
async def stream_bed_telemetry(websocket: WebSocket, bed_id: int, user: Optional[User] = Depends(get_ws_user)):
    if user is None:
        await websocket.close(code=4401, reason="Unauthorized WebSocket token")
        return

    await websocket.accept()
    sync_hub.subscribers.setdefault(bed_id, set()).add(websocket)
    sync_hub.ensure_worker(bed_id, user.username)

    try:
        while True:
            msg = await websocket.receive_text()
            cmd = json.loads(msg)
            if cmd.get("action") == "reset":
                sync_hub.reset_flags.add(bed_id)
                await sync_hub.broadcast_bed_event(bed_id, {"type": "counter_reset", "by": user.username})
            elif cmd.get("action") == "config":
                if "prescribed_dpm" in cmd and user.role not in ("doctor", "admin"):
                    await websocket.send_json({"type": "error", "detail": "Only doctors or admins may modify prescribed DPM"})
                    continue

                db = SessionLocal()
                try:
                    bed = db.query(InfusionBed).filter(InfusionBed.id == bed_id).first()
                    if not bed:
                        continue
                    eff_dpm = cmd.get("prescribed_dpm", bed.prescribed_dpm)
                    eff_df = float(cmd.get("drop_factor", bed.drop_factor))
                    if eff_dpm is not None:
                        ev = evaluate_infusion_order(
                            prescribed_dpm=eff_dpm,
                            drop_factor=eff_df,
                            target_volume_ml=bed.target_volume_ml,
                            target_duration_min=bed.target_duration_min,
                            patient_weight_kg=bed.patient.weight_kg if bed.patient else None,
                            is_pediatric=bed.patient.is_pediatric if bed.patient else False,
                        )
                        if not ev.feasible:
                            await websocket.send_json({
                                "type": "error",
                                "detail": ev.explanation,
                                "recommended_adjustment": ev.recommended_adjustment,
                            })
                            continue

                    cfg_payload = sync_hub.live_configs.get(bed_id, _serialize_bed(bed))
                    if "prescribed_dpm" in cmd:
                        cfg_payload["prescribed_dpm"] = cmd["prescribed_dpm"]
                    if "drop_factor" in cmd and float(cmd["drop_factor"]) > 0:
                        cfg_payload["drop_factor"] = float(cmd["drop_factor"])
                    if "roi" in cmd and len(cmd["roi"]) == 4:
                        cfg_payload["roi"] = cmd["roi"]
                    if "corridor" in cmd and len(cmd["corridor"]) == 4:
                        cfg_payload["corridor"] = cmd["corridor"]
                    if "auto_track" in cmd:
                        cfg_payload["auto_track"] = bool(cmd["auto_track"])
                    sync_hub.live_configs[bed_id] = cfg_payload
                finally:
                    db.close()
    except WebSocketDisconnect:
        pass
    finally:
        sync_hub.subscribers.get(bed_id, set()).discard(websocket)
