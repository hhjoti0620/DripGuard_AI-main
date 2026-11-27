"""
DripGuard AI - clinical_engine.py
Single source of truth for drop counting + telemetry.

- DripFlowEngine : detections (normalised boxes + timestamp) -> drops, dpm, mL/hr, status
                  Drop = one pendant-drop DETACHMENT cycle at the nozzle (present -> gone), counted in real time.
- OnnxDropDetector : onnxruntime + numpy only (no ultralytics) -> deployable on CPU with Letterbox
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

Box = Tuple[float, float, float, float, float]  # x1, y1, x2, y2 (normalised 0..1), conf


@dataclass
class EngineConfig:
    drop_factor: float = 20.0               # gtt per mL (20 = macro, 60 = micro)
    prescribed_dpm: Optional[float] = None  # None -> no NORMAL/SLOW/FAST verdict
    tolerance: float = 0.25                 # +/- band around prescribed rate
    min_conf: float = 0.25                  # filter weak noise while keeping real drops
    corridor: Optional[Tuple[float, float, float, float]] = None  # x0,y0,x1,y1 normalised
    auto_track: bool = False                # manual control only
    max_area: float = 0.055                 # reject large non-drop blobs
    min_aspect: float = 0.25
    max_aspect: float = 2.40
    # --- pendant-drop cycle counter (v2) ---
    nms_iomin: float = 0.50                 # merge nested duplicate boxes (intersection / smaller box)
    warmup_s: float = 1.5                   # learn nozzle zone from topmost detections
    zone_half_y: float = 0.07               # +/- band around learned nozzle y (ROI-normalised)
    zone_half_x: float = 0.20               # +/- band around tracked drop x (handles hand-held drift)
    zone_follow: float = 0.30               # EMA gain so the zone follows camera shake
    present_min_frames: float = 1.5         # frames a drop must be seen to count as "forming"
    absent_min_frames: float = 1.8          # frames of absence to confirm "detached"
    min_grow_s: float = 0.10                # drop must hang >= this long before a detach counts
    min_interval_s: float = 0.20            # hard floor (<= 300 dpm); adaptive part below
    refractory_frac: float = 0.50           # adaptive refractory = frac * median interval
    static_after_s: float = 5.0
    static_drift: float = 0.02
    rate_window_s: float = 20.0
    min_events_for_rate: int = 4
    startup_s: float = 10.0
    stop_floor_s: float = 5.0
    stop_mult: float = 3.0


@dataclass
class FlowState:
    t: float
    total_drops: int
    dpm: Optional[float]
    ml_hr: Optional[float]
    ml_infused: float
    status: str
    reason: str
    confidence: str
    n_events_window: int
    boxes: List[Box] = field(default_factory=list)
    active_corridor: Optional[Tuple[float, float, float, float]] = None
    new_event: bool = False

    def as_dict(self) -> Dict:
        return asdict(self)


class DripFlowEngine:
    """Counts pendant-drop detachment cycles at the nozzle.

    Why not track-death counting: the detector fires on the hanging drop on ~every frame, so
    "box disappeared" fired on jitter/duplicates and a fixed 0.55 s refractory then decided the
    final count. Here one drop == one (present for >= min_grow_s) -> (absent) transition inside
    an adaptive nozzle zone, debounced in FRAMES so it works at 24, 30 and 60 fps.
    """

    def __init__(self, cfg: Optional[EngineConfig] = None):
        self.cfg = cfg or EngineConfig()
        self.reset()

    def reset(self) -> None:
        self.events: List[float] = []
        self.t0: Optional[float] = None
        self.last_t = 0.0
        self._dts: Deque[float] = deque(maxlen=30)
        self._warm_y: List[float] = []
        self._warm_x: List[float] = []
        self.zone_y: Optional[float] = None
        self.zone_x: Optional[float] = None
        self._present = False          # confirmed "drop hanging at nozzle"
        self._run_start: Optional[float] = None
        self._present_since: Optional[float] = None
        self._last_seen: Optional[float] = None
        self._frozen_until_t: float = -1.0

    def sync_base_corridor(self, corridor: Optional[Tuple[float, float, float, float]]) -> None:
        self.cfg.corridor = corridor

    # ---------- gating ----------
    def _gate(self, boxes: Sequence[Box]) -> List[Box]:
        c = self.cfg
        kept: List[Box] = []
        for x1, y1, x2, y2, conf in boxes:
            if conf < c.min_conf:
                continue
            w = max(x2 - x1, 1e-6)
            h = max(y2 - y1, 1e-6)
            if (w * h) > c.max_area:
                continue
            aspect = w / h
            if aspect < c.min_aspect or aspect > c.max_aspect:
                continue
            kept.append((x1, y1, x2, y2, conf))

        if c.corridor is not None:
            a, b, d, e = c.corridor
            kept = [
                box for box in kept
                if a <= (box[0] + box[2]) / 2.0 <= d and b <= (box[1] + box[3]) / 2.0 <= e
            ]
        return self._merge_nested(kept)

    def _merge_nested(self, boxes: List[Box]) -> List[Box]:
        """IoMin-NMS: one physical drop often yields 2-3 nested boxes of different size (IoU is low)."""
        keep: List[Box] = []
        for b in sorted(boxes, key=lambda x: -x[4]):
            dup = False
            ab = max((b[2] - b[0]) * (b[3] - b[1]), 1e-9)
            for k in keep:
                ix = max(0.0, min(b[2], k[2]) - max(b[0], k[0]))
                iy = max(0.0, min(b[3], k[3]) - max(b[1], k[1]))
                ak = max((k[2] - k[0]) * (k[3] - k[1]), 1e-9)
                if ix * iy / min(ab, ak) > self.cfg.nms_iomin:
                    dup = True
                    break
            if not dup:
                keep.append(b)
        return keep

    # ---------- events ----------
    def _median_dt(self) -> Optional[float]:
        if len(self.events) < 3:
            return None
        return float(np.median(np.diff(self.events[-9:])))

    def _register(self, t: float) -> bool:
        med = self._median_dt()
        refr = self.cfg.min_interval_s
        if med is not None:
            refr = max(refr, self.cfg.refractory_frac * med)
        if self.events and (t - self.events[-1]) < refr:
            return False
        self.events.append(t)
        return True

    def _zone_hit(self, dets: List[Box]) -> Optional[Box]:
        """Best detection inside the (learned) nozzle zone, or None."""
        if self.zone_y is None:
            return None
        c = self.cfg
        best, best_d = None, 1e9
        for b in dets:
            cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
            if abs(cy - self.zone_y) > c.zone_half_y:
                continue
            if self.zone_x is not None and abs(cx - self.zone_x) > c.zone_half_x:
                continue
            d = abs(cy - self.zone_y) + (abs(cx - self.zone_x) if self.zone_x is not None else 0.0)
            if d < best_d:
                best, best_d = b, d
        return best

    def update(self, t: float, boxes: Sequence[Box]) -> FlowState:
        c = self.cfg
        if self.t0 is None:
            self.t0 = t
        elif t > self.last_t:
            self._dts.append(t - self.last_t)
        self.last_t = t

        frame_dt = float(np.median(self._dts)) if self._dts else 1.0 / 30.0
        present_min_s = max(0.04, c.present_min_frames * frame_dt)
        absent_min_s = max(0.04, c.absent_min_frames * frame_dt)
        dets = self._gate(boxes)
        new_event = False

        # --- learn nozzle zone: topmost detection per frame during warm-up ---
        if self.zone_y is None:
            if dets:
                top = min(dets, key=lambda b: (b[1] + b[3]) / 2.0)
                self._warm_y.append((top[1] + top[3]) / 2.0)
                self._warm_x.append((top[0] + top[2]) / 2.0)
            if (t - self.t0) >= c.warmup_s and len(self._warm_y) >= 5:
                self.zone_y = float(np.median(self._warm_y))
                self._warm_y.clear()
                self.zone_x = float(np.median(self._warm_x))
                self._warm_x.clear()
            return self._state(t, dets, False)

        hit = self._zone_hit(dets)
        if hit is not None:
            cx, cy = (hit[0] + hit[2]) / 2.0, (hit[1] + hit[3]) / 2.0
            g = c.zone_follow
            self.zone_y += g * (cy - self.zone_y)       # follow hand-held camera shake
            self.zone_x += g * (cx - self.zone_x)
            self._last_seen = t
            if not self._present:
                if self._run_start is None:
                    self._run_start = t
                if (t - self._run_start) >= present_min_s:
                    self._present = True
                    self._present_since = self._run_start
        else:
            if not self._present:
                self._run_start = None                 # flicker before confirmation: restart
            elif self._last_seen is not None and (t - self._last_seen) >= absent_min_s:
                # drop that was hanging is gone -> DETACHED
                grew = (self._last_seen - (self._present_since or self._last_seen)) >= c.min_grow_s
                self._present = False
                self._run_start = None
                if grew and self._register(self._last_seen):
                    new_event = True

        return self._state(t, dets, new_event)

    def _rate(self, t: float) -> Tuple[Optional[float], int]:
        win = [e for e in self.events if e >= t - self.cfg.rate_window_s]
        n = len(win)
        if n < self.cfg.min_events_for_rate:
            return None, n
        span = win[-1] - win[0]
        if span <= 0:
            return None, n
        return 60.0 * (n - 1) / span, n

    def _confidence(self, n: int) -> str:
        if len(self.events) < 4:
            return "NONE"
        iv = np.diff(self.events[-10:])
        cv = float(np.std(iv) / max(np.mean(iv), 1e-9))
        if n >= 6 and cv < 0.20:
            return "HIGH"
        if cv < 0.40:
            return "MEDIUM"
        return "LOW"

    def _state(self, t: float, dets: List[Box], new_event: bool) -> FlowState:
        c = self.cfg
        dpm, n = self._rate(t)
        ml_hr = None if dpm is None else dpm * 60.0 / c.drop_factor
        elapsed = t - (self.t0 if self.t0 is not None else t)
        med = self._median_dt()
        stop_after = max(c.stop_floor_s, c.stop_mult * (med or 0.0))
        since_last = (t - self.events[-1]) if self.events else elapsed

        if not self.events and elapsed >= c.startup_s:
            status, reason = "NO_DROPS_DETECTED", "No drop seen since start - check camera/ROI and verify at bedside"
        elif self.events and since_last > stop_after:
            status, reason = "STOPPED_OR_OCCLUDED", f"No drop for {since_last:.1f}s - verify at bedside"
        elif dpm is None:
            status, reason = "MEASURING", f"{len(self.events)} drops, need {c.min_events_for_rate} in window"
        elif c.prescribed_dpm is None:
            status, reason = "MEASURING", "No prescribed rate set - verdict disabled"
        else:
            lo = c.prescribed_dpm * (1 - c.tolerance)
            hi = c.prescribed_dpm * (1 + c.tolerance)
            if dpm > hi:
                status, reason = "FAST", f"{dpm:.0f} dpm > {hi:.0f}"
            elif dpm < lo:
                status, reason = "SLOW", f"{dpm:.0f} dpm < {lo:.0f}"
            else:
                status, reason = "NORMAL", f"{dpm:.0f} dpm within {lo:.0f}-{hi:.0f}"

        return FlowState(
            t=t,
            total_drops=len(self.events),
            dpm=None if dpm is None else round(dpm, 1),
            ml_hr=None if ml_hr is None else round(ml_hr, 1),
            ml_infused=round(len(self.events) / c.drop_factor, 3),
            status=status,
            reason=reason,
            confidence=self._confidence(n),
            n_events_window=n,
            boxes=dets,
            active_corridor=c.corridor,
            new_event=new_event,
        )

    def session_dpm(self) -> Optional[float]:
        if len(self.events) < 2 or self.events[-1] <= self.events[0]:
            return None
        return 60.0 * (len(self.events) - 1) / (self.events[-1] - self.events[0])


def _nms(xyxy: np.ndarray, scores: np.ndarray, iou_thr: float) -> List[int]:
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(xyxy[i, 0], xyxy[rest, 0])
        yy1 = np.maximum(xyxy[i, 1], xyxy[rest, 1])
        xx2 = np.minimum(xyxy[i, 2], xyxy[rest, 2])
        yy2 = np.minimum(xyxy[i, 3], xyxy[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        a_i = (xyxy[i, 2] - xyxy[i, 0]) * (xyxy[i, 3] - xyxy[i, 1])
        a_r = (xyxy[rest, 2] - xyxy[rest, 0]) * (xyxy[rest, 3] - xyxy[rest, 1])
        iou = inter / np.maximum(a_i + a_r - inter, 1e-9)
        order = rest[iou <= iou_thr]
    return keep


class OnnxDropDetector:
    """onnxruntime-only detector with Ultralytics-exact letterbox preprocessing."""

    def __init__(self, onnx_path: str, conf: float = 0.25, iou: float = 0.60, size: int = 640,
                 providers: Sequence[str] = ("CPUExecutionProvider",), threads: int = 0):
        import onnxruntime as ort
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = int(threads)
        self.sess = ort.InferenceSession(str(onnx_path), sess_options=so, providers=list(providers))
        self.inp = self.sess.get_inputs()[0].name
        self.conf, self.iou, self.size = conf, iou, size
        self.provider = self.sess.get_providers()[0]

    def _letterbox(self, bgr: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
        import cv2
        h, w = bgr.shape[:2]
        r = min(self.size / h, self.size / w)
        nw, nh = int(round(w * r)), int(round(h * r))
        dw, dh = (self.size - nw) / 2.0, (self.size - nh) / 2.0

        if (w, h) != (nw, nh):
            bgr = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        padded = cv2.copyMakeBorder(bgr, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        im = padded[:, :, ::-1].astype(np.float32) / 255.0
        blob = np.ascontiguousarray(im.transpose(2, 0, 1)[None])
        return blob, r, dw, dh

    def detect(self, bgr: np.ndarray, max_det: int = 20) -> List[Box]:
        h, w = bgr.shape[:2]
        blob, r, dw, dh = self._letterbox(bgr)
        raw = self.sess.run(None, {self.inp: blob})[0]

        p = raw[0].T  # (N, 5)
        p = p[p[:, 4] >= self.conf]
        if p.shape[0] == 0:
            return []

        x1 = ((p[:, 0] - p[:, 2] / 2.0) - dw) / (r * w)
        y1 = ((p[:, 1] - p[:, 3] / 2.0) - dh) / (r * h)
        x2 = ((p[:, 0] + p[:, 2] / 2.0) - dw) / (r * w)
        y2 = ((p[:, 1] + p[:, 3] / 2.0) - dh) / (r * h)
        xyxy = np.clip(np.stack([x1, y1, x2, y2], axis=1), 0.0, 1.0)

        keep = _nms(xyxy, p[:, 4], self.iou)[:max_det]
        return [
            (float(xyxy[i, 0]), float(xyxy[i, 1]), float(xyxy[i, 2]), float(xyxy[i, 3]), float(p[i, 4]))
            for i in keep
        ]
