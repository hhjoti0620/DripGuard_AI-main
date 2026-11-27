"""
DripGuard AI - clinical_engine.py (restored proven counting logic)
Single source of truth for drop counting + telemetry.

- DripFlowEngine : detections (normalised boxes + timestamp) -> drops, dpm, mL/hr, status
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
    drop_factor: float = 20.0
    prescribed_dpm: Optional[float] = None
    tolerance: float = 0.25
    min_conf: float = 0.0
    corridor: Optional[Tuple[float, float, float, float]] = None
    auto_track: bool = False
    min_area: float = 0.0002
    max_area: float = 0.08
    min_aspect: float = 0.35
    max_aspect: float = 2.40
    tripwire_y: Optional[float] = None
    max_jump: float = 0.18
    max_missed_s: float = 0.12
    min_hits: int = 2
    reset_area_ratio: float = 0.50
    reset_jump_up: float = 0.055
    min_interval_s: float = 0.22
    refractory_frac: float = 0.40
    static_after_s: float = 8.0
    static_drift: float = 0.025
    rate_window_s: float = 20.0
    min_events_for_rate: int = 4
    startup_s: float = 10.0
    stop_floor_s: float = 5.0
    stop_mult: float = 3.0


@dataclass
class _Track:
    tid: int
    t_first: float
    t_last: float
    cx: float
    cy: float
    area: float
    max_area: float
    cy0: float
    hits: int = 1
    counted: bool = False
    static: bool = False


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
    def __init__(self, cfg: Optional[EngineConfig] = None):
        self.cfg = cfg or EngineConfig()
        self.reset()

    def reset(self) -> None:
        self.tracks: List[_Track] = []
        self.events: List[float] = []
        self._next_id = 0
        self.t0: Optional[float] = None
        self.last_t = 0.0
        self._dts: Deque[float] = deque(maxlen=30)

    def sync_base_corridor(self, corridor: Optional[Tuple[float, float, float, float]]) -> None:
        self.cfg.corridor = corridor

    def _gate(self, boxes):
        c = self.cfg
        out = []
        for x1, y1, x2, y2, conf in boxes:
            if conf < c.min_conf: continue
            w = max(x2 - x1, 1e-6); h = max(y2 - y1, 1e-6); area = w*h
            if area < c.min_area or area > c.max_area: continue
            aspect = w/h
            if aspect < c.min_aspect or aspect > c.max_aspect: continue
            cx, cy = (x1+x2)/2.0, (y1+y2)/2.0
            if c.corridor is not None:
                a, b, d, e = c.corridor
                if not (a <= cx <= d and b <= cy <= e): continue
            out.append((x1, y1, x2, y2, conf))
        return out

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

    def update(self, t: float, boxes: Sequence[Box]) -> FlowState:
        c = self.cfg
        if self.t0 is None:
            self.t0 = t
        elif t > self.last_t:
            self._dts.append(t - self.last_t)
        self.last_t = t

        frame_dt = float(np.median(self._dts)) if self._dts else 0.0
        miss_budget = max(c.max_missed_s, 2.2 * frame_dt)
        dets = self._gate(boxes)
        new_event = False

        pairs = []
        for ti, tr in enumerate(self.tracks):
            for di, (x1, y1, x2, y2, _) in enumerate(dets):
                ncx, ncy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                dist = float(np.hypot(ncx - tr.cx, ncy - tr.cy))
                dx = abs(ncx - tr.cx)
                if dist <= c.max_jump and dx <= c.max_jump * 0.65:
                    pairs.append((dist, ti, di))
        pairs.sort()

        used_t, used_d = set(), set()
        for dist, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            tr = self.tracks[ti]
            x1, y1, x2, y2, _ = dets[di]
            ncx, ncy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            narea = max((x2 - x1) * (y2 - y1), 1e-9)

            jumped_up = (tr.cy - ncy) > c.reset_jump_up
            shrank_sharp = (tr.hits >= 2) and (narea < tr.max_area * c.reset_area_ratio) and (ncy <= tr.cy + 0.02)

            if (jumped_up or shrank_sharp) and tr.hits >= c.min_hits and not tr.counted and not tr.static:
                if self._register(t):
                    new_event = True
                tr.t_first = t
                tr.t_last = t
                tr.cx, tr.cy, tr.area, tr.max_area, tr.cy0 = ncx, ncy, narea, narea, ncy
                tr.hits = 1
                tr.counted = False
                tr.static = False
                continue

            tr.cx, tr.cy, tr.area = ncx, ncy, narea
            tr.max_area = max(tr.max_area, narea)
            tr.t_last = t
            tr.hits += 1

            if c.tripwire_y is not None and not tr.counted and not tr.static:
                if tr.cy0 < c.tripwire_y <= ncy and tr.hits >= c.min_hits:
                    if self._register(t):
                        new_event = True
                    tr.counted = True

            area_ratio = tr.max_area / max(tr.area, 1e-9)
            if (t - tr.t_first) > c.static_after_s and abs(tr.cy - tr.cy0) < c.static_drift and area_ratio < 1.35:
                tr.static = True

        for di, (x1, y1, x2, y2, _) in enumerate(dets):
            if di in used_d:
                continue
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            a = max((x2 - x1) * (y2 - y1), 1e-9)
            self.tracks.append(_Track(self._next_id, t, t, cx, cy, a, a, cy))
            self._next_id += 1

        alive = []
        for tr in self.tracks:
            if (t - tr.t_last) > miss_budget:
                if not tr.static and not tr.counted and tr.hits >= c.min_hits:
                    if self._register(tr.t_last):
                        new_event = True
                continue
            alive.append(tr)
        self.tracks = alive

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

    def __init__(self, onnx_path: str, conf: float = 0.163, iou: float = 0.60, size: int = 640,
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
