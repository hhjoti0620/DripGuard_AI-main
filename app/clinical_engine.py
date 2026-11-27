"""
DripGuard AI - clinical_engine.py
Single source of truth for drop counting + dynamic chamber-tracking telemetry.

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
    drop_factor: float = 20.0               # gtt per mL (20 = macro, 60 = micro)
    prescribed_dpm: Optional[float] = None  # None -> no NORMAL/SLOW/FAST verdict
    tolerance: float = 0.25                 # +/- band around prescribed rate
    min_conf: float = 0.45                  # high confidence floor to reject bubbles/glare/noise
    corridor: Optional[Tuple[float, float, float, float]] = None  # x0,y0,x1,y1 normalised
    auto_track: bool = True                 # dynamically follow swaying IV drip chamber
    min_area: float = 0.00025               # reject tiny micro-bubbles & sensor noise
    max_area: float = 0.055                 # reject large non-drop blobs (caps/meniscus/hand)
    min_aspect: float = 0.40                # reject narrow vertical streaks
    max_aspect: float = 1.90                # reject wide/flat bubble clusters or liquid menisci
    min_fall_dy: float = 0.012              # minimum downward y-travel required to confirm a real falling drop
    tripwire_y: Optional[float] = None      # normalised y; optional virtual tripwire
    max_jump: float = 0.18                  # max centroid jump between consecutive frames
    max_missed_s: float = 0.14              # fast track retirement at >=25fps
    min_hits: int = 2                       # require >=2 frames so 1-frame flickers never count
    reset_area_ratio: float = 0.50          # area shrinks < 50% of peak => drop detached from nozzle
    reset_jump_up: float = 0.055            # upward centroid jump => drop detached, new drop at nozzle
    min_interval_s: float = 0.25            # hard refractory window (~240 dpm max)
    refractory_frac: float = 0.45           # adaptive refractory = frac * median interval
    static_after_s: float = 6.0             # static wall bubble / glare filter
    static_drift: float = 0.025
    rate_window_s: float = 20.0
    min_events_for_rate: int = 4
    startup_s: float = 10.0
    stop_floor_s: float = 5.0               # STOPPED if silent for max(floor, stop_mult * median dt)
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
        self._missed_corridor_frames = 0
        self._ema_cx: Optional[float] = None
        self._ema_cy: Optional[float] = None
        self.dynamic_corridor: Optional[Tuple[float, float, float, float]] = self.cfg.corridor

    def sync_base_corridor(self, corridor: Optional[Tuple[float, float, float, float]]) -> None:
        self.cfg.corridor = corridor
        self.dynamic_corridor = corridor
        self._ema_cx = None
        self._ema_cy = None
        self._missed_corridor_frames = 0

    def _shape_and_conf_filter(self, boxes: Sequence[Box]) -> List[Box]:
        c = self.cfg
        out = []
        for x1, y1, x2, y2, conf in boxes:
            if conf < c.min_conf:
                continue
            w = max(x2 - x1, 1e-6)
            h = max(y2 - y1, 1e-6)
            area = w * h
            if area < c.min_area or area > c.max_area:
                continue
            aspect = w / h
            if aspect < c.min_aspect or aspect > c.max_aspect:
                continue
            out.append((x1, y1, x2, y2, conf))
        return out

    def _gate(self, boxes: Sequence[Box]) -> List[Box]:
        c = self.cfg
        valid_shapes = self._shape_and_conf_filter(boxes)
        active_corr = self.dynamic_corridor or c.corridor

        if active_corr is None:
            return valid_shapes

        a, b, d, e = active_corr
        in_corr = []
        for box in valid_shapes:
            cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
            if a <= cx <= d and b <= cy <= e:
                in_corr.append(box)

        if not c.auto_track:
            return in_corr

        # Dynamic Auto-Follow Logic:
        # 1) If high-confidence drops are inside corridor, smoothly follow chamber sway via EMA
        if in_corr:
            self._missed_corridor_frames = 0
            best = max(in_corr, key=lambda x: x[4])
            bcx = (best[0] + best[2]) / 2.0
            bcy = (best[1] + best[3]) / 2.0
            if self._ema_cx is None or self._ema_cy is None:
                self._ema_cx, self._ema_cy = bcx, bcy
            else:
                self._ema_cx = 0.85 * self._ema_cx + 0.15 * bcx
                self._ema_cy = 0.85 * self._ema_cy + 0.15 * bcy

            cw = max(0.32, d - a)
            ch = max(0.55, e - b)
            nx0 = float(np.clip(self._ema_cx - cw * 0.5, 0.02, 0.98 - cw))
            ny0 = float(np.clip(self._ema_cy - ch * 0.35, 0.02, 0.98 - ch))
            self.dynamic_corridor = (round(nx0, 3), round(ny0, 3), round(nx0 + cw, 3), round(ny0 + ch, 3))
            return in_corr

        # 2) If corridor has 0 hits for ~15 frames while high-confidence drops exist elsewhere, auto-relock
        if valid_shapes:
            self._missed_corridor_frames += 1
            if self._missed_corridor_frames >= 15:
                best = max(valid_shapes, key=lambda x: x[4])
                bcx = (best[0] + best[2]) / 2.0
                bcy = (best[1] + best[3]) / 2.0
                self._ema_cx, self._ema_cy = bcx, bcy
                cw = max(0.36, d - a)
                ch = max(0.60, e - b)
                nx0 = float(np.clip(bcx - cw * 0.5, 0.02, 0.98 - cw))
                ny0 = float(np.clip(bcy - ch * 0.30, 0.02, 0.98 - ch))
                self.dynamic_corridor = (round(nx0, 3), round(ny0, 3), round(nx0 + cw, 3), round(ny0 + ch, 3))
                self._missed_corridor_frames = 0
                return [best]
        else:
            self._missed_corridor_frames = 0

        return in_corr

    def _is_valid_drop_trajectory(self, tr: _Track) -> bool:
        """Rejects stationary/jittering bubbles and non-falling objects."""
        if tr.static or tr.counted or tr.hits < self.cfg.min_hits:
            return False
        net_fall_y = tr.cy - tr.cy0
        return net_fall_y >= self.cfg.min_fall_dy

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

        # 1) Nearest-centroid association with horizontal gating
        pairs = []
        for ti, tr in enumerate(self.tracks):
            for di, (x1, y1, x2, y2, _) in enumerate(dets):
                ncx, ncy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                dx = abs(ncx - tr.cx)
                dy = ncy - tr.cy
                dist = float(np.hypot(dx, dy))
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
            shrank_sharp = (tr.hits >= 2) and (narea < tr.max_area * c.reset_area_ratio) and (ncy <= tr.cy + 0.025)

            if (jumped_up or shrank_sharp) and self._is_valid_drop_trajectory(tr):
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
                if tr.cy0 < c.tripwire_y <= ncy and self._is_valid_drop_trajectory(tr):
                    if self._register(t):
                        new_event = True
                    tr.counted = True

            # Static bubble / reflection / jittering blob filter
            total_travel = float(np.hypot(tr.cx - tr.cx, tr.cy - tr.cy0))
            area_ratio = tr.max_area / max(tr.area, 1e-9)
            if (t - tr.t_first) > c.static_after_s and total_travel < c.static_drift and area_ratio < 1.35:
                tr.static = True

        # 2) Create new tracks for unmatched detections
        for di, (x1, y1, x2, y2, _) in enumerate(dets):
            if di in used_d:
                continue
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            a = max((x2 - x1) * (y2 - y1), 1e-9)
            self.tracks.append(_Track(self._next_id, t, t, cx, cy, a, a, cy))
            self._next_id += 1

        # 3) Retire tracks that vanished (confirm only if it moved downward like a real drop)
        alive = []
        for tr in self.tracks:
            if (t - tr.t_last) > miss_budget:
                if self._is_valid_drop_trajectory(tr):
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
            status, reason = "NO_DROPS_DETECTED", "No falling drop seen since start - verify chamber view or clamp"
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
            active_corridor=self.dynamic_corridor or c.corridor,
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

    def __init__(self, onnx_path: str, conf: float = 0.45, iou: float = 0.60, size: int = 640,
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
