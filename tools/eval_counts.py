"""Usage (from project root):
  python tools/eval_counts.py sample_videos/13187829_1080_1920_24fps.mp4 --roi 0.12 0.12 0.88 0.88 \
      --corridor 0.22 0.12 0.78 0.86 --truth 25
Runs the real ONNX detector + DripFlowEngine exactly like main.py and prints events vs your hand count."""
import argparse, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import cv2, numpy as np
from app.clinical_engine import DripFlowEngine, EngineConfig, OnnxDropDetector

ap = argparse.ArgumentParser()
ap.add_argument("video"); ap.add_argument("--truth", type=int)
ap.add_argument("--roi", type=float, nargs=4, default=[0.12, 0.12, 0.88, 0.88])
ap.add_argument("--corridor", type=float, nargs=4, default=[0.22, 0.12, 0.78, 0.86])
ap.add_argument("--conf", type=float, default=0.163)
a = ap.parse_args()

det = OnnxDropDetector("models/best.onnx", conf=a.conf)
eng = DripFlowEngine(EngineConfig(corridor=tuple(a.corridor), min_conf=0.0))
cap = cv2.VideoCapture(a.video); fps = cap.get(cv2.CAP_PROP_FPS) or 30.0; i = 0
while True:
    ok, fr = cap.read()
    if not ok: break
    H, W = fr.shape[:2]
    r = a.roi
    crop = fr[int(r[1]*H):int(r[3]*H), int(r[0]*W):int(r[2]*W)]
    eng.update(i / fps, det.detect(crop)); i += 1
ev = eng.events; iv = np.diff(ev)
print(f"frames={i} dur={i/fps:.1f}s engine_count={len(ev)}" + (f" truth={a.truth} err={len(ev)-a.truth:+d}" if a.truth else ""))
print("events(s):", [round(x, 2) for x in ev])
if len(iv) > 1: print(f"median interval={np.median(iv):.2f}s  CV={iv.std()/iv.mean():.2f}  ~{60/np.median(iv):.0f} dpm")
