# DripGuard AI — Clinical Bedside IV Infusion Telemetry System

**Track:** World Problem Solver (Patient Safety & Clinical Automation)  
**Declared Advanced Challenges:**  
- **A01 — Real-Time Multi-Clinician Synchronization:** Atomic compare-and-swap optimistic concurrency (`UPDATE ... WHERE id=:id AND version_id=:v`), WebSocket broadcast hub (`BedSyncHub`), and automatic `409 Conflict` state reconciliation.  
- **A02 — Clinical Decision Support & Explainable AI (XAI):** Physiological constraint solver (`app/clinical_rules.py`) rejecting infeasible fluid orders (e.g., gravity ceiling `>180 DPM` or pediatric overload) with live mathematical rationale on every telemetry frame.

## Engineering & Model Highlights
- **Video-Level Grouped Split:** Training, validation, and test partitions are grouped strictly by source video to avoid consecutive-frame memorization across splits.
- **CPU-Optimized ONNX Runtime:** Ultralytics-exact letterbox preprocessing and NMS implemented in pure NumPy + ONNX Runtime (`models/best.onnx`, static Opset 17) with non-blocking `asyncio.to_thread` execution.
- **Stateful Droplet Detachment Physics:** Filters out static chamber wall bubbles, plastic meniscus glare, and handheld shake using aspect-ratio gates, temporal hit tracking, and nozzle area-shrink detachment reset.
- **RBAC Security & Audit Trail:** JWT authentication with role-based permissions (`nurse`, `doctor`, `admin`), path-traversal protection, and persistent `AuditLog` tracking.

## Quick Start (Local PC / VS Code)

1. **Activate virtual environment & install dependencies:**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/Mac:
   source .venv/bin/activate

   pip install -r requirements.txt