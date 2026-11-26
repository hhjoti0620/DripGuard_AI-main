# DripGuard AI — Clinical Bedside IV Infusion Telemetry System

**Track:** World Problem Solver (Patient Safety & Clinical Automation)  
**Declared Advanced Challenges:**  
- **A01 — Real-Time Multi-Clinician Synchronization:** Atomic compare-and-swap optimistic concurrency (`UPDATE ... WHERE id=:id AND version_id=:v`), WebSocket broadcast hub (`BedSyncHub`), and automatic `409 Conflict` state reconciliation.  
- **A02 — Clinical Decision Support & Explainable AI (XAI):** Physiological constraint solver (`app/clinical_rules.py`) rejecting infeasible fluid orders (e.g., gravity ceiling `>180 DPM` or pediatric overload) with live mathematical rationale on every telemetry frame.

---

## Live Deployment, Repository & Demo Credentials

* **Live Demo URL:** `https://dripguardai-main-production.up.railway.app`
* **GitHub Repository:** `https://github.com/hhjoti0620/DripGuard_AI-main`
* **Colab Training & Export Notebook:** `notebooks/` (add your `.ipynb` training, evaluation, and ONNX export notebook inside the `notebooks/` folder)
* **Demo Credentials (Role-Based Access):**
  * **Doctor (`doctor1`):**
    * **Username:** `doctor1`
    * **Password:** `ChangeMe@2026`
    * **Role Access:** Full clinical & order-management permissions — register patients, create/update infusion beds, modify prescribed DPM, acknowledge alerts, view audit logs, and monitor live WebSocket telemetry.
  * **Nurse (`nurse1`):**
    * **Username:** `nurse1`
    * **Password:** `ChangeMe@2026`
    * **Role Access:** Bedside operational monitoring permissions — view live bed telemetry and video streams, reset drop/volume counters, update bedside parameters/lifecycle states, and acknowledge alerts; restricted from doctor/admin-only actions such as modifying prescribed DPM, registering patients, creating/deleting beds, or viewing audit logs.

---

## Dataset Source & Provenance (`infusionv2`)

* **Dataset Name & Version:** `infusionv2 - v2 2023-10-21 7:12am`
* **Roboflow Universe URL:** `https://universe.roboflow.com/andika-rizaldy/infusionv2`
* **Dataset Version URL:** `https://universe.roboflow.com/andika-rizaldy/infusionv2/dataset/2`
* **Provided By:** Roboflow user `andika-rizaldy`
* **License:** `CC BY 4.0`
* **Export Timestamp:** January 21, 2024 at 11:42 AM GMT
* **Total Images:** **9,587 images**
* **Annotation Format:** **YOLOv8**
* **Pre-processing Applied:** **Resize to 640x640 (Stretch)**
* **Augmentation Applied:** **None** (no image augmentation techniques were applied in the exported dataset)
* **Roboflow Notes & Resources:**
  * Roboflow end-to-end computer vision platform for team collaboration, image collection/organization, unstructured data search, annotation/dataset creation, model export/training/deployment, and active learning over time.
  * Roboflow Notebooks: `https://github.com/roboflow/notebooks`
  * Roboflow Universe: `https://universe.roboflow.com`
* **Dataset Configuration (`data.yaml`):**
  ```yaml
  train: ../train/images
  val: ../valid/images
  test: ../test/images

  nc: 3
  names: ['S. Anak', 'S. Dewasa', 'tetesan']

  roboflow:
    workspace: andika-rizaldy
    project: infusionv2
    version: 2
    license: CC BY 4.0
    url: [https://universe.roboflow.com/andika-rizaldy/infusionv2/dataset/2](https://universe.roboflow.com/andika-rizaldy/infusionv2/dataset/2)
  ```

---

## Sequential End-to-End Engineering Workflow

### Step 1: Dataset Sourcing & Provenance (`infusionv2`)
* Sourced `infusionv2 - v2 2023-10-21 7:12am` (`9,587` images, YOLOv8 annotations, `640x640` stretch resize, no augmentation applied during export, `CC BY 4.0` license) by `andika-rizaldy` from Roboflow Universe.

### Step 2: Data Audit, Video-Level Grouped Split & Class Mapping
* Audited the `9,587` images across source infusion videos in Google Colab to verify zero duplicate samples and prevent consecutive-frame or source-video data leakage across `train`, `valid`, and `test` partitions.
* Inspected class distribution, bounding box scale, training/validation loss and mAP curves, and confusion matrix behavior.
* Mapped raw dataset labels to clinical IV drip entities:
  * `S. Anak` $\rightarrow$ Pediatric drip chamber (`Pediatric_Chamber`, micro-drip `60 gtt/mL` protocol support)
  * `S. Dewasa` $\rightarrow$ Adult drip chamber (`Adult_Chamber`, macro-drip `20 gtt/mL` protocol support)
  * `tetesan` $\rightarrow$ Falling fluid drop (`Fluid_Drop`)

### Step 3: Model Training, Evaluation & ONNX Export (`notebooks/`)
* Trained `yolov8n` (`dripguard_yolov8n` / `dripguard_yolov8n-2`, `30` epochs, best checkpoint at epoch `22`) in Google Colab using the `infusionv2` dataset.
* Evaluated on validation and unseen test sets, achieving **0.9140 overall mAP50** on `best.onnx`.
* Verified ONNX drop detection, drops-per-minute (`dpm`), volumetric flow rate (`mL/hr`), and status overlays (such as `NORMAL INFUSION` vs. `WARNING: FAST FLOW >50 dpm`) on sample infusion videos (`demo_infusion_1.mp4`, `demo_infusion_2.mp4`, and `6248839-hd_1080_1920_30fps.mp4`).
* Exported the trained detector to `models/best.onnx` and stored the Google Colab training, evaluation, and export notebook under `notebooks/`.

### Step 4: Clinical Telemetry Backend, A01/A02 & Dashboard
* **FastAPI & WebSocket Hub (`app/main.py`):** Serves REST endpoints, real-time WebSocket streams (`/ws/stream/{bed_id}`), and startup database/sample seeding.
* **Authentication & RBAC (`app/auth.py`):** Handles password hashing, JWT authentication, and role-based access control (`nurse`, `doctor`, `admin`).
* **Database & Audit Logging (`app/database.py`):** SQLAlchemy ORM models for patients, beds, infusion sessions, telemetry records, alerts, and immutable `AuditLog` entries, with optimistic concurrency support (`version_id`).
* **Drop Detector & Clinical Rules (`app/clinical_engine.py`, `app/clinical_rules.py`):** Runs pure NumPy + ONNX Runtime (`models/best.onnx`) inference in non-blocking worker threads, tracks droplet detachment physics, computes `dpm` / `mL/hr` / infused `mL`, and evaluates physiological constraints and explainable clinical status rationale (`A02`).
* **Frontend Dashboard (`static/index.html`):** Bedside and ward telemetry UI with live stream overlays, drop/flow metrics, alert controls, and order checks.

### Step 5: Containerization & Railway Cloud Deployment
* Built a reproducible container image using `Dockerfile` (`python:3.11-slim`, OpenCV runtime libraries `libgl1` and `libglib2.0-0`, non-root user `dripuser`, persistent `/data` directory for SQLite storage, and `PORT=8000`).
* Deployed `https://github.com/hhjoti0620/DripGuard_AI-main` to **Railway** at `https://dripguardai-main-production.up.railway.app` so live monitoring and telemetry remain available independently of the local workstation.

---

## Engineering & Model Highlights
- **Video-Level Grouped Split:** Training, validation, and test partitions are grouped strictly by source video to avoid consecutive-frame memorization across splits.
- **CPU-Optimized ONNX Runtime:** Ultralytics-exact letterbox preprocessing and NMS implemented in pure NumPy + ONNX Runtime (`models/best.onnx`, static Opset 17) with non-blocking `asyncio.to_thread` execution.
- **Stateful Droplet Detachment Physics:** Filters out static chamber wall bubbles, plastic meniscus glare, and handheld shake using aspect-ratio gates, temporal hit tracking, and nozzle area-shrink detachment reset.
- **RBAC Security & Audit Trail:** JWT authentication with role-based permissions (`nurse`, `doctor`, `admin`), path-traversal protection, and persistent `AuditLog` tracking.

---

## Repository Structure

```text
DripGuard_AI-main/
├── app/
│   ├── auth.py              # Password hashing & JWT authentication / role checks
│   ├── clinical_engine.py   # ONNX drop detector & real-time drip flow engine
│   ├── clinical_rules.py    # Infusion order validation & decision explanations
│   ├── database.py          # SQLAlchemy ORM models, DB init & audit logging
│   └── main.py              # FastAPI endpoints, WebSocket hub & startup seeding
├── models/
│   └── best.onnx            # Exported YOLOv8n ONNX drop-detection weights
├── sample_videos/           # Sample IV infusion videos for live testing
├── static/
│   └── index.html           # Clinical mission control frontend
├── notebooks/               # Google Colab training & ONNX export notebook (.ipynb)
├── Dockerfile               # Container build & runtime configuration
├── engine_defaults.json     # Default detector confidence, IoU & input resolution
├── requirements.txt         # Python package dependencies
└── README.md                # Project documentation
```

---

## Quick Start (Local PC / VS Code)

1. **Activate virtual environment & install dependencies:**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/Mac:
   source .venv/bin/activate

   pip install -r requirements.txt
   ```

2. **Start the FastAPI server:**
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

3. **Open the application:**
   Visit `http://localhost:8000` (or the live Railway URL `https://dripguardai-main-production.up.railway.app`) and sign in with `doctor1` or `nurse1` using password `ChangeMe@2026`.

---

## Attribution & References

* **Dataset:** Andika Rizaldy, *infusionv2* (`v2 2023-10-21 7:12am`, exported January 21, 2024 at 11:42 AM GMT), Roboflow Universe — `https://universe.roboflow.com/andika-rizaldy/infusionv2` (`CC BY 4.0`).
* **Roboflow Resources:** `https://github.com/roboflow/notebooks` and `https://universe.roboflow.com`.
