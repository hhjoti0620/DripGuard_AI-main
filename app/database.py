import os
from datetime import datetime, timezone
from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey, Integer, String, create_engine, update
)
from sqlalchemy.orm import Session, declarative_base, relationship, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./dripguard.db")

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    full_name = Column(String, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(String, default="nurse", nullable=False)  # nurse | doctor | admin
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    mrn = Column(String, unique=True, index=True, nullable=False)
    full_name = Column(String, nullable=False)
    age_years = Column(Integer, nullable=False)
    weight_kg = Column(Float, nullable=False)
    is_pediatric = Column(Boolean, default=False)
    ward_notes = Column(String, default="")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    beds = relationship("InfusionBed", back_populates="patient")


class InfusionBed(Base):
    __tablename__ = "infusion_beds"

    id = Column(Integer, primary_key=True, index=True)
    bed_code = Column(String, unique=True, index=True, nullable=False)
    ward_name = Column(String, default="ICU-A", nullable=False)
    patient_id = Column(Integer, ForeignKey("patients.id", ondelete="SET NULL"), nullable=True)
    fluid_type = Column(String, default="Normal Saline 0.9%", nullable=False)
    target_volume_ml = Column(Float, default=500.0, nullable=False)
    target_duration_min = Column(Float, default=240.0, nullable=False)
    drop_factor = Column(Float, default=20.0, nullable=False)
    prescribed_dpm = Column(Float, nullable=True)
    tolerance = Column(Float, default=0.25, nullable=False)
    video_source = Column(String, default="15011120_1080_1920_30fps.mp4", nullable=False)
    roi_json = Column(String, default="[0.36, 0.36, 0.65, 0.72]", nullable=False)
    corridor_json = Column(String, default="[0.45, 0.36, 0.85, 0.72]", nullable=False)
    lifecycle_state = Column(String, default="ORDERED", nullable=False)
    version_id = Column(Integer, default=1, nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    patient = relationship("Patient", back_populates="beds")
    sessions = relationship("InfusionSession", back_populates="bed", cascade="all, delete-orphan")
    logs = relationship("TelemetryLog", back_populates="bed", cascade="all, delete-orphan")
    alerts = relationship("Alert", back_populates="bed", cascade="all, delete-orphan")


class InfusionSession(Base):
    __tablename__ = "infusion_sessions"

    id = Column(Integer, primary_key=True, index=True)
    bed_id = Column(Integer, ForeignKey("infusion_beds.id", ondelete="CASCADE"), nullable=False)
    started_by = Column(String, nullable=False)
    started_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    ended_at = Column(DateTime, nullable=True)
    state = Column(String, default="RUNNING", nullable=False)
    total_drops = Column(Integer, default=0)
    total_infused_ml = Column(Float, default=0.0)

    bed = relationship("InfusionBed", back_populates="sessions")


class TelemetryLog(Base):
    __tablename__ = "telemetry_logs"

    id = Column(Integer, primary_key=True, index=True)
    bed_id = Column(Integer, ForeignKey("infusion_beds.id", ondelete="CASCADE"), nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    total_drops = Column(Integer, nullable=False)
    dpm = Column(Float, nullable=True)
    ml_hr = Column(Float, nullable=True)
    ml_infused = Column(Float, nullable=False)
    status = Column(String, nullable=False)
    confidence = Column(String, nullable=False)
    reason = Column(String, nullable=False)

    bed = relationship("InfusionBed", back_populates="logs")


class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    bed_id = Column(Integer, ForeignKey("infusion_beds.id", ondelete="CASCADE"), nullable=False)
    severity = Column(String, nullable=False)
    alert_type = Column(String, nullable=False)
    message = Column(String, nullable=False)
    acknowledged = Column(Boolean, default=False, nullable=False)
    acknowledged_by = Column(String, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)

    bed = relationship("InfusionBed", back_populates="alerts")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    actor = Column(String, nullable=False)
    role = Column(String, nullable=False)
    action = Column(String, nullable=False)
    target_type = Column(String, nullable=False)
    target_id = Column(String, nullable=False)
    detail = Column(String, nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)


def atomic_bed_update(db: Session, bed_id: int, expected_version: int, values: dict) -> bool:
    values["version_id"] = expected_version + 1
    values["updated_at"] = datetime.now(timezone.utc)
    stmt = (
        update(InfusionBed)
        .where(InfusionBed.id == bed_id, InfusionBed.version_id == expected_version)
        .values(**values)
    )
    res = db.execute(stmt)
    db.commit()
    return res.rowcount == 1


def record_audit(db: Session, actor: str, role: str, action: str, target_type: str, target_id: str, detail: str) -> None:
    db.add(AuditLog(
        actor=actor,
        role=role,
        action=action,
        target_type=target_type,
        target_id=str(target_id),
        detail=detail,
    ))
    db.commit()


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()