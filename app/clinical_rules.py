from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

VALID_TRANSITIONS: Dict[str, List[str]] = {
    "ORDERED": ["RUNNING", "CANCELLED"],
    "RUNNING": ["PAUSED", "COMPLETED", "STOPPED"],
    "PAUSED": ["RUNNING", "COMPLETED", "STOPPED"],
    "STOPPED": ["ORDERED", "COMPLETED"],
    "COMPLETED": [],
    "CANCELLED": [],
}


@dataclass
class OrderEvaluation:
    feasible: bool
    risk_level: str  # SAFE | CAUTION | INFEASIBLE
    computed_dpm: float
    computed_ml_hr: float
    explanation: str
    violated_constraints: List[str] = field(default_factory=list)
    recommended_adjustment: Optional[Dict[str, Any]] = None


def validate_lifecycle_transition(current_state: str, target_state: str) -> Tuple[bool, str]:
    curr = (current_state or "").upper()
    nxt = (target_state or "").upper()
    if curr == nxt:
        return True, "State unchanged."
    allowed = VALID_TRANSITIONS.get(curr, [])
    if nxt not in allowed:
        return False, f"Invalid lifecycle transition from '{curr}' to '{nxt}'. Allowed: {allowed or 'None (terminal state)'}."
    return True, f"Transition {curr} -> {nxt} permitted."


def evaluate_infusion_order(
    prescribed_dpm: Optional[float],
    drop_factor: float,
    target_volume_ml: float,
    target_duration_min: float,
    patient_weight_kg: Optional[float] = None,
    is_pediatric: bool = False,
) -> OrderEvaluation:
    """
    A02 Constraint Solver & Conflict Detector:
    Checks physical gravity limits, volume-duration consistency, pediatric micro-drip rules,
    and weight-based physiological safety ceilings (Holliday-Segar).
    """
    violations: List[str] = []

    if drop_factor not in (10.0, 15.0, 20.0, 60.0):
        violations.append(f"Non-standard IV drop factor ({drop_factor} gtt/mL). Standard clinical sets are 10, 15, 20, or 60 gtt/mL.")

    duration_hr = max(target_duration_min / 60.0, 0.01)
    schedule_ml_hr = round(target_volume_ml / duration_hr, 2)
    schedule_dpm = round((schedule_ml_hr * drop_factor) / 60.0, 1)

    active_dpm = float(prescribed_dpm) if prescribed_dpm is not None else schedule_dpm
    active_ml_hr = round((active_dpm * 60.0) / drop_factor, 2)

    # 1. Pediatric chamber mismatch check
    if is_pediatric and drop_factor < 60.0:
        violations.append(
            f"Pediatric Safety Conflict: Pediatric patient requires a 60 gtt/mL micro-drip chamber, but {int(drop_factor)} gtt/mL macro-drip was ordered."
        )

    # 2. Schedule vs. Prescribed DPM consistency conflict (>25% mismatch)
    if prescribed_dpm is not None and schedule_dpm > 0:
        mismatch_pct = abs(active_dpm - schedule_dpm) / schedule_dpm * 100.0
        if mismatch_pct > 25.0:
            violations.append(
                f"Order Parameter Conflict: {target_volume_ml:.0f} mL over {target_duration_min:.0f} min requires {schedule_dpm:.1f} dpm "
                f"({schedule_ml_hr:.1f} mL/hr), conflicting with prescribed {active_dpm:.1f} dpm ({mismatch_pct:.0f}% discrepancy)."
            )

    # 3. Physical gravity & KVO limits
    if active_dpm < 3.0 or active_ml_hr < 5.0:
        violations.append(f"Occlusion / Vein Clotting Risk: {active_ml_hr:.1f} mL/hr ({active_dpm:.1f} dpm) is below minimum KVO rate (5 mL/hr).")
    if active_dpm > 250.0:
        violations.append(f"Mechanical Limit Exceeded: {active_dpm:.1f} dpm exceeds reliable optical/gravity chamber ceiling (250 dpm).")

    # 4. Weight-based physiological ceiling (Holliday-Segar + bolus cap)
    if patient_weight_kg is not None and patient_weight_kg > 0:
        if patient_weight_kg <= 10:
            maint_hr = 4.0 * patient_weight_kg
        elif patient_weight_kg <= 20:
            maint_hr = 40.0 + 2.0 * (patient_weight_kg - 10.0)
        else:
            maint_hr = 60.0 + 1.0 * (patient_weight_kg - 20.0)

        max_safe_hr = round(maint_hr * (2.0 if is_pediatric else 3.0), 1)
        if active_ml_hr > max_safe_hr:
            violations.append(
                f"Fluid Overload Hazard: {active_ml_hr:.1f} mL/hr exceeds weight-adjusted safety ceiling of {max_safe_hr:.1f} mL/hr "
                f"for a {patient_weight_kg:.1f} kg {'pediatric' if is_pediatric else 'adult'} patient."
            )
    else:
        hard_cap = 120.0 if is_pediatric else 500.0
        if active_ml_hr > hard_cap:
            violations.append(f"Hard Flow Ceiling Exceeded: {active_ml_hr:.1f} mL/hr exceeds {hard_cap:.0f} mL/hr clinical limit.")

    # Compute feasible recommendation if conflicts exist
    rec_df = 60.0 if is_pediatric else drop_factor
    safe_ml_hr = schedule_ml_hr
    if patient_weight_kg is not None and patient_weight_kg > 0:
        maint = 4.0 * patient_weight_kg if patient_weight_kg <= 10 else (40.0 + 2.0 * (patient_weight_kg - 10.0) if patient_weight_kg <= 20 else 60.0 + (patient_weight_kg - 20.0))
        safe_ml_hr = min(safe_ml_hr, round(maint * (2.0 if is_pediatric else 3.0), 1))
    else:
        safe_ml_hr = min(safe_ml_hr, 120.0 if is_pediatric else 500.0)

    safe_ml_hr = max(safe_ml_hr, 10.0)
    safe_dpm = round(min(max((safe_ml_hr * rec_df) / 60.0, 5.0), 200.0), 1)

    if violations:
        return OrderEvaluation(
            feasible=False,
            risk_level="INFEASIBLE",
            computed_dpm=active_dpm,
            computed_ml_hr=active_ml_hr,
            explanation=violations[0],
            violated_constraints=violations,
            recommended_adjustment={
                "drop_factor": rec_df,
                "prescribed_dpm": safe_dpm,
                "estimated_ml_hr": round((safe_dpm * 60.0) / rec_df, 1),
            },
        )

    risk = "CAUTION" if (is_pediatric and active_ml_hr > 80.0) else "SAFE"
    return OrderEvaluation(
        feasible=True,
        risk_level=risk,
        computed_dpm=active_dpm,
        computed_ml_hr=active_ml_hr,
        explanation=f"Verified feasible order: {active_dpm:.1f} dpm delivers {active_ml_hr:.1f} mL/hr ({int(drop_factor)} gtt/mL set).",
        violated_constraints=[],
        recommended_adjustment=None,
    )


def explain_telemetry_decision(
    status: str,
    dpm: Optional[float],
    prescribed_dpm: Optional[float],
    tolerance: float,
    ml_hr: Optional[float],
    drop_factor: float,
    raw_reason: str,
) -> Dict[str, Any]:
    """
    Produces structured Explainable AI (XAI) telemetry trace for the UI & Alert engine.
    Safely handles None values during initial MEASURING / CALIBRATING frames.
    """
    target_str = f"{prescribed_dpm:.1f} dpm" if prescribed_dpm else "Unset"
    has_rate = dpm is not None
    dev_pct = (
        round(((dpm - prescribed_dpm) / prescribed_dpm) * 100.0, 1)
        if (has_rate and prescribed_dpm and prescribed_dpm > 0)
        else 0.0
    )

    delta_str = f"{abs(dpm - prescribed_dpm):.1f}" if (has_rate and prescribed_dpm) else "0.0"
    action_map = {
        "NORMAL": "Maintain current roller clamp position.",
        "MEASURING": "Hold clamp steady while optical drop baseline stabilizes.",
        "CALIBRATING": "Hold clamp steady while optical drop baseline stabilizes.",
        "FAST": f"Throttle roller clamp immediately to reduce flow by {delta_str} dpm.",
        "SLOW": f"Open roller clamp slightly to restore target rate ({target_str}).",
        "STOPPED_OR_OCCLUDED": "CRITICAL: Inspect tubing for kinks, closed clamp, or empty IV bag immediately.",
    }

    live_rate_str = (
        f"{dpm:.1f} dpm ({ml_hr:.1f} mL/hr)"
        if (has_rate and ml_hr is not None)
        else "Measuring initial drop interval"
    )

    return {
        "formula": f"mL/hr = (DPM × 60) / {int(drop_factor)} gtt/mL",
        "decision_status": status,
        "measured_dpm": round(dpm, 1) if has_rate else None,
        "target_dpm": prescribed_dpm,
        "deviation_pct": dev_pct,
        "tolerance_band_pct": round(tolerance * 100.0, 0),
        "equivalent_ml_hr": round(ml_hr, 1) if ml_hr is not None else None,
        "drop_factor": int(drop_factor),
        "reasoning_chain": [
            f"Optical calibration: {int(drop_factor)} gtt/mL chamber.",
            f"Live rate {live_rate_str} vs target {target_str} (deviation {dev_pct:+.1f}%).",
            raw_reason,
        ],
        "action_required": action_map.get(status, raw_reason),
    }