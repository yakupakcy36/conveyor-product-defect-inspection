"""FastAPI application for Conveyor Product Defect Inspection & Pneumatic Sorting."""

import time
from contextlib import asynccontextmanager
from typing import Optional
import numpy as np
from fastapi import FastAPI, HTTPException, status

from src.config import DEFAULT_CONVEYOR_CONFIG
from src.generator import generate_conveyor_frame, inject_product_defect
from src.detector import ConveyorDefectDetector
from api.schemas import (
    InspectionBoxSchema,
    ConveyorInspectionResponse,
    SyntheticInspectionRequest,
    ConveyorBatchSimulationRequest,
    BatchSimulationResponse,
    HealthResponse,
)

detector: Optional[ConveyorDefectDetector] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector
    detector = ConveyorDefectDetector(config=DEFAULT_CONVEYOR_CONFIG)
    yield


app = FastAPI(
    title="Conveyor Product Defect Inspection & Pneumatic Sorting API",
    description=(
        "Production Automated Optical Inspection (AOI) service for packaging lines, "
        "detecting punctured seals, crushed corners, misaligned labels, and surface cracks "
        "with pneumatic ejection signaling."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["Diagnostics"])
def health():
    return HealthResponse(
        status="healthy",
        camera_ready=detector is not None,
        rejection_actuator_ready=detector is not None,
        target_ppm=DEFAULT_CONVEYOR_CONFIG.target_ppm,
        frame_resolution=f"{DEFAULT_CONVEYOR_CONFIG.image_width}x{DEFAULT_CONVEYOR_CONFIG.image_height}",
    )


@app.post("/inspect/synthetic", response_model=ConveyorInspectionResponse, tags=["Inspection"])
def inspect_synthetic(req: SyntheticInspectionRequest):
    """Generates synthetic product frame and executes inline defect detection."""
    if detector is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Detector unavailable")

    base = generate_conveyor_frame(
        width=DEFAULT_CONVEYOR_CONFIG.image_width,
        height=DEFAULT_CONVEYOR_CONFIG.image_height,
        seed=req.seed,
    )
    frame, _ = inject_product_defect(base, defect_type=req.defect_type, seed=req.seed)
    res = detector.inspect_frame(frame, unit_id=req.unit_id)

    boxes = [InspectionBoxSchema(**b.__dict__) for b in res.detected_defects]
    return ConveyorInspectionResponse(
        unit_id=res.unit_id,
        pass_status=res.pass_status,
        rejection_actuator_triggered=res.rejection_actuator_triggered,
        defect_severity=res.defect_severity,
        primary_defect_type=res.primary_defect_type,
        anomaly_score=res.anomaly_score,
        detected_defects=boxes,
        latency_ms=res.latency_ms,
    )


@app.post("/conveyor/stream-batch", response_model=BatchSimulationResponse, tags=["Conveyor Simulation"])
def simulate_conveyor_batch(req: ConveyorBatchSimulationRequest):
    """Simulates a continuous sequence of items traveling on the conveyor line."""
    if detector is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Detector unavailable")

    rng = np.random.default_rng(123)
    defect_options = ["puncture", "crushed_corner", "label_misalignment", "crack"]
    latencies = []
    passed = 0
    rejected = 0
    rejected_summary = []

    for i in range(req.batch_size):
        has_defect = rng.uniform(0.0, 1.0) < req.defect_probability
        dt = rng.choice(defect_options) if has_defect else "normal"

        base = generate_conveyor_frame(
            width=DEFAULT_CONVEYOR_CONFIG.image_width,
            height=DEFAULT_CONVEYOR_CONFIG.image_height,
            seed=100 + i,
        )
        frame, _ = inject_product_defect(base, defect_type=dt, seed=100 + i)

        t0 = time.perf_counter()
        res = detector.inspect_frame(frame, unit_id=f"ITEM_{i + 1:04d}")
        latencies.append((time.perf_counter() - t0) * 1000.0)

        if res.pass_status == "PASS":
            passed += 1
        else:
            rejected += 1
            rejected_summary.append({
                "unit_id": res.unit_id,
                "defect_type": res.primary_defect_type,
                "severity": res.defect_severity,
                "anomaly_score": res.anomaly_score,
            })

    mean_lat = float(np.mean(latencies))
    sim_ppm = (60.0 * 1000.0) / max(1e-3, mean_lat)

    return BatchSimulationResponse(
        total_units_inspected=req.batch_size,
        units_passed=passed,
        units_rejected=rejected,
        defect_rate_pct=round((rejected / req.batch_size) * 100.0, 2),
        simulated_ppm=round(sim_ppm, 1),
        mean_latency_ms=round(mean_lat, 2),
        rejected_units_summary=rejected_summary,
    )
