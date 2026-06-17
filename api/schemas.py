"""Pydantic v2 schemas for the Conveyor Product Defect Inspection API."""

from typing import List, Dict, Any
from pydantic import BaseModel, Field


class InspectionBoxSchema(BaseModel):
    defect_id: int
    y_min: int
    x_min: int
    y_max: int
    x_max: int
    area_px: int
    aspect_ratio: float
    defect_class: str
    confidence: float


class ConveyorInspectionResponse(BaseModel):
    unit_id: str
    pass_status: str = Field(..., description="PASS or REJECT")
    rejection_actuator_triggered: bool
    defect_severity: str = Field(..., description="NONE, MINOR, CRITICAL")
    primary_defect_type: str
    anomaly_score: float
    detected_defects: List[InspectionBoxSchema]
    latency_ms: float


class SyntheticInspectionRequest(BaseModel):
    defect_type: str = Field(default="puncture", description="normal, puncture, crushed_corner, label_misalignment, crack")
    seed: int = Field(default=42, ge=0)
    unit_id: str = Field(default="PROD_UNIT_4401")


class ConveyorBatchSimulationRequest(BaseModel):
    batch_size: int = Field(default=25, ge=5, le=100)
    defect_probability: float = Field(default=0.20, ge=0.0, le=1.0)


class BatchSimulationResponse(BaseModel):
    total_units_inspected: int
    units_passed: int
    units_rejected: int
    defect_rate_pct: float
    simulated_ppm: float
    mean_latency_ms: float
    rejected_units_summary: List[Dict[str, Any]]


class HealthResponse(BaseModel):
    status: str
    camera_ready: bool
    rejection_actuator_ready: bool
    target_ppm: int
    frame_resolution: str
