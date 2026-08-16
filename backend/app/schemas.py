from pydantic import BaseModel, Field
from enum import Enum
from typing import Dict, Any, Optional
from datetime import datetime

class ModalityEnum(str, Enum):
    IMAGE = "IMAGE"
    TEXT = "TEXT"

class TaskStatusEnum(str, Enum):
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class TaskAcceptedResponse(BaseModel):
    task_id: str
    status: TaskStatusEnum = TaskStatusEnum.QUEUED
    estimated_ms: int = 450
    created_at: datetime = Field(default_factory=datetime.utcnow)

class ForensicVerdict(BaseModel):
    is_ai_generated: bool
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    classification: str

class TaskResultResponse(BaseModel):
    task_id: str
    status: TaskStatusEnum
    verdict: Optional[ForensicVerdict] = None
    sub_metrics: Optional[Dict[str, float]] = None
    artifacts: Optional[Dict[str, Any]] = None
    execution_time_ms: Optional[int] = None
    error: Optional[str] = None