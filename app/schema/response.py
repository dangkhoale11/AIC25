from pydantic import BaseModel, Field


class KeyframeServiceReponse(BaseModel):
    key: int = Field(..., description="Keyframe key")
    video_num: int = Field(..., description="Video ID")
    group_num: int = Field(..., description="Group ID")
    keyframe_num: int = Field(..., description="Keyframe number")
    confidence_score: float = Field(..., description="Keyframe number")



class SingleKeyframeDisplay(BaseModel):
    path: str
    score: float
    key: int

from typing import List, Optional


class KeyframeDisplay(BaseModel):
    results: list[SingleKeyframeDisplay]
    raw_results: Optional[List[KeyframeServiceReponse]] = None


class TemporalEvent(BaseModel):
    """Represents a single temporal event with a start and end frame."""
    start_frame: Optional[SingleKeyframeDisplay] = None
    end_frame: Optional[SingleKeyframeDisplay] = None

class TemporalSearchResponse(BaseModel):
    """Response for temporal search, containing a list of events."""
    events: List[TemporalEvent]


class RerankSearchResponse(BaseModel):
    """Response for a reranked search, containing the reranked results."""
    results: list[SingleKeyframeDisplay]
    raw_results: Optional[List[KeyframeServiceReponse]] = None
    rerank_type: str