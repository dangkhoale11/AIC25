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


class TemporalSearchResponse(BaseModel):
    """Response for temporal search"""
    start_frame: Optional[SingleKeyframeDisplay] = None
    end_frame: Optional[SingleKeyframeDisplay] = None