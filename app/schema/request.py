from pydantic import BaseModel, Field
from typing import List, Optional


class BaseSearchRequest(BaseModel):
    """Base search request with common parameters"""
    query: str = Field(..., description="Search query text", min_length=1, max_length=1000)
    top_k: int = Field(default=10, ge=1, le=500, description="Number of top results to return")
    score_threshold: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum confidence score threshold")


class TextSearchRequest(BaseSearchRequest):
    """Simple text search request"""
    pass


class TextSearchWithExcludeGroupsRequest(BaseSearchRequest):
    """Text search request with group exclusion"""
    exclude_groups: List[int] = Field(
        default_factory=list,
        description="List of group IDs to exclude from search results",
    )


from .response import KeyframeServiceReponse


class TextSearchWithOcrRequest(BaseSearchRequest):
    """Text search request with OCR filtering"""
    ocr_query: str = Field(..., description="OCR search query text", min_length=1, max_length=1000)
    ocr_weight: float = Field(default=0.5, ge=0.0, le=1.0, description="Weight for OCR score in re-ranking")


class OcrRerankRequest(BaseModel):
    """Request to re-rank a list of keyframes using an OCR query"""
    results: List[KeyframeServiceReponse] = Field(..., description="List of keyframe results to re-rank")
    ocr_query: str = Field(..., description="OCR search query text", min_length=1, max_length=1000)
    top_k: int = Field(default=10, ge=1, le=500, description="Number of top results to return")
    ocr_weight: float = Field(default=0.5, ge=0.0, le=1.0, description="Weight for OCR score in re-ranking")


class TextSearchWithSelectedGroupsAndVideosRequest(BaseSearchRequest):
    """Text search request with specific group and video selection"""
    include_groups: List[int] = Field(
        default_factory=list,
        description="List of group IDs to include in search results",
    )
    include_videos: List[int] = Field(
        default_factory=list,
        description="List of video IDs to include in search results",
    )


class TemporalSearchRequest(BaseModel):
    """Request for temporal search"""
    start_query: str = Field(..., description="Search query for the start of the event")
    end_query: str = Field(..., description="Search query for the end of the event")
    pivot_frame: KeyframeServiceReponse = Field(..., description="The pivot keyframe for the temporal search")


class RerankSearchRequest(BaseSearchRequest):
    """Request for search with reranking"""
    rerank_type: str = Field(..., description="The reranking method to use (e.g., 'ocr', 'gem')")
    ocr_query: Optional[str] = Field(None, description="OCR search query text, required for OCR rerank")
