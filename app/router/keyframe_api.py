
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from typing import List, Optional

from schema.request import (
    TextSearchRequest,
    TextSearchWithExcludeGroupsRequest,
    TextSearchWithSelectedGroupsAndVideosRequest,
    TextSearchWithOcrRequest,
    OcrRerankRequest,
    TemporalSearchRequest,
    RerankSearchRequest,
)
from schema.response import (
    KeyframeServiceReponse,
    SingleKeyframeDisplay,
    KeyframeDisplay,
    TemporalSearchResponse,
)
from controller.query_controller import QueryController
from core.dependencies import get_query_controller
from core.logger import SimpleLogger


logger = SimpleLogger(__name__)


router = APIRouter(
    prefix="/keyframe",
    tags=["keyframe"],
    responses={404: {"description": "Not found"}},
)


@router.post(
    "/search/rerank",
    response_model=KeyframeDisplay,
    summary="Search with reranking",
    description="Perform a search and then rerank the results using a specified method.",
    response_description="List of reranked keyframes with confidence scores"
)
async def search_with_rerank(
    request: RerankSearchRequest,
    controller: QueryController = Depends(get_query_controller)
):
    """
    Search for keyframes with reranking.
    """

    logger.info(f"Rerank search request: query='{request.query}', rerank_type='{request.rerank_type}'")

    results = await controller.search_with_rerank(
        query=request.query,
        top_k=request.top_k,
        score_threshold=request.score_threshold,
        rerank_type=request.rerank_type,
        ocr_query=request.ocr_query,
    )

    logger.info(f"Found {len(results)} results with {request.rerank_type} reranking")

    display_results = []
    for r in results:
        path, score = controller.convert_model_to_path(r)
        display_results.append(SingleKeyframeDisplay(path=path, score=score, key=r.key))
    return KeyframeDisplay(results=display_results, raw_results=results)


@router.post(
    "/search",
    response_model=KeyframeDisplay,
    summary="Simple text search for keyframes",
    description="""
    Perform a simple text-based search for keyframes using semantic similarity.
    
    This endpoint converts the input text query to an embedding and searches for 
    the most similar keyframes in the database.
    
    **Parameters:**
    - **query**: The search text (1-1000 characters)
    - **top_k**: Maximum number of results to return (1-100, default: 10)
    - **score_threshold**: Minimum confidence score (0.0-1.0, default: 0.0)
    
    **Returns:**
    List of keyframes with their metadata and confidence scores, ordered by similarity.
    
    **Example:**
    ```json
    {
        "query": "person walking in the park",
        "top_k": 5,
        "score_threshold": 0.7
    }
    ```
    """,
    response_description="List of matching keyframes with confidence scores"
)
async def search_keyframes(
    request: TextSearchRequest,
    controller: QueryController = Depends(get_query_controller)
):
    """
    Search for keyframes using text query with semantic similarity.
    """
    
    logger.info(f"Text search request: query='{request.query}', top_k={request.top_k}, threshold={request.score_threshold}")
    
    results = await controller.search_text(
        query=request.query,
        top_k=request.top_k,
        score_threshold=request.score_threshold
    )
    
    logger.info(f"Found {len(results)} results for query: '{request.query}'")
    display_results = []
    for r in results:
        path, score = controller.convert_model_to_path(r)
        display_results.append(SingleKeyframeDisplay(path=path, score=score, key=r.key))
    return KeyframeDisplay(results=display_results, raw_results=results)


@router.post(
    "/rerank/ocr",
    response_model=KeyframeDisplay,
    summary="Re-rank keyframes with OCR",
    description="Re-rank a list of keyframes based on an OCR query.",
    response_description="List of re-ranked keyframes with confidence scores"
)
async def rerank_keyframes_with_ocr(
    request: OcrRerankRequest,
    controller: QueryController = Depends(get_query_controller)
):
    """
    Re-rank keyframes with OCR filtering.
    """

    logger.info(f"OCR rerank request: ocr_query='{request.ocr_query}'")

    results = await controller.rerank_with_ocr(
        results=request.results,
        ocr_query=request.ocr_query,
        top_k=request.top_k,
        ocr_weight=request.ocr_weight,
    )

    logger.info(f"Found {len(results)} results with OCR reranking")

    display_results = []
    for r in results:
        path, score = controller.convert_model_to_path(r)
        display_results.append(SingleKeyframeDisplay(path=path, score=score, key=r.key))
    return KeyframeDisplay(results=display_results, raw_results=results)


@router.post(
    "/search/ocr-filter",
    response_model=KeyframeDisplay,
    summary="Text search with OCR filtering",
    description="""
    Perform a text-based search for keyframes and then re-rank the results based on an OCR query.

    This endpoint first performs a standard text search and then uses a second query
    to search the OCR content of the initial results, combining the scores for a final ranking.

    **Parameters:**
    - **query**: The primary search text
    - **ocr_query**: The search text for OCR content
    - **top_k**: Maximum number of results to return
    - **score_threshold**: Minimum confidence score
    - **ocr_weight**: Weight for OCR score in re-ranking (0.0-1.0, default: 0.5)
    
    **Example:**
    ```json
    {
        "query": "a person at a table",
        "ocr_query": "menu",
        "top_k": 10,
        "score_threshold": 0.5,
        "ocr_weight": 0.7
    }
    ```
    """,
    response_description="List of matching keyframes, re-ranked with OCR scores"
)
async def search_keyframes_with_ocr_filter(
    request: TextSearchWithOcrRequest,
    controller: QueryController = Depends(get_query_controller)
):
    """
    Search for keyframes with OCR filtering.
    """

    logger.info(f"Text search with OCR filter: query='{request.query}', ocr_query='{request.ocr_query}'")

    results = await controller.search_text_with_ocr_filter(
        query=request.query,
        ocr_query=request.ocr_query,
        top_k=request.top_k,
        score_threshold=request.score_threshold,
        ocr_weight=request.ocr_weight,
    )

    logger.info(f"Found {len(results)} results with OCR filtering")

    display_results = []
    for r in results:
        path, score = controller.convert_model_to_path(r)
        display_results.append(SingleKeyframeDisplay(path=path, score=score, key=r.key))
    return KeyframeDisplay(results=display_results, raw_results=results)

@router.post(
    "/search/exclude-groups",
    response_model=KeyframeDisplay,
    summary="Text search with group exclusion",
    description="""
    Perform text-based search for keyframes while excluding specific groups.
    
    This endpoint allows you to search for keyframes while filtering out 
    results from specified groups (e.g., to avoid certain video categories).
    
    **Parameters:**
    - **query**: The search text
    - **top_k**: Maximum number of results to return
    - **score_threshold**: Minimum confidence score
    - **exclude_groups**: List of group IDs to exclude from results
    
    **Use Cases:**
    - Exclude specific video categories or datasets
    - Filter out content from certain time periods
    - Remove specific collections from search results
    
    **Example:**
    ```json
    {
        "query": "sunset landscape",
        "top_k": 15,
        "score_threshold": 0.6,
        "exclude_groups": [1, 3, 7]
    }
    ```
    """,
    response_description="List of matching keyframes excluding specified groups"
)
async def search_keyframes_exclude_groups(
    request: TextSearchWithExcludeGroupsRequest,
    controller: QueryController = Depends(get_query_controller)
):
    """
    Search for keyframes with group exclusion filtering.
    """

    logger.info(f"Text search with group exclusion: query='{request.query}', exclude_groups={request.exclude_groups}")
    
    results: list[KeyframeServiceReponse] = await controller.search_text_with_exlude_group(
        query=request.query,
        top_k=request.top_k,
        score_threshold=request.score_threshold,
        list_group_exlude=request.exclude_groups
    )
    
    logger.info(f"Found {len(results)} results excluding groups {request.exclude_groups}")\
    
    

    display_results = []
    for r in results:
        path, score = controller.convert_model_to_path(r)
        display_results.append(SingleKeyframeDisplay(path=path, score=score, key=r.key))
    return KeyframeDisplay(results=display_results, raw_results=results)






@router.post(
    "/search/selected-groups-videos",
    response_model=KeyframeDisplay,
    summary="Text search within selected groups and videos",
    description="""
    Perform text-based search for keyframes within specific groups and videos only.
    
    This endpoint allows you to limit your search to specific groups and videos,
    effectively creating a filtered search scope.
    
    **Parameters:**
    - **query**: The search text
    - **top_k**: Maximum number of results to return
    - **score_threshold**: Minimum confidence score
    - **include_groups**: List of group IDs to search within
    - **include_videos**: List of video IDs to search within
    
    **Behavior:**
    - Only keyframes from the specified groups AND videos will be searched
    - If a keyframe belongs to an included group OR an included video, it will be considered
    - Empty lists mean no filtering for that category
    
    **Use Cases:**
    - Search within specific video collections
    - Focus on particular time periods or datasets
    - Limit search to curated content sets
    
    **Example:**
    ```json
    {
        "query": "car driving on highway",
        "top_k": 20,
        "score_threshold": 0.5,
        "include_groups": [2, 4, 6],
        "include_videos": [101, 102, 203, 204]
    }
    ```
    """,
    response_description="List of matching keyframes from selected groups and videos"
)
async def search_keyframes_selected_groups_videos(
    request: TextSearchWithSelectedGroupsAndVideosRequest,
    controller: QueryController = Depends(get_query_controller)
):
    """
    Search for keyframes within selected groups and videos.
    """

    logger.info(f"Text search with selection: query='{request.query}', include_groups={request.include_groups}, include_videos={request.include_videos}")
    
    results = await controller.search_with_selected_video_group(
        query=request.query,
        top_k=request.top_k,
        score_threshold=request.score_threshold,
        list_of_include_groups=request.include_groups,
        list_of_include_videos=request.include_videos
    )
    
    logger.info(f"Found {len(results)} results within selected groups/videos")

    display_results = []
    for r in results:
        path, score = controller.convert_model_to_path(r)
        display_results.append(SingleKeyframeDisplay(path=path, score=score, key=r.key))
    return KeyframeDisplay(results=display_results, raw_results=results)


@router.post(
    "/search/temporal",
    response_model=TemporalSearchResponse,
    summary="Temporal search for an event",
    description="""
    Perform a temporal search for an event defined by a start and end query, starting from a pivot keyframe.

    **Parameters:**
    - **start_query**: The search text for the beginning of the event.
    - **end_query**: The search text for the end of the event.
    - **pivot_frame**: The keyframe to start the search from.

    **Returns:**
    The start and end keyframes of the event.
    """,
    response_description="The start and end keyframes of the event."
)
async def search_temporal_event(
    request: TemporalSearchRequest,
    controller: QueryController = Depends(get_query_controller)
):
    """
    Search for a temporal event.
    """

    logger.info(f"Temporal search request: start_query='{request.start_query}', end_query='{request.end_query}'")

    start_frame, end_frame = await controller.search_temporal(
        start_query=request.start_query,
        end_query=request.end_query,
        pivot_frame=request.pivot_frame,
    )
    
    start_frame_display = None
    if start_frame:
        path, score = controller.convert_model_to_path(start_frame)
        start_frame_display = SingleKeyframeDisplay(path=path, score=score)

    end_frame_display = None
    if end_frame:
        path, score = controller.convert_model_to_path(end_frame)
        end_frame_display = SingleKeyframeDisplay(path=path, score=score)

    return TemporalSearchResponse(start_frame=start_frame_display, end_frame=end_frame_display)
