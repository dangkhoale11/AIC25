
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from typing import List, Optional, Union

from schema.request import (
    TextSearchRequest,
    TextSearchWithExcludeGroupsRequest,
    TextSearchWithSelectedGroupsAndVideosRequest,
    TextSearchWithOcrRequest,
    OcrRerankRequest,
    RerankSearchRequest,
    SearchStepRequest,
)
from schema.response import (
    KeyframeServiceReponse,
    SingleKeyframeDisplay,
    KeyframeDisplay,
    TemporalSearchResponse,
    TemporalEvent,
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


async def _handle_search_response(
    request: Union[TextSearchRequest, RerankSearchRequest],
    initial_results: list[KeyframeServiceReponse],
    controller: QueryController
) -> Union[KeyframeDisplay, TemporalSearchResponse]:
    """
    Handles the response logic for a search request, including optional temporal search.
    """
    if request.use_temporal and request.temporal_start_query and request.temporal_end_query:
        logger.info("Performing temporal search on initial results.")
        temporal_events_data = await controller.search_temporal(
            start_query=request.temporal_start_query,
            end_query=request.temporal_end_query,
            search_results=initial_results,
            search_range=request.temporal_search_range,
        )

        temporal_events = []
        for event_data in temporal_events_data:
            start_frame = event_data.get("start_frame")
            end_frame = event_data.get("end_frame")

            start_frame_display = None
            if start_frame:
                path, score = controller.convert_model_to_path(start_frame)
                start_frame_display = SingleKeyframeDisplay(path=path, score=score, key=start_frame.key)

            end_frame_display = None
            if end_frame:
                path, score = controller.convert_model_to_path(end_frame)
                end_frame_display = SingleKeyframeDisplay(path=path, score=score, key=end_frame.key)

            if start_frame_display and end_frame_display:
                temporal_events.append(
                    TemporalEvent(start_frame=start_frame_display, end_frame=end_frame_display)
                )

        return TemporalSearchResponse(events=temporal_events)

    else:
        logger.info(f"Found {len(initial_results)} results for query: '{request.query}'")
        display_results = []
        for r in initial_results:
            path, score = controller.convert_model_to_path(r)
            display_results.append(SingleKeyframeDisplay(path=path, score=score, key=r.key))
        return KeyframeDisplay(results=display_results, raw_results=initial_results)


@router.post(
    "/search/rerank",
    response_model=Union[KeyframeDisplay, TemporalSearchResponse],
    summary="Search with reranking",
    description="Perform a search and then rerank the results using a specified method. Can be combined with temporal search.",
    response_description="List of reranked keyframes or temporal search results."
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
        # ocr_query=request.ocr_query,
        p_qe=request.p_qe,
        p_dr=request.p_dr,
        m_neighbors=request.m_neighbors,
        sim_metric=request.sim_metric,
    )

    return await _handle_search_response(request, results, controller)


@router.post(
    "/search",
    response_model=Union[KeyframeDisplay, TemporalSearchResponse],
    summary="Simple text search for keyframes",
    description="Perform a simple text-based search for keyframes. Can be combined with temporal search.",
    response_description="List of matching keyframes or temporal search results."
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
    
    return await _handle_search_response(request, results, controller)


@router.post(
    "/search/step",
    response_model=KeyframeDisplay,
    summary="Perform one step in a multi-step search",
    description="Performs a search on a batch and combines it with previous results in the session based on the mode ('new', 'group', 'exclude').",
    response_description="The current combined list of keyframes for the session."
)
async def search_step(
    request: SearchStepRequest,
    controller: QueryController = Depends(get_query_controller)
):
    """
    Perform a single search step in a multi-step search session.
    """
    logger.info(f"Search step request for session '{request.session_id}': mode='{request.mode}', query='{request.query}'")

    if request.mode not in ["new", "group", "exclude"]:
        raise HTTPException(status_code=400, detail="Invalid mode. Must be 'new', 'group', or 'exclude'.")

    results = await controller.search_step(
        session_id=request.session_id,
        query=request.query,
        top_k=request.top_k,
        score_threshold=request.score_threshold,
        mode=request.mode
    )

    display_results = []
    for r in results:
        path, score = controller.convert_model_to_path(r)
        display_results.append(SingleKeyframeDisplay(path=path, score=score, key=r.key))

    return KeyframeDisplay(results=display_results, raw_results=results)



@router.post(
    "/search/exclude-groups",
    response_model=Union[KeyframeDisplay, TemporalSearchResponse],
    summary="Text search with group exclusion",
    description="Perform text-based search for keyframes while excluding specific groups. Can be combined with temporal search.",
    response_description="List of matching keyframes or temporal search results."
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
    
    return await _handle_search_response(request, results, controller)


@router.post(
    "/search/selected-groups-videos",
    response_model=Union[KeyframeDisplay, TemporalSearchResponse],
    summary="Text search within selected groups and videos",
    description="Perform text-based search for keyframes within specific groups and videos only. Can be combined with temporal search.",
    response_description="List of matching keyframes or temporal search results."
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
    
    return await _handle_search_response(request, results, controller)
