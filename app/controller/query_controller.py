from pathlib import Path
import json

import os
import sys
ROOT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), '../'
    )
)

sys.path.insert(0, ROOT_DIR)

from service import ModelService, KeyframeQueryService
from service.temporal_search_service import TemporalSearchService
from schema.response import KeyframeServiceReponse
from core.translation import TextTranslator


SEARCH_CACHE = {}


class QueryController:
    
    def __init__(
        self,
        data_folder: Path,
        id2index_path: Path,
        model_service: ModelService,
        keyframe_service: KeyframeQueryService,
        temporal_search_service: TemporalSearchService,
        batch: int
    ):
        self.data_folder = data_folder
        self.id2index = json.load(open(id2index_path, 'r'))
        self.model_service = model_service
        self.keyframe_service = keyframe_service
        self.temporal_search_service = temporal_search_service
        self.translator = TextTranslator()
        self.batch = batch

    
    def convert_model_to_path(
        self,
        model: KeyframeServiceReponse
    ) -> tuple[str, float]:
        return os.path.join(
            self.data_folder,
            f"L{model.group_num:02d}",
            f"V{model.video_num:03d}",
            f"{model.keyframe_num:06d}.webp"
        ), model.confidence_score
    
        
    async def search_text(
        self, 
        query: str,
        top_k: int,
        score_threshold: float
    ):
        cache_key = f"search_text_{self.batch}_{query}_{top_k}_{score_threshold}"
        if cache_key in SEARCH_CACHE:
            return SEARCH_CACHE[cache_key]

        translated_query = self.translator.translate(query)
        embedding = self.model_service.embedding(translated_query).tolist()[0]

        result = await self.keyframe_service.search_by_text(embedding, top_k, score_threshold)
        SEARCH_CACHE[cache_key] = result
        return result


    async def search_with_rerank(
        self,
        query: str,
        top_k: int,
        score_threshold: float,
        rerank_type: str,
        ocr_query: str | None = None,
        p_qe: float = 3.0,
        p_dr: float = 3.0,
        m_neighbors: int = 5,
        sim_metric: str = "cosine",
    ):
        cache_key = f"search_with_rerank_{self.batch}_{query}_{top_k}_{score_threshold}_{rerank_type}_{ocr_query}_{p_qe}_{p_dr}_{m_neighbors}_{sim_metric}"
        if cache_key in SEARCH_CACHE:
            return SEARCH_CACHE[cache_key]

        translated_query = self.translator.translate(query)
        text_embedding = self.model_service.embedding(translated_query).tolist()[0]

        ocr_embedding = None
        if rerank_type == "ocr" and ocr_query:
            translated_ocr_query = self.translator.translate(ocr_query)
            ocr_embedding = self.model_service.embedding_ocr(translated_ocr_query).tolist()

        result = await self.keyframe_service.search_with_rerank(
            text_embedding=text_embedding,
            top_k=top_k,
            score_threshold=score_threshold,
            method=rerank_type,
            ocr_embedding=ocr_embedding,
            p_qe=p_qe,
            p_dr=p_dr,
            m_neighbors=m_neighbors,
            sim_metric=sim_metric,
        )
        SEARCH_CACHE[cache_key] = result
        return result


    async def search_text_with_exlude_group(
        self,
        query: str,
        top_k: int,
        score_threshold: float,
        list_group_exlude: list[int]
    ):
        cache_key = f"search_text_with_exlude_group_{self.batch}_{query}_{top_k}_{score_threshold}_{list_group_exlude}"
        if cache_key in SEARCH_CACHE:
            return SEARCH_CACHE[cache_key]

        exclude_ids = [
            int(k) for k, v in self.id2index.items()
            if int(v.split('/')[0]) in list_group_exlude
        ]

        
        translated_query = self.translator.translate(query)
        embedding = self.model_service.embedding(translated_query).tolist()[0]

        result = await self.keyframe_service.search_by_text_exclude_ids(embedding, top_k, score_threshold, exclude_ids)

        # Safety filter
        result = [
            r for r in result
            if r.group_num not in list_group_exlude
        ]

        SEARCH_CACHE[cache_key] = result
        return result


    async def search_with_selected_video_group(
        self,
        query: str,
        top_k: int,
        score_threshold: float,
        list_of_include_groups: list[int],
        list_of_include_videos: list[int]
    ):
        """
        Search keyframes with optional filtering by groups and/or videos.
        """
        cache_key = f"search_with_selected_video_group_{self.batch}_{query}_{top_k}_{score_threshold}_{list_of_include_groups}_{list_of_include_videos}"
        if cache_key in SEARCH_CACHE:
            return SEARCH_CACHE[cache_key]

        # --- Bước 1: Chuẩn bị exclude_ids (giữ nguyên string key) ---
        exclude_ids: list[str] = []

        if list_of_include_groups and not list_of_include_videos:
            # Lọc theo group
            exclude_ids = [
                k for k, v in self.id2index.items()
                if int(v.split('/')[0]) not in list_of_include_groups
            ]

        elif not list_of_include_groups and list_of_include_videos:
            # Lọc theo video
            exclude_ids = [
                k for k, v in self.id2index.items()
                if int(v.split('/')[1]) not in list_of_include_videos
            ]

        elif not list_of_include_groups and not list_of_include_videos:
            # Không exclude gì cả
            exclude_ids = []

        else:
            # Có cả group lẫn video → loại bỏ những cái không nằm trong cả 2
            exclude_ids = [
                k for k, v in self.id2index.items()
                if (
                    int(v.split('/')[0]) not in list_of_include_groups or
                    int(v.split('/')[1]) not in list_of_include_videos
                )
            ]

        # logger.info(f"Exclude {len(exclude_ids)} ids out of {len(self.id2index)} total")

        # --- Bước 2: Embed query ---
        translated_query = self.translator.translate(query)
        embedding = self.model_service.embedding(translated_query).tolist()[0]

        # --- Bước 3: Search vector DB ---
        results = await self.keyframe_service.search_by_text_exclude_ids(
            embedding,
            top_k,
            score_threshold,
            exclude_ids
        )

        # --- Bước 4: Safety filter hậu kiểm ---
        results = [
            r for r in results
            if (not list_of_include_groups or r.group_num in list_of_include_groups)
            and (not list_of_include_videos or r.video_num in list_of_include_videos)
        ]

        SEARCH_CACHE[cache_key] = results
        return results
        

    

    async def search_text_with_ocr_filter(
        self,
        query: str,
        ocr_query: str,
        top_k: int,
        score_threshold: float,
        ocr_weight: float,
    ):
        translated_query = self.translator.translate(query)
        translated_ocr_query = self.translator.translate(ocr_query)
        text_embedding = self.model_service.embedding(translated_query).tolist()[0]
        ocr_embedding = self.model_service.embedding_ocr(translated_ocr_query).tolist()

        result = await self.keyframe_service.search_by_text_and_filter_with_ocr(
            text_embedding, ocr_embedding, top_k, score_threshold, ocr_weight
        )
        return result


    async def rerank_with_ocr(
        self,
        results: list[KeyframeServiceReponse],
        ocr_query: str,
        top_k: int,
        ocr_weight: float,
    ):
        translated_ocr_query = self.translator.translate(ocr_query)
        ocr_embedding = self.model_service.embedding_ocr(translated_ocr_query).tolist()

        result = await self.keyframe_service.rerank_by_ocr(
            results, ocr_embedding, top_k, ocr_weight
        )
        return result


    async def search_temporal(
        self,
        start_query: str,
        end_query: str,
        search_results: list[KeyframeServiceReponse],
        search_range: tuple[int, int],
    ):
        """
        Orchestrates a temporal search for multiple pivot frames.
        """
        translated_start_query = self.translator.translate(start_query)
        translated_end_query = self.translator.translate(end_query)
        start_embedding = self.model_service.embedding(translated_start_query).tolist()[0]
        end_embedding = self.model_service.embedding(translated_end_query).tolist()[0]

        start_idx, end_idx = search_range
        pivots = search_results[start_idx:end_idx]

        temporal_events = []
        for pivot_frame in pivots:
            start_frame, end_frame = await self.temporal_search_service.search_temporal_event(
                start_query_embedding=start_embedding,
                end_query_embedding=end_embedding,
                pivot_frame=pivot_frame,
            )
            if start_frame and end_frame:
                temporal_events.append({"start_frame": start_frame, "end_frame": end_frame})

        return temporal_events


    async def search_step(
        self,
        session_id: str,
        query: str,
        top_k: int,
        score_threshold: float,
        mode: str, # "new", "group", "exclude"
    ):
        # Perform search to get a list of KeyframeServiceResponse objects
        search_results = await self.search_text(query, top_k, score_threshold)

        # Extract just the IDs for caching and manipulation
        result_ids = {result.key for result in search_results}

        if mode == "new":
            SEARCH_CACHE[session_id] = result_ids
        elif mode == "group":
            if session_id in SEARCH_CACHE:
                SEARCH_CACHE[session_id].update(result_ids)
            else:
                SEARCH_CACHE[session_id] = result_ids
        elif mode == "exclude":
            if session_id in SEARCH_CACHE:
                SEARCH_CACHE[session_id].difference_update(result_ids)
        else: # Should not happen with proper validation in the router
            return {"error": "Invalid search mode"}

        # After updating the cache, retrieve the full objects for the current IDs
        final_ids = list(SEARCH_CACHE.get(session_id, []))
        final_results = await self.keyframe_service._retrieve_keyframes(final_ids)

        return final_results
