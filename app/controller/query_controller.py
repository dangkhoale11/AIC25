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

from service import ModelService
from factory.factory import ServiceFactory
from schema.response import KeyframeServiceReponse
from core.translation import TextTranslator


class QueryController:
    
    def __init__(
        self,
        data_folder: Path,
        id2index_path: Path,
        model_service: ModelService,
        service_factory: ServiceFactory
    ):
        self.data_folder = data_folder
        self.id2index = json.load(open(id2index_path, 'r'))
        self.model_service = model_service
        self.service_factory = service_factory
        self.translator = TextTranslator()
        self.cache = {}

    
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
    
    def clear_cache(self):
        self.cache = {}
        
    async def search_text(
        self, 
        query: str,
        top_k: int,
        score_threshold: float,
        search_mode: str = "batch-1"
    ):
        translated_query = self.translator.translate(query)
        embedding = self.model_service.embedding(translated_query).tolist()[0]

        keyframe_service = self.service_factory.get_keyframe_query_service(search_mode)
        result = await keyframe_service.search_by_text(embedding, top_k, score_threshold)
        self.cache['last_search'] = result
        return result


    async def search_with_rerank(
        self,
        query: str,
        top_k: int,
        score_threshold: float,
        rerank_type: str,
        search_mode: str = "batch-1",
        ocr_query: str | None = None,
        p_qe: float = 3.0,
        p_dr: float = 3.0,
        m_neighbors: int = 5,
        sim_metric: str = "cosine",
    ):
        translated_query = self.translator.translate(query)
        text_embedding = self.model_service.embedding(translated_query).tolist()[0]

        ocr_embedding = None
        if rerank_type == "ocr" and ocr_query:
            translated_ocr_query = self.translator.translate(ocr_query)
            ocr_embedding = self.model_service.embedding_ocr(translated_ocr_query).tolist()

        keyframe_service = self.service_factory.get_keyframe_query_service(search_mode)

        # Use cached results if available
        initial_results = self.cache.get('last_search')
        if not initial_results:
            initial_results = await keyframe_service.search_by_text(text_embedding, top_k, score_threshold)
            self.cache['last_search'] = initial_results

        result = await keyframe_service.search_with_rerank(
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
        self.cache['last_search'] = result # Override cache with reranked results
        return result


    async def search_text_with_exlude_group(
        self,
        query: str,
        top_k: int,
        score_threshold: float,
        list_group_exlude: list[int],
        search_mode: str = "batch-1"
    ):
        exclude_ids = [
            int(k) for k, v in self.id2index.items()
            if int(v.split('/')[0]) in list_group_exlude
        ]

        
        translated_query = self.translator.translate(query)
        embedding = self.model_service.embedding(translated_query).tolist()[0]

        keyframe_service = self.service_factory.get_keyframe_query_service(search_mode)
        result = await keyframe_service.search_by_text_exclude_ids(embedding, top_k, score_threshold, exclude_ids)
        self.cache['last_search'] = result
        return result


    async def search_with_selected_video_group(
        self,
        query: str,
        top_k: int,
        score_threshold: float,
        list_of_include_groups: list[int],
        list_of_include_videos: list[int],
        search_mode: str = "batch-1"
    ):
        """
        Search keyframes with optional filtering by groups and/or videos.
        """

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
        keyframe_service = self.service_factory.get_keyframe_query_service(search_mode)
        results = await keyframe_service.search_by_text_exclude_ids(
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
        self.cache['last_search'] = results
        return results
        

    

    async def search_text_with_ocr_filter(
        self,
        query: str,
        ocr_query: str,
        top_k: int,
        score_threshold: float,
        ocr_weight: float,
        search_mode: str = "batch-1"
    ):
        translated_query = self.translator.translate(query)
        translated_ocr_query = self.translator.translate(ocr_query)
        text_embedding = self.model_service.embedding(translated_query).tolist()[0]
        ocr_embedding = self.model_service.embedding_ocr(translated_ocr_query).tolist()

        keyframe_service = self.service_factory.get_keyframe_query_service(search_mode)
        result = await keyframe_service.search_by_text_and_filter_with_ocr(
            text_embedding, ocr_embedding, top_k, score_threshold, ocr_weight
        )
        self.cache['last_search'] = result
        return result


    async def rerank_with_ocr(
        self,
        results: list[KeyframeServiceReponse],
        ocr_query: str,
        top_k: int,
        ocr_weight: float,
        search_mode: str = "batch-1"
    ):
        translated_ocr_query = self.translator.translate(ocr_query)
        ocr_embedding = self.model_service.embedding_ocr(translated_ocr_query).tolist()

        keyframe_service = self.service_factory.get_keyframe_query_service(search_mode)
        result = await keyframe_service.rerank_by_ocr(
            results, ocr_embedding, top_k, ocr_weight
        )
        self.cache['last_search'] = result
        return result


    async def search_temporal(
        self,
        start_query: str,
        end_query: str,
        search_results: list[KeyframeServiceReponse],
        search_range: tuple[int, int],
        search_mode: str = "batch-1"
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

        temporal_search_service = self.service_factory.get_temporal_search_service(search_mode)
        temporal_events = []
        for pivot_frame in pivots:
            start_frame, end_frame = await temporal_search_service.search_temporal_event(
                start_query_embedding=start_embedding,
                end_query_embedding=end_embedding,
                pivot_frame=pivot_frame,
            )
            if start_frame and end_frame:
                temporal_events.append({"start_frame": start_frame, "end_frame": end_frame})

        return temporal_events
