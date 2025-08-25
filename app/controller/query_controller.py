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


class QueryController:
    
    def __init__(
        self,
        data_folder: Path,
        id2index_path: Path,
        model_service: ModelService,
        keyframe_service: KeyframeQueryService,
        temporal_search_service: TemporalSearchService
    ):
        self.data_folder = data_folder
        self.id2index = json.load(open(id2index_path, 'r'))
        self.model_service = model_service
        self.keyframe_service = keyframe_service
        self.temporal_search_service = temporal_search_service
        self.translator = TextTranslator()

    
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
        translated_query = self.translator.translate(query)
        embedding = self.model_service.embedding(translated_query).tolist()[0]

        result = await self.keyframe_service.search_by_text(embedding, top_k, score_threshold)
        return result


    async def search_with_rerank(
        self,
        query: str,
        top_k: int,
        score_threshold: float,
        rerank_type: str,
        ocr_query: str | None = None,
    ):
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
        )
        return result


    async def search_text_with_exlude_group(
        self,
        query: str,
        top_k: int,
        score_threshold: float,
        list_group_exlude: list[int]
    ):
        exclude_ids = [
            int(k) for k, v in self.id2index.items()
            if int(v.split('/')[0]) in list_group_exlude
        ]

        
        translated_query = self.translator.translate(query)
        embedding = self.model_service.embedding(translated_query).tolist()[0]

        result = await self.keyframe_service.search_by_text_exclude_ids(embedding, top_k, score_threshold, exclude_ids)
        return result


    async def search_with_selected_video_group(
        self,
        query: str,
        top_k: int,
        score_threshold: float,
        list_of_include_groups: list[int]  ,
        list_of_include_videos: list[int]  
    ):     
        

        exclude_ids = None
        if len(list_of_include_groups) > 0   and len(list_of_include_videos) == 0:
            print("hi")
            exclude_ids = [
                int(k) for k, v in self.id2index.items()
                if int(v.split('/')[0]) not in list_of_include_groups
            ]
        
        elif len(list_of_include_groups) == 0   and len(list_of_include_videos) >0 :
            exclude_ids = [
                int(k) for k, v in self.id2index.items()
                if int(v.split('/')[1]) not in list_of_include_videos
            ]

        elif len(list_of_include_groups) == 0  and len(list_of_include_videos) == 0 :
            exclude_ids = []
        else:
            exclude_ids = [
                int(k) for k, v in self.id2index.items()
                if (
                    int(v.split('/')[0]) not in list_of_include_groups or
                    int(v.split('/')[1]) not in list_of_include_videos
                )
            ]

            print(len(exclude_ids))

        translated_query = self.translator.translate(query)
        embedding = self.model_service.embedding(translated_query).tolist()[0]
        # print(exclude_ids)
        result = await self.keyframe_service.search_by_text_exclude_ids(embedding, top_k, score_threshold, exclude_ids)
        return result
    

    

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
        pivot_frame: KeyframeServiceReponse,
    ):
        translated_start_query = self.translator.translate(start_query)
        translated_end_query = self.translator.translate(end_query)
        start_embedding = self.model_service.embedding(translated_start_query).tolist()[0]
        end_embedding = self.model_service.embedding(translated_end_query).tolist()[0]

        start_frame, end_frame = await self.temporal_search_service.search_temporal_event(
            start_query_embedding=start_embedding,
            end_query_embedding=end_embedding,
            pivot_frame=pivot_frame,
        )

        return start_frame, end_frame
