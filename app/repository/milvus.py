"""
The implementation of Vector Repository. The following class is responsible for getting the vector by many ways
Including Faiss and Usearch
"""


import os
import sys
from typing import List
ROOT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), '../'
    )
)
sys.path.insert(0, ROOT_DIR)


from typing import cast
from common.repository import MilvusBaseRepository
from pymilvus import Collection as MilvusCollection
from pymilvus.client.search_result import SearchResult
from schema.interface import  MilvusSearchRequest, MilvusSearchResult, MilvusSearchResponse

import asyncio





class KeyframeVectorRepository(MilvusBaseRepository):
    def __init__(
        self, 
        collection: MilvusCollection,
        search_params: dict
    ):
        
        super().__init__(collection)
        self.search_params = search_params
    
    async def search_by_embedding(
        self,
        request: MilvusSearchRequest
    ):
        expr = None
        if request.exclude_ids:
            expr = f"id not in {request.exclude_ids}"
        
        search_results= cast(SearchResult, self.collection.search(
            data=[request.embedding],
            anns_field="embedding",
            param=self.search_params,
            limit=request.top_k,
            expr=expr ,
            output_fields=["id", "embedding"],
            _async=False
        ))


        results = []
        for hits in search_results:
            for hit in hits:
                result = MilvusSearchResult(
                    id_=hit.id,
                    distance=hit.distance,
                    embedding=hit.entity.get("embedding") if hasattr(hit, 'entity') else None
                )
                results.append(result)
        
        return MilvusSearchResponse(
            results=results,
            total_found=len(results),
        )
    async def get_embeddings_by_ids(
        self,
        ids: List[int]
    ) -> List[List[float]]:
        """
        Get embeddings from Milvus by a list of IDs.
        """
        if not ids:
            return []

        # query để lấy field 'embedding'
        results = self.collection.query(
            expr=f"id in {ids}",
            output_fields=["embedding"],
        )

        # Milvus trả về list[dict], cần map ra đúng thứ tự id
        id_to_emb = {res["id"]: res["embedding"] for res in results}

        # Trả về theo đúng thứ tự input ids
        embeddings = [id_to_emb.get(i) for i in ids if i in id_to_emb]
        return embeddings


    def get_all_id(self) -> list[int]:
        return list(range(self.collection.num_entities))






class OcrVectorRepository(MilvusBaseRepository):
    def __init__(
        self,
        collection: MilvusCollection,
        search_params: dict
    ):

        super().__init__(collection)
        self.search_params = search_params
    
    async def search_by_embedding_and_ids(
        self,
        request: MilvusSearchRequest,
        ids: list[int]
    ):
        expr = f"id in {ids}"
        if request.exclude_ids:
            expr = f"id in {ids} and id not in {request.exclude_ids}"

        search_results= cast(SearchResult, self.collection.search(
            data=[request.embedding],
            anns_field="embedding",
            param=self.search_params,
            limit=request.top_k,
            expr=expr ,
            output_fields=["id", "embedding"],
            _async=False
        ))

        results = []
        for hits in search_results:
            for hit in hits:
                result = MilvusSearchResult(
                    id_=hit.id,
                    distance=hit.distance,
                    embedding=hit.entity.get("embedding") if hasattr(hit, 'entity') else None
                )
                results.append(result)

        return MilvusSearchResponse(
            results=results,
            total_found=len(results),
        )

