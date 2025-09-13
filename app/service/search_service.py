import os
import sys
import numpy as np
from typing import List
ROOT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), '../'
    )
)
sys.path.insert(0, ROOT_DIR)


from repository.milvus import KeyframeVectorRepository
# , OcrVectorRepository
from repository.milvus import MilvusSearchRequest
from repository.mongo import KeyframeRepository

from schema.response import KeyframeServiceReponse


from sklearn.metrics.pairwise import cosine_similarity



class KeyframeQueryService:
    def __init__(
            self, 
            keyframe_vector_repo: KeyframeVectorRepository,
            keyframe_mongo_repo: KeyframeRepository,
            # ocr_vector_repo: OcrVectorRepository,
            
        ):

        self.keyframe_vector_repo = keyframe_vector_repo
        self.keyframe_mongo_repo= keyframe_mongo_repo
        # self.ocr_vector_repo = ocr_vector_repo

    
    
    async def _retrieve_keyframes(self, ids: list[int]):
        keyframes = await self.keyframe_mongo_repo.get_keyframe_by_list_of_keys(ids)
        print(keyframes[:5])
  
        keyframe_map = {k.key: k for k in keyframes}
        return_keyframe = [
            keyframe_map[k] for k in ids
        ]   
        return return_keyframe

    async def _search_keyframes(
        self,
        text_embedding: list[float],
        top_k: int,
        score_threshold: float | None = None,
        exclude_indices: list[int] | None = None
    ) -> list[KeyframeServiceReponse]:
        
        search_request = MilvusSearchRequest(
            embedding=text_embedding,
            top_k=top_k,
            exclude_ids=exclude_indices
        )

        search_response = await self.keyframe_vector_repo.search_by_embedding(search_request)

        
        filtered_results = [
            result for result in search_response.results
            if score_threshold is None or result.distance > score_threshold
        ]

        sorted_results = sorted(
            filtered_results, key=lambda r: r.distance, reverse=True
        )

        sorted_ids = [result.id_ for result in sorted_results]

        keyframes = await self._retrieve_keyframes(sorted_ids)



        keyframe_map = {k.key: k for k in keyframes}
        response = []

        for result in sorted_results:
            keyframe = keyframe_map.get(result.id_) 
            if keyframe is not None:
                response.append(
                    KeyframeServiceReponse(
                        key=keyframe.key,
                        video_num=keyframe.video_num,
                        group_num=keyframe.group_num,
                        keyframe_num=keyframe.keyframe_num,
                        confidence_score=result.distance
                    )
                )
        return response
    

    async def search_by_text(
        self,
        text_embedding: list[float],
        top_k: int,
        score_threshold: float | None = 0.5,
    ):
        return await self._search_keyframes(text_embedding, top_k, score_threshold, None)   
    

    async def search_by_text_range(
        self,
        text_embedding: list[float],
        top_k: int,
        score_threshold: float | None,
        range_queries: list[tuple[int,int]]
    ):
        """
        range_queries: a bunch of start end indices, and we just search inside these, ignore everything
        """

        all_ids = self.keyframe_vector_repo.get_all_id()
        allowed_ids = set()
        for start, end in range_queries:
            allowed_ids.update(range(start, end + 1))
        
        
        exclude_ids = [id_ for id_ in all_ids if id_ not in allowed_ids]

        return await self._search_keyframes(text_embedding, top_k, score_threshold, exclude_ids)   
    

    async def search_by_text_exclude_ids(
        self,
        text_embedding: list[float],
        top_k: int,
        score_threshold: float | None,
        exclude_ids: list[int] | None
    ):
        """
        range_queries: a bunch of start end indices, and we just search inside these, ignore everything
        """
        return await self._search_keyframes(text_embedding, top_k, score_threshold, exclude_ids)   
    
    

    # async def search_by_text_and_filter_with_ocr(
    #     self,
    #     text_embedding: list[float],
    #     ocr_embedding: list[float],
    #     top_k: int,
    #     score_threshold: float | None,
    #     ocr_weight: float = 0.5,
    # ):
    #     # 1. Initial search on keyframes
    #     initial_results = await self._search_keyframes(text_embedding, top_k, score_threshold, None)

    #     if not initial_results:
    #         return []

    #     return await self.rerank_by_ocr(initial_results, ocr_embedding, top_k, ocr_weight)


    # async def rerank_by_ocr(
    #     self,
    #     initial_results: list[KeyframeServiceReponse],
    #     ocr_embedding: list[float],
    #     top_k: int,
    #     ocr_weight: float,
    # ):
    #     initial_ids = [result.key for result in initial_results]

    #     # 2. Re-rank based on OCR search
    #     search_request = MilvusSearchRequest(
    #         embedding=ocr_embedding,
    #         top_k=top_k
    #     )
        
    #     ocr_search_response = await self.ocr_vector_repo.search_by_embedding_and_ids(search_request, initial_ids)

    #     # 3. Create a map of id -> ocr_score
    #     ocr_scores = {result.id_: result.distance for result in ocr_search_response.results}

    #     # 4. Combine and re-sort results
    #     combined_results = []
    #     for result in initial_results:
    #         ocr_score = ocr_scores.get(result.key, 0.0)
    #         # a simple average, can be replaced with more sophisticated weighting
    #         combined_score = (1 - ocr_weight) * result.confidence_score + ocr_weight * ocr_score
    #         result.confidence_score = combined_score
    #         combined_results.append(result)

    #     # 5. Sort by the new combined score
    #     sorted_results = sorted(
    #         combined_results, key=lambda r: r.confidence_score, reverse=True
    #     )

    #     return sorted_results


    def gem_pooling_batch(self, vectors: np.ndarray, p: float = 1.0) -> np.ndarray:
        """
        Generalized Mean (GeM) pooling trên batch vector.
        Args:
            vectors (np.ndarray): shape (N, D)
            p (float): tham số pooling
                    - p=1: mean pooling
                    - p lớn: gần max pooling
                    - p=np.inf: max pooling
        Returns:
            np.ndarray: vector sau pooling (chưa normalize)
        """
        if vectors is None or len(vectors) == 0:
            return np.array([], dtype=np.float32)

        epsilon = 1e-12
        vectors = np.abs(vectors) + epsilon

        if np.isinf(p):
            result = np.max(vectors, axis=0)
        else:
            powered = np.power(vectors, p)
            mean_powered = np.mean(powered, axis=0)
            result = np.power(mean_powered, 1.0 / p)

        return result.astype(np.float32)

    # ----- similarity metrics -----
    def _cos_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        if a is None or b is None:
            return 0.0
        try:
            a = a.reshape(1, -1)
            b = b.reshape(1, -1)
            return float(cosine_similarity(a, b)[0][0])
        except Exception:
            return 0.0

    def _dot_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        if a is None or b is None:
            return 0.0
        try:
            a = a.reshape(-1)
            b = b.reshape(-1)
            return float(np.dot(a, b))
        except Exception:
            return 0.0

    def _neg_euclid_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        if a is None or b is None:
            return 0.0
        try:
            a = a.reshape(-1)
            b = b.reshape(-1)
            return -float(np.linalg.norm(a - b))
        except Exception:
            return 0.0

    async def rerank_by_gem(
        self,
        initial_results: list,
        query_embedding: list[float],
        top_k: int,
        m_neighbors: int = 2,
        p_qe: float = 1.0,
        p_dr: float = 1.0,
        sim_metric: str = "cosine"
    ):
        if not initial_results:
            return []

        # ----- Query embedding g_q -----
        g_q = np.array(query_embedding, dtype=np.float32)

        # ----- Lấy embedding của các keyframe -----
        frame_ids = [res.key for res in initial_results]
        frame_embs = np.array(
            await self.keyframe_vector_repo.get_embeddings_by_ids(frame_ids),
            dtype=np.float32
        )
        n, d = frame_embs.shape

        # Nếu m_neighbors > số lượng frame thì giới hạn lại
        if m_neighbors is None or m_neighbors > n - 1:
            m_neighbors = max(0, n - 1)

        # ----- Query refine (g_qe) -----
        qe_vectors = np.vstack([g_q.reshape(1, -1), frame_embs])
        g_qe = self.gem_pooling_batch(qe_vectors, p=p_qe)

        # ----- similarity matrix (luôn dùng cosine để chọn neighbor) -----
        norms = np.linalg.norm(frame_embs, axis=1, keepdims=True) + 1e-12
        frame_embs_norm = frame_embs / norms
        sim_matrix = frame_embs_norm @ frame_embs_norm.T
        np.fill_diagonal(sim_matrix, -np.inf)

        # chọn metric
        if sim_metric == "cosine":
            sim_fn = self._cos_sim
        elif sim_metric == "euclid":
            sim_fn = self._neg_euclid_sim
        else:
            sim_fn = self._dot_sim

        combined_results = []
        for idx, res in enumerate(initial_results):
            g_d = frame_embs[idx]

            # ----- Lấy m_neighbors neighbors của frame idx -----
            sims = sim_matrix[idx]
            if m_neighbors == 0:
                neighbor_indices = []
            elif m_neighbors >= n - 1:
                neighbor_indices = [i for i in range(n) if i != idx]
            else:
                part = np.argpartition(-sims, m_neighbors)[:m_neighbors]
                neighbor_indices = part[np.argsort(-sims[part])]

            # ----- Tính g_dr -----
            if len(neighbor_indices) == 0:
                g_dr = g_d.copy()
            else:
                dr_vectors = np.vstack([g_d.reshape(1, -1), frame_embs[neighbor_indices]])
                g_dr = self.gem_pooling_batch(dr_vectors, p=p_dr)

            # ----- Score -----
            S1 = sim_fn(g_q, g_dr)   # query gốc vs frame refine
            S2 = sim_fn(g_qe, g_d)   # query refine vs frame gốc
            score = 0.5 * (S1 + S2)

            res.confidence_score = float(score)
            combined_results.append(res)

        # ----- Sort kết quả -----
        sorted_results = sorted(combined_results, key=lambda r: r.confidence_score, reverse=True)
        return sorted_results[:top_k]

    # ------------------------
    # Entry point
    # ------------------------
    async def rerank(
        self,
        initial_results: list,
        query_embedding: list[float],
        top_k: int,
        method: str = "GEM",   # "ocr" | "gem" | "temporal"
        # ocr_embedding: list[float] = None,
        p_qe: float = 3.0,
        p_dr: float = 3.0,
        m_neighbors: int = 5,
        sim_metric: str = "cosine",
    ):
        # if method == "OCR" and ocr_embedding is not None:
        #     return await self.rerank_by_ocr(initial_results, ocr_embedding, top_k)
        if method == "GEM":
            return await self.rerank_by_gem(
                initial_results, query_embedding, top_k,
                m_neighbors=m_neighbors, p_qe=p_qe, p_dr=p_dr, sim_metric=sim_metric
            )
