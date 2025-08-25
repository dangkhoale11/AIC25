import os
import sys
import numpy as np
ROOT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), '../'
    )
)
sys.path.insert(0, ROOT_DIR)


from repository.milvus import KeyframeVectorRepository, OcrVectorRepository
from repository.milvus import MilvusSearchRequest
from repository.mongo import KeyframeRepository

from schema.response import KeyframeServiceReponse

class KeyframeQueryService:
    def __init__(
            self, 
            keyframe_vector_repo: KeyframeVectorRepository,
            keyframe_mongo_repo: KeyframeRepository,
            ocr_vector_repo: OcrVectorRepository,
        ):
        self.keyframe_vector_repo = keyframe_vector_repo
        self.keyframe_mongo_repo= keyframe_mongo_repo
        self.ocr_vector_repo = ocr_vector_repo

    # ------------------------
    # Helper
    # ------------------------
    def _cos_sim(self, a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    async def _retrieve_keyframes(self, ids: list[int]):
        keyframes = await self.keyframe_mongo_repo.get_keyframe_by_list_of_keys(ids)
        keyframe_map = {k.key: k for k in keyframes}
        return [keyframe_map[k] for k in ids if k in keyframe_map]

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

    # ------------------------
    # Temporal Search (Adaptive stopping)
    # ------------------------
    async def temporal_search(
        self,
        query_embedding: list[float],
        pivot_idx: int,
        frame_embeddings: list[np.ndarray],
        direction: str = "forward",
        threshold: int = 3
    ):
        """
        Adaptive temporal search starting from pivot_idx.
        direction: "forward" or "backward"
        """
        q_emb = np.array(query_embedding, dtype=np.float32)
        best_idx = pivot_idx
        best_score = -1.0
        tolerance = 0

        if direction == "forward":
            idx_range = range(pivot_idx + 1, len(frame_embeddings))
        else:
            idx_range = range(pivot_idx - 1, -1, -1)

        for idx in idx_range:
            sim = self._cos_sim(q_emb, frame_embeddings[idx])
            if sim > best_score:
                best_score = sim
                best_idx = idx
                tolerance = 0
            else:
                tolerance += 1
                if tolerance >= threshold:
                    break

        return best_idx, best_score

    # ------------------------
    # OCR Rerank
    # ------------------------
    async def rerank_by_ocr(
        self,
        initial_results: list[KeyframeServiceReponse],
        ocr_embedding: list[float],
        top_k: int,
        ocr_weight: float = 0.5,
    ):
        initial_ids = [result.key for result in initial_results]
        search_request = MilvusSearchRequest(
            embedding=ocr_embedding,
            top_k=top_k
        )
        ocr_search_response = await self.ocr_vector_repo.search_by_embedding_and_ids(
            search_request, initial_ids
        )

        ocr_scores = {result.id_: result.distance for result in ocr_search_response.results}

        combined_results = []
        for result in initial_results:
            ocr_score = ocr_scores.get(result.key, 0.0)
            combined_score = (1 - ocr_weight) * result.confidence_score + ocr_weight * ocr_score
            result.confidence_score = combined_score
            combined_results.append(result)

        sorted_results = sorted(
            combined_results, key=lambda r: r.confidence_score, reverse=True
        )
        return sorted_results

    # ------------------------
    # GEM Rerank
    # ------------------------
    async def rerank_by_gem(
        self,
        initial_results: list[KeyframeServiceReponse],
        query_embedding: list[float],
        top_k: int,
        alpha: float = 0.7,
    ):
        if not initial_results:
            return []

        frame_ids = [res.key for res in initial_results]
        frame_embeddings = await self.keyframe_vector_repo.get_embeddings_by_ids(frame_ids)
        frame_embs = [np.array(fe, dtype=np.float32) for fe in frame_embeddings]

        q = np.array(query_embedding, dtype=np.float32)

        n = len(frame_ids)
        frame_sim_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                sim = self._cos_sim(frame_embs[i], frame_embs[j])
                frame_sim_matrix[i, j] = sim
                frame_sim_matrix[j, i] = sim

        combined_results = []
        for idx, res in enumerate(initial_results):
            f_emb = frame_embs[idx]
            sim_qf = self._cos_sim(q, f_emb)
            sim_global = float(np.mean(frame_sim_matrix[idx]))
            gem_score = alpha * sim_qf + (1 - alpha) * sim_global
            res.confidence_score = gem_score
            combined_results.append(res)

        sorted_results = sorted(combined_results, key=lambda r: r.confidence_score, reverse=True)
        return sorted_results[:top_k]

    # ------------------------
    # Entry point
    # ------------------------
    async def search_with_rerank(
        self,
        text_embedding: list[float],
        top_k: int,
        method: str = "ocr",   # "ocr" | "gem" | "temporal"
        ocr_embedding: list[float] = None,
        score_threshold: float = 0.5,
    ):
        initial_results = await self._search_keyframes(text_embedding, top_k, score_threshold)

        if method == "ocr" and ocr_embedding is not None:
            return await self.rerank_by_ocr(initial_results, ocr_embedding, top_k)
        elif method == "gem":
            return await self.rerank_by_gem(initial_results, text_embedding, top_k)
        elif method == "temporal":
            # Lấy top1 làm pivot, rồi expand forward + backward
            if not initial_results:
                return []
            frame_ids = [res.key for res in initial_results]
            frame_embeddings = await self.keyframe_vector_repo.get_embeddings_by_ids(frame_ids)
            frame_embs = [np.array(fe, dtype=np.float32) for fe in frame_embeddings]

            pivot_idx = 0
            forward_idx, forward_score = await self.temporal_search(
                text_embedding, pivot_idx, frame_embs, direction="forward"
            )
            backward_idx, backward_score = await self.temporal_search(
                text_embedding, pivot_idx, frame_embs, direction="backward"
            )

            chosen_idx = forward_idx if forward_score >= backward_score else backward_idx
            return [initial_results[chosen_idx]]
        else:
            return initial_results
