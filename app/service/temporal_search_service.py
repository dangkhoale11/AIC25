import os
import sys
import numpy as np
from typing import List, Tuple

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(0, ROOT_DIR)

from repository.milvus import KeyframeVectorRepository
from schema.response import KeyframeServiceReponse

class TemporalSearchService:
    def __init__(
        self,
        keyframe_vector_repo: KeyframeVectorRepository,
    ):
        self.keyframe_vector_repo = keyframe_vector_repo

    def _cos_sim(self, a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    async def _find_best_match_in_slice(
        self,
        query_embedding: np.ndarray,
        frames: List[KeyframeServiceReponse],
        frame_embeddings: List[np.ndarray],
    ) -> Tuple[KeyframeServiceReponse, float]:
        """
        Finds the frame in a list that best matches the query embedding.
        """
        best_score = -1.0
        best_frame = None

        for i, frame in enumerate(frames):
            sim = self._cos_sim(query_embedding, frame_embeddings[i])
            if sim > best_score:
                best_score = sim
                best_frame = frame

        if best_frame:
            best_frame.confidence_score = best_score

        return best_frame, best_score

    async def search_temporal_event(
        self,
        start_query_embedding: list[float],
        end_query_embedding: list[float],
        search_results: List[KeyframeServiceReponse],
        search_range: tuple[int, int],
    ):
        """
        Performs a temporal-style search on a slice of initial search results.
        """
        start_range, end_range = search_range

        # 1. Slice the initial search results based on the provided range
        # Ensure range is valid
        if not (0 <= start_range < end_range <= len(search_results)):
            return None, None

        results_slice = search_results[start_range:end_range]

        if not results_slice:
            return None, None

        # 2. Get embeddings for the frames in the slice
        frame_ids = [k.key for k in results_slice]
        frame_embeddings = await self.keyframe_vector_repo.get_embeddings_by_ids(frame_ids)

        if not frame_embeddings:
            return None, None

        # 3. Convert embeddings to numpy arrays for calculation
        frame_embeddings_np = [np.array(fe, dtype=np.float32) for fe in frame_embeddings]
        start_query_embedding_np = np.array(start_query_embedding, dtype=np.float32)
        end_query_embedding_np = np.array(end_query_embedding, dtype=np.float32)

        # 4. Find the best match for the start and end queries within the slice
        start_frame, _ = await self._find_best_match_in_slice(
            query_embedding=start_query_embedding_np,
            frames=results_slice,
            frame_embeddings=frame_embeddings_np,
        )

        end_frame, _ = await self._find_best_match_in_slice(
            query_embedding=end_query_embedding_np,
            frames=results_slice,
            frame_embeddings=frame_embeddings_np,
        )

        return start_frame, end_frame
