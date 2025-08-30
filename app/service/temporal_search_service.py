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
        window_size: int = 10,
    ):
        """
        Performs a temporal search using a sliding window approach around pivot points.
        """
        start_idx, end_idx = search_range

        if not (0 <= start_idx < end_idx <= len(search_results)):
            return None, None

        candidate_start_frames = []
        candidate_end_frames = []

        start_query_embedding_np = np.array(start_query_embedding, dtype=np.float32)
        end_query_embedding_np = np.array(end_query_embedding, dtype=np.float32)

        for pivot_idx in range(start_idx, end_idx):
            window_start = max(0, pivot_idx - window_size)
            window_end = min(len(search_results), pivot_idx + window_size + 1)

            results_slice = search_results[window_start:window_end]
            if not results_slice:
                continue

            frame_ids = [k.key for k in results_slice]
            frame_embeddings = await self.keyframe_vector_repo.get_embeddings_by_ids(frame_ids)
            if not frame_embeddings:
                continue

            frame_embeddings_np = [np.array(fe, dtype=np.float32) for fe in frame_embeddings]

            start_frame, start_score = await self._find_best_match_in_slice(
                query_embedding=start_query_embedding_np,
                frames=results_slice,
                frame_embeddings=frame_embeddings_np,
            )
            if start_frame:
                candidate_start_frames.append(start_frame)

            end_frame, end_score = await self._find_best_match_in_slice(
                query_embedding=end_query_embedding_np,
                frames=results_slice,
                frame_embeddings=frame_embeddings_np,
            )
            if end_frame:
                candidate_end_frames.append(end_frame)

        best_start_frame = max(candidate_start_frames, key=lambda f: f.confidence_score, default=None)
        best_end_frame = max(candidate_end_frames, key=lambda f: f.confidence_score, default=None)

        return best_start_frame, best_end_frame
