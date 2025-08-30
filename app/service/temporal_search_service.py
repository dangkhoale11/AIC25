import os
import sys
import numpy as np
from typing import List

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(0, ROOT_DIR)

from repository.milvus import KeyframeVectorRepository
from repository.mongo import KeyframeRepository
from schema.response import KeyframeServiceReponse

class TemporalSearchService:
    def __init__(
        self,
        keyframe_vector_repo: KeyframeVectorRepository,
        keyframe_mongo_repo: KeyframeRepository,
    ):
        self.keyframe_vector_repo = keyframe_vector_repo
        self.keyframe_mongo_repo = keyframe_mongo_repo

    def _cos_sim(self, a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    async def _adaptive_temporal_search(
        self,
        query_embedding: np.ndarray,
        frames: List[KeyframeServiceReponse],
        frame_embeddings: List[np.ndarray],
        pivot_index: int,
        direction: str,
        threshold: int = 3,
    ):
        best_idx = pivot_index
        best_score = -1.0
        tolerance = 0

        if direction == "forward":
            idx_range = range(pivot_index + 1, len(frames))
        else:
            idx_range = range(pivot_index - 1, -1, -1)

        for idx in idx_range:
            sim = self._cos_sim(query_embedding, frame_embeddings[idx])
            if sim > best_score:
                best_score = sim
                best_idx = idx
                tolerance = 0
            else:
                tolerance += 1
                if tolerance >= threshold:
                    break

        if best_idx == pivot_index:
            best_score = self._cos_sim(query_embedding, frame_embeddings[pivot_index])

        return best_idx, best_score

    async def search_temporal_event(
        self,
        start_query_embedding: list[float],
        end_query_embedding: list[float],
        pivot_frame: KeyframeServiceReponse,
        temporal_tolerance: int = 3,
    ):
        # 1. Get all keyframes from the same video as the pivot frame
        video_num = pivot_frame.video_num
        keyframes_in_video = await self.keyframe_mongo_repo.get_keyframes_by_pivot(pivot_frame=pivot_frame)

        if not keyframes_in_video:
            return None, None

        # Sort frames by keyframe_num to ensure chronological order
        sorted_keyframes = sorted(keyframes_in_video, key=lambda k: k.keyframe_num)

        frame_ids = [k.key for k in sorted_keyframes]
        frame_embeddings = await self.keyframe_vector_repo.get_embeddings_by_ids(frame_ids)

        if not frame_embeddings:
            return None, None

        # Convert to numpy arrays
        frame_embeddings_np = [np.array(fe, dtype=np.float32) for fe in frame_embeddings]
        start_query_embedding_np = np.array(start_query_embedding, dtype=np.float32)
        end_query_embedding_np = np.array(end_query_embedding, dtype=np.float32)

        # 2. Find the pivot index in the sorted list
        try:
            pivot_index = frame_ids.index(pivot_frame.key)
        except ValueError:
            return None, None # Pivot not found in the video, should not happen

        # 3. Perform temporal search
        start_idx, start_score = await self._adaptive_temporal_search(
            query_embedding=start_query_embedding_np,
            frames=sorted_keyframes,
            frame_embeddings=frame_embeddings_np,
            pivot_index=pivot_index,
            direction="backward",
            threshold=temporal_tolerance,
        )

        end_idx, end_score = await self._adaptive_temporal_search(
            query_embedding=end_query_embedding_np,
            frames=sorted_keyframes,
            frame_embeddings=frame_embeddings_np,
            pivot_index=pivot_index,
            direction="forward",
            threshold=temporal_tolerance,
        )

        # 4. Lấy ra frame object rồi wrap thành KeyframeServiceReponse
        def to_service_response(frame, score) -> KeyframeServiceReponse:
            return KeyframeServiceReponse(
                key=frame.key,
                video_num=frame.video_num,
                group_num=frame.group_num,
                keyframe_num=frame.keyframe_num,
                confidence_score=score
            )

        start_frame = to_service_response(sorted_keyframes[start_idx], start_score)
        end_frame = to_service_response(sorted_keyframes[end_idx], end_score)

        return start_frame, end_frame
