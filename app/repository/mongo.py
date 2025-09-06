"""
The implementation of Keyframe repositories. The following class is responsible for getting the keyframe by many ways
"""

import os
import sys
ROOT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), '../'
    )
)

sys.path.insert(0, ROOT_DIR)

from typing import Any, TypeVar
from beanie import Document
from common.repository import MongoBaseRepository
from schema.interface import KeyframeInterface


BeanieDocument = TypeVar('BeanieDocument', bound=Document)


class KeyframeRepository(MongoBaseRepository[BeanieDocument]):
    async def get_keyframe_by_list_of_keys(
        self, keys: list[int]
    ):
        result = await self.find({"key": {"$in": keys}})
        return [
            KeyframeInterface(
                key=keyframe.key,
                video_num=keyframe.video_num,
                group_num=keyframe.group_num,
                keyframe_num=keyframe.keyframe_num
            ) for keyframe in result

        ]

    async def get_keyframes_by_pivot(
        self,
        pivot_frame: KeyframeInterface,
    ):
        video_num = pivot_frame.video_num
        group_num = pivot_frame.group_num

        # Truy vấn tất cả keyframes cùng video_num và group_num (AND)
        result = await self.find({
            "$and": [
                {"video_num": video_num},
                {"group_num": group_num}
            ]
        })

        return [
            KeyframeInterface(
                key=keyframe.key,
                video_num=keyframe.video_num,
                group_num=keyframe.group_num,
                keyframe_num=keyframe.keyframe_num
            )
            for keyframe in result
        ]

    async def get_keyframe_by_keyframe_num(
        self, 
        keyframe_num: int,
    ):
        result = await self.find({"keyframe_num": keyframe_num})
        return [
            KeyframeInterface(
                key=keyframe.key,
                video_num=keyframe.video_num,
                group_num=keyframe.group_num,
                keyframe_num=keyframe.keyframe_num
            ) for keyframe in result
        ]   


