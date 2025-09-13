from beanie import Document, Indexed
from typing import Annotated

def keyframe_model_factory(collection_name: str):
    class Keyframe(Document):
        key: Annotated[int, Indexed(unique=True)]
        video_num: Annotated[int, Indexed()]
        group_num: Annotated[int, Indexed()]
        keyframe_num: Annotated[int, Indexed()]

        class Settings:
            name = collection_name

    return Keyframe
