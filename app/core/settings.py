from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv
load_dotenv()


class MongoDBSettings(BaseSettings):
    MONGO_HOST: str = Field(..., alias='MONGO_HOST')
    MONGO_PORT: int = Field(..., alias='MONGO_PORT')
    MONGO_DB: str = Field(..., alias='MONGO_DB')
    MONGO_USER: str = Field(..., alias='MONGO_USER')
    MONGO_PASSWORD: str = Field(..., alias='MONGO_PASSWORD')





class IndexPathSettings(BaseSettings):
    FAISS_INDEX_PATH: str | None  
    USEARCH_INDEX_PATH: str | None





class KeyFrameIndexMilvusSetting(BaseSettings):
    COLLECTION_NAME: str = "keyframe"
    HOST: str = 'localhost'
    PORT: str = '19530'
    METRIC_TYPE: str = 'COSINE'
    INDEX_TYPE: str = 'FLAT'
    BATCH_SIZE: int =10000
    SEARCH_PARAMS: dict = {}


class KeyFrameIndexMilvusSettingBatch2(KeyFrameIndexMilvusSetting):
    COLLECTION_NAME: str = "keyframe_batch2"


class KeyFrameIndexMilvusSettingBatch3(KeyFrameIndexMilvusSetting):
    COLLECTION_NAME: str = "keyframe_batch3"


class MilvusSettings(BaseSettings):
    batch_1: KeyFrameIndexMilvusSetting = KeyFrameIndexMilvusSetting()
    batch_2: KeyFrameIndexMilvusSettingBatch2 = KeyFrameIndexMilvusSettingBatch2()
    batch_3: KeyFrameIndexMilvusSettingBatch3 = KeyFrameIndexMilvusSettingBatch3()


class OcrIndexMilvusSetting(BaseSettings):
    COLLECTION_NAME: str = "ocr"
    HOST: str = 'localhost'
    PORT: str = '19530'
    METRIC_TYPE: str = 'COSINE'
    INDEX_TYPE: str = 'FLAT'
    BATCH_SIZE: int =10000
    SEARCH_PARAMS: dict = {}


class AppSettings(BaseSettings):
    DATA_FOLDER: str = "C:/HCMAI2025_Baseline/Data"
    DATA_FOLDER_BATCH_2: str | None = None
    DATA_FOLDER_BATCH_3: str | None = None
    ID2INDEX_PATH: str = "id2index.json"
    MODEL_NAME: str = "hf-hub:laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg-soup"
    MODEL_OCR_NAME: str = "all-MiniLM-L6-v2"
    USE_CACHE: bool = Field(False, description="Enable caching for search and rerank queries")