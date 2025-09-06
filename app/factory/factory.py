import os
import sys
from sentence_transformers import SentenceTransformer
ROOT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), '../'
    )
)

sys.path.insert(0, ROOT_DIR)



from repository.mongo import KeyframeRepository
from repository.milvus import KeyframeVectorRepository, OcrVectorRepository
from service import KeyframeQueryService, ModelService
from service.temporal_search_service import TemporalSearchService
from models.keyframe import Keyframe
import open_clip
from pymilvus import connections, Collection as MilvusCollection

from core.settings import MilvusSettings, OcrIndexMilvusSetting, AppSettings, KeyFrameIndexMilvusSetting


class ServiceFactory:
    def __init__(
        self,
        milvus_settings: MilvusSettings,
        ocr_milvus_setting: OcrIndexMilvusSetting,
        app_settings: AppSettings,
        mongo_collection=Keyframe,
    ):
        self._mongo_keyframe_repo = KeyframeRepository(collection=mongo_collection)
        self._milvus_settings = milvus_settings
        self._ocr_milvus_setting = ocr_milvus_setting
        self._app_settings = app_settings
        self._model_service = self._init_model_service(app_settings.MODEL_NAME, app_settings.MODEL_OCR_NAME)
        self._repo_cache = {}
        self._ocr_repo_cache = {}

    def _init_milvus_repo(
        self,
        search_params: dict,
        collection_name: str,
        host: str,
        port: str,
        user: str,
        password: str,
        db_name: str = "default",
        alias: str = "default"
    ):
        if connections.has_connection(alias):
            connections.remove_connection(alias)

        conn_params = {
            "host": host,
            "port": port,
            "db_name": db_name
        }

        if user and password:
            conn_params["user"] = user
            conn_params["password"] = password

        connections.connect(alias=alias, **conn_params)
        collection = MilvusCollection(collection_name, using=alias)

        return KeyframeVectorRepository(collection=collection, search_params=search_params)

    def _get_milvus_setting(self, search_mode: str) -> KeyFrameIndexMilvusSetting:
        if search_mode == "batch-1":
            return self._milvus_settings.batch_1
        elif search_mode == "batch-2":
            return self._milvus_settings.batch_2
        else:
            raise ValueError(f"Invalid search_mode: {search_mode}")

    def get_milvus_keyframe_repo(self, search_mode: str = "batch-1") -> KeyframeVectorRepository:
        if search_mode in self._repo_cache:
            return self._repo_cache[search_mode]

        setting = self._get_milvus_setting(search_mode)
        alias = f"default_{search_mode}"

        repo = self._init_milvus_repo(
            search_params=setting.SEARCH_PARAMS,
            collection_name=setting.COLLECTION_NAME,
            host=setting.HOST,
            port=setting.PORT,
            user="",
            password="",
            db_name="default",
            alias=alias
        )
        self._repo_cache[search_mode] = repo
        return repo

    def _init_milvus_ocr_repo(
        self,
        search_params: dict,
        collection_name: str,
        host: str,
        port: str,
        user: str,
        password: str,
        db_name: str = "default",
        alias: str = "ocr"
    ):
        if connections.has_connection(alias):
            connections.remove_connection(alias)

        conn_params = {
            "host": host,
            "port": port,
            "db_name": db_name
        }

        if user and password:
            conn_params["user"] = user
            conn_params["password"] = password

        connections.connect(alias=alias, **conn_params)
        collection = MilvusCollection(collection_name, using=alias)

        return OcrVectorRepository(collection=collection, search_params=search_params)

    def get_milvus_ocr_repo(self) -> OcrVectorRepository:
        if "ocr" in self._ocr_repo_cache:
            return self._ocr_repo_cache["ocr"]

        setting = self._ocr_milvus_setting
        repo = self._init_milvus_ocr_repo(
            search_params=setting.SEARCH_PARAMS,
            collection_name=setting.COLLECTION_NAME,
            host=setting.HOST,
            port=setting.PORT,
            user="",
            password="",
            db_name="default",
            alias="ocr"
        )
        self._ocr_repo_cache["ocr"] = repo
        return repo

    def _init_model_service(self, model_name: str, model_ocr_name: str):
        model, _, preprocess = open_clip.create_model_and_transforms(model_name)
        tokenizer = open_clip.get_tokenizer(model_name)
        model_ocr = SentenceTransformer(model_ocr_name)
        return ModelService(model=model,model_ocr=model_ocr, preprocess=preprocess, tokenizer=tokenizer)

    def get_mongo_keyframe_repo(self) -> KeyframeRepository:
        return self._mongo_keyframe_repo

    def get_model_service(self) -> ModelService:
        return self._model_service

    def get_keyframe_query_service(self, search_mode: str = "batch-1") -> KeyframeQueryService:
        keyframe_vector_repo = self.get_milvus_keyframe_repo(search_mode)
        ocr_vector_repo = self.get_milvus_ocr_repo()

        return KeyframeQueryService(
            keyframe_mongo_repo=self._mongo_keyframe_repo,
            keyframe_vector_repo=keyframe_vector_repo,
            ocr_vector_repo=ocr_vector_repo
        )

    def get_temporal_search_service(self, search_mode: str = "batch-1") -> TemporalSearchService:
        keyframe_vector_repo = self.get_milvus_keyframe_repo(search_mode)
        return TemporalSearchService(
            keyframe_vector_repo=keyframe_vector_repo,
            keyframe_mongo_repo=self._mongo_keyframe_repo,
        )
