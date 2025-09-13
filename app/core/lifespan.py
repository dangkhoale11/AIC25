from contextlib import asynccontextmanager
from fastapi import FastAPI
from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie

import os
import sys
ROOT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), '../'
    )
)
sys.path.insert(0, ROOT_DIR)

from core.settings import MongoDBSettings, KeyFrameIndexMilvusSetting, AppSettings
from factory.factory import ServiceFactory
from core.logger import SimpleLogger
from models.factories import keyframe_model_factory

mongo_client: AsyncIOMotorClient = None
logger = SimpleLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager for startup and shutdown events
    """
    logger.info("Starting up application...")

    try:
        # ==== Load settings ====
        mongo_settings = MongoDBSettings()
        milvus_settings = KeyFrameIndexMilvusSetting()
        appsetting = AppSettings()

        # ==== MongoDB connect ====
        global mongo_client
        mongo_connection_string = (
            f"mongodb://{mongo_settings.MONGO_USER}:{mongo_settings.MONGO_PASSWORD}"
            f"@{mongo_settings.MONGO_HOST}:{mongo_settings.MONGO_PORT}"
        )
        mongo_client = AsyncIOMotorClient(mongo_connection_string)
        await mongo_client.admin.command("ping")
        logger.info("Successfully connected to MongoDB")

        database = mongo_client[mongo_settings.MONGO_DB]

        # ==== Khởi tạo Beanie models ====
        batches = [1, 2]  # có thể đọc từ config/env
        models = {b: keyframe_model_factory(f"keyframe_batch{b}") for b in batches}

        await init_beanie(
            database=database,
            document_models=list(models.values())
        )
        logger.info("Beanie initialized successfully")

        # ==== Lưu models & service factories vào app.state ====
        app.state.models = models
        app.state.service_factories = {}

        for batch in batches:
            logger.info(f"Initializing service factory for batch {batch}...")

            milvus_search_params = {
                "metric_type": milvus_settings.METRIC_TYPE,
                "params": milvus_settings.SEARCH_PARAMS,
            }

            service_factory = ServiceFactory(
                batch=batch,
                mongo_keyframe_model=app.state.models[batch],  # 👈 truyền thẳng model
                milvus_host=milvus_settings.HOST,
                milvus_port=milvus_settings.PORT,
                milvus_user="",
                milvus_password="",
                milvus_search_params=milvus_search_params,
                model_name=appsetting.MODEL_NAME,
            )

            app.state.service_factories[batch] = service_factory
            logger.info(f"Service factory for batch {batch} initialized successfully")

        app.state.mongo_client = mongo_client

        logger.info("Application startup completed successfully")

    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise

    yield

    # ==== Shutdown ====
    logger.info("Shutting down application...")
    try:
        if mongo_client:
            mongo_client.close()
            logger.info("MongoDB connection closed")
        logger.info("Application shutdown completed successfully")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")
