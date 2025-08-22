import torch
import numpy as np
from pymilvus import Collection, connections, FieldSchema, CollectionSchema, DataType, utility
from typing import Optional
from tqdm import tqdm
import argparse
import json
from sentence_transformers import SentenceTransformer

import sys
import os
ROOT_FOLDER = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')
)
sys.path.insert(0, ROOT_FOLDER)



from app.core.settings import OcrIndexMilvusSetting


class MilvusOcrInjector:
    def __init__(
        self,
        setting: OcrIndexMilvusSetting,
        collection_name: str,
        host: str = "localhost",
        port: str = "19530",
        user: str = "",
        password: str = "",
        db_name: str = "default",
        alias: str = "default"

    ):
        self.setting = setting
        self.collection_name = collection_name
        self.alias = alias

        self._connect(host, port, user, password, db_name, alias)

    def _connect(self, host: str, port: str, user: str, password: str, db_name: str, alias: str):

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
        print(f"Connected to Milvus at {host}:{port}")


    def create_collection(self, embedding_dim: int, index_params: Optional[dict] = None):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="keyframe_id", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim)
        ]

        schema = CollectionSchema(fields, f"Collection for {self.collection_name} embeddings")

        collection = Collection(self.collection_name, schema, using=self.alias)
        print(f"Created collection '{self.collection_name}' with dimension {embedding_dim}")

        if index_params is None:
            index_params = {
                "metric_type": self.setting.METRIC_TYPE,
                "index_type": self.setting.INDEX_TYPE,
            }

        collection.create_index("embedding", index_params)
        print("Created index for embedding field")

        return collection

    def inject_ocr_data(
        self, 
        embedding_file_path: str, 
        batch_size: int = 1000,
    ):
        print(f"Loading embeddings from {embedding_file_path}")

        embeddings = None
        ids = None

        if embedding_file_path.endswith(".pkl"):
            import pickle
            with open(embedding_file_path, "rb") as f:
                data = pickle.load(f)

            # data: list of dict
            ids = [item["keyframe_id"] for item in data]
            embeddings = np.array([item["embedding"] for item in data], dtype=np.float32)

        else:  # torch tensor file (.pt, .pth)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            embeddings = torch.load(
                embedding_file_path,
                map_location=device,
                weights_only=False
            )
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.cpu().numpy()
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)
            ids = list(range(len(embeddings)))

        num_vectors, embedding_dim = embeddings.shape
        print(f"Loaded {num_vectors} embeddings with dimension {embedding_dim}")

        # Nếu collection tồn tại thì drop
        if utility.has_collection(self.collection_name, using=self.alias):
            print(f"Dropping existing collection '{self.collection_name}' before creation...")
            utility.drop_collection(self.collection_name, using=self.alias)

        collection = self.create_collection(embedding_dim)

        print(f"Inserting {num_vectors} embeddings in batches of {batch_size}")
        for i in tqdm(range(0, num_vectors, batch_size), desc="Inserting batches"):
            end_idx = min(i + batch_size, num_vectors)
            batch_embeddings = embeddings[i:end_idx].tolist()
            batch_ids = ids[i:end_idx]
            entities = [batch_ids, batch_embeddings]
            collection.insert(entities)

        collection.flush()
        print("Data flushed to disk")

        collection.load()
        print("Collection loaded for search")

        return collection

    def get_collection_info(self):

        collection = Collection(self.collection_name, using=self.alias)
        num_entities = collection.num_entities
        print(f"Collection '{self.collection_name}' has {num_entities} entities")
        return num_entities


    def disconnect(self):
        if connections.has_connection(self.alias):
            connections.remove_connection(self.alias)
            print("Disconnected from Milvus")


def inject_ocr_data_simple(
    embedding_file_path: str,
    setting: OcrIndexMilvusSetting
):
    injector = MilvusOcrInjector(
        setting=setting,
        collection_name=setting.COLLECTION_NAME,
        host=setting.HOST,
        port=setting.PORT,
        alias="ocr"
    )

    injector.inject_ocr_data(
        embedding_file_path=embedding_file_path,
        batch_size=setting.BATCH_SIZE
    )
    count = injector.get_collection_info()
    print(f"Successfully injected ocr data! Total entities: {count}")



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Migrate OCR data to Milvus.")
    parser.add_argument(
        "--file_path", type=str, help="Path to data."
    )
    args = parser.parse_args()

    setting =  OcrIndexMilvusSetting()
    inject_ocr_data_simple(
        embedding_file_path=args.file_path,
        setting=setting
    )