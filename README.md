# HCMAI2025_Baseline

A FastAPI-based AI application powered by Milvus for vector search, MongoDB for metadata storage, and MinIO for object storage.

## 🧑‍💻 Getting Started

### Prerequisites
- Docker
- Docker Compose
- Python 3.10
- uv



Convert the global2imgpath.json to this following format(id2index.json)
```json
{
  "0": "1/1/0",
  "1": "1/1/16",
  "2": "1/1/49",
  "3": "1/1/169",
  "4": "1/1/428",
  "5": "1/1/447",
  "6": "1/1/466",
  "7": "1/1/467",
}
```



2. Install uv and setup env
```bash
pip install uv
uv init --python=3.10
uv add aiofiles beanie dotenv fastapi[standard] httpx ipykernel motor nicegui numpy open-clip-torch pydantic-settings pymilvus streamlit torch typing-extensions usearch uvicorn sentence-transformers googletrans==4.0.0-rc1
```

### Change the dataset to format 
Dataset should follow this structure:

```text
Data/
├── L21/
│   ├── V001/
│   │   ├── 000001.webp
│   │   ├── 000002.webp
│   │   └── ...
│   ├── V002/
│   │   ├── 000001.webp
│   │   ├── 000002.webp
│   │   └── ...
│   └── ...
│
├── L22/
│   ├── V001/
│   │   ├── 000001.webp
│   │   ├── 000002.webp
│   │   └── ...
│   └── ...
│
├── L23/
│   └── ...
│
└── L30/
```


### In app/core/settings.py change the file path of DATA_FOLDER and ID2INDEX appropriate with your computer


3. Activate .venv
```bash
source .venv/bin/activate
.venv/Scripts/activte #windown
```
4. Run docker compose
```bash
docker compose up -d
```

5. Data Migration 
OCR isn't used yet
```bash
python migration/embedding_migration.py --file_path <emnedding.pt file>
python migration/keyframe_migration.py --file_path <id2index.json file path>
python migration/ocr_migration.py --file_path <ocr_embeddings.pkl> 
```

6. Run the application

Open 2 tabs

6.1. Run the FastAPI application
```bash
cd gui
streamlit run main.py
```

6.2. Run the Streamlit application
```bash
cd app
python main.py
```



 

