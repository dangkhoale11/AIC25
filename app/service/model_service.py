import torch
import numpy as np

class ModelService:
    def __init__(self, model, model_ocr, preprocess, tokenizer, device=None):
        # Nếu không truyền device thì tự chọn GPU nếu có, không thì CPU
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        
        # Đưa model vision lên GPU/CPU
        self.model = model.to(self.device)
        self.model.eval()
        
        # Đưa model OCR (SentenceTransformer) lên GPU/CPU
        self.model_ocr = model_ocr.to(self.device)
        
        self.preprocess = preprocess
        self.tokenizer = tokenizer

        # ✅ Log device
        print(f"[INFO] Vision model loaded on: {next(self.model.parameters()).device}")
        print(f"[INFO] OCR model loaded on: {self.model_ocr.device}")

    def embedding(self, query_text: str) -> np.ndarray:
        with torch.no_grad():
            text_tokens = self.tokenizer([query_text]).to(self.device)
            query_embedding = (
                self.model.encode_text(text_tokens)
                .cpu()
                .detach()
                .numpy()
                .astype(np.float32)
            )
        return query_embedding
    
    def embedding_ocr(self, query_text: str) -> np.ndarray:
        """Sinh embedding từ OCR text bằng all-MiniLM-L6-v2"""
        with torch.no_grad():
            query_embedding = self.model_ocr.encode(
                query_text,
                convert_to_numpy=True,
                normalize_embeddings=True  # chuẩn hoá vector
            ).astype(np.float32)
        return query_embedding
