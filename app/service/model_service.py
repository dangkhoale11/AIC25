import torch
import numpy as np


class ModelService:
    def __init__(self, model, preprocess, tokenizer, device=None):
        # Nếu không truyền device thì tự chọn GPU nếu có, không thì CPU
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.model.eval()
    
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

            