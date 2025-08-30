import torch

file_path = "all_embeddings.pt"
embeddings = torch.load(file_path, map_location="cuda")

print(type(embeddings))

print(embeddings.shape)