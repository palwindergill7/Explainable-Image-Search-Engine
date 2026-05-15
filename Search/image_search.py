import torch
from pathlib import Path
from PIL import Image
import numpy as np
from torchvision import transforms

def search_images_by_image(input_image, image_embeddings, image_paths, model, top_k=50):
    """Search images based on an input image."""
    if isinstance(input_image, str):
        input_image = Image.open(input_image).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
    ])
    input_tensor = transform(input_image).unsqueeze(0)
    
    with torch.no_grad():
        image_features = model.encode_image(input_tensor.to(model.device))

    query_embedding = image_features.cls_token.squeeze()

    image_embeddings_tensor = torch.tensor(image_embeddings).squeeze(1).to(model.device)
    
    image_embeddings_norm = image_embeddings_tensor / image_embeddings_tensor.norm(dim=-1, keepdim=True)
    query_embedding_norm = query_embedding / query_embedding.norm(dim=-1, keepdim=True)
    
    similarity_scores = torch.matmul(image_embeddings_norm, query_embedding_norm.T).squeeze()
    
    top_k_indices = torch.topk(similarity_scores, k=min(top_k, len(image_paths))).indices
    
    results = []
    for idx in top_k_indices:
        idx = idx.item()
        image_path = image_paths[idx]
        results.append({
            'image_path': image_path,
            'score': similarity_scores[idx].item(),
            'filename': Path(image_path).name
        })
    
    return results
    