import torch
from pathlib import Path

def search_images(query, image_embeddings, image_paths, model, top_k=50):
    """Search images based on a text query."""
    with torch.no_grad():
        text_embeddings = model.encode_text(query)
    
    image_embeddings_tensor = torch.tensor(image_embeddings).squeeze(1)
    
    image_embeddings_norm = image_embeddings_tensor / image_embeddings_tensor.norm(dim=-1, keepdim=True)
    text_embeddings_norm = text_embeddings.cpu() / text_embeddings.cpu().norm(dim=-1, keepdim=True)
    
    similarity_scores = torch.matmul(image_embeddings_norm, text_embeddings_norm.T).squeeze()
    
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
