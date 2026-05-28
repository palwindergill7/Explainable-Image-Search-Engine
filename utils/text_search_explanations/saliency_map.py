import torch
import numpy as np
import matplotlib.pyplot as plt

def generate_saliency_map(image_tensor_data, query, model):
    """Generates a Saliency Map visualization."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.train()
    for param in model.vision_encoder.parameters():
        param.requires_grad = True

    image_tensor = image_tensor_data.clone().requires_grad_(True).to(device)
    image_tensor.retain_grad()
    
    with torch.no_grad():
        text_embeddings = model.encode_text(query)
        text_embeddings_norm = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

    with torch.enable_grad():
        forward_output = model.vision_encoder.forward_features(image_tensor)
        x_norm = forward_output["x_norm_1st_clstoken"]
        image_embedding = x_norm.squeeze(0)
        image_embedding_norm = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
        
        similarity_score = torch.matmul(image_embedding_norm, text_embeddings_norm.T.to(device)).squeeze()
        
        model.zero_grad()
        similarity_score.backward() 

    saliency_map = None
    if image_tensor.grad is not None:
        saliency = image_tensor.grad.data.abs().squeeze().cpu()
        saliency, _ = torch.max(saliency, dim=0)
        saliency_map = saliency.numpy()

    model.eval()
    return saliency_map
