import torch
from PIL import Image
from torchvision import transforms

def generate_saliency_map_for_image(target_image, input_image, model):
    """
    Generates a Saliency Map visualization for image-to-image search.

    This highlights which parts of the target_image were most influential
    in matching the input_image.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.train()
    for param in model.parameters():
        param.requires_grad = True

    # --- Prepare target image ---
    # We want to find the saliency map on the target image.
    # So we need to calculate gradients with respect to it.
    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
    ])
    target_tensor = transform(target_image).unsqueeze(0).clone().requires_grad_(True).to(device)
    target_tensor.retain_grad()

    # --- Prepare input image (the "query" image) ---
    # We don't need gradients for the input image, just its embedding.
    input_tensor = transform(input_image).unsqueeze(0).to(device)
    with torch.no_grad():
        input_embedding = model.encode_image(input_tensor).cls_token.squeeze()
        input_embedding_norm = input_embedding / input_embedding.norm(dim=-1, keepdim=True)

    # --- Calculate Similarity and Gradients ---
    with torch.enable_grad():
        # Get the embedding of the target image
        target_features = model.vision_encoder.forward_features(target_tensor)
        target_embedding = target_features["x_norm_1st_clstoken"].squeeze(0)
        target_embedding_norm = target_embedding / target_embedding.norm(dim=-1, keepdim=True)

        # Calculate similarity score
        similarity_score = torch.matmul(target_embedding_norm, input_embedding_norm.T).squeeze()

        # Backpropagate to get gradients
        model.zero_grad()
        similarity_score.backward()

    # --- Generate Saliency Map ---
    saliency_map = None
    if target_tensor.grad is not None:
        saliency = target_tensor.grad.data.abs().squeeze()
        saliency, _ = torch.max(saliency, dim=0)
        saliency_map = saliency.cpu().numpy()

    model.eval()
    return saliency_map
