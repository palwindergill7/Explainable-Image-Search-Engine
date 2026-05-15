import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output
            
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_cam(self, image_tensor, query):
        self.model.train() # Switch to train mode for hooks
        device = next(self.model.parameters()).device
        image_tensor = image_tensor.to(device)

        with torch.no_grad():
            text_embeddings = self.model.encode_text(query)
            text_embeddings_norm = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

        with torch.enable_grad():
            image_tensor.requires_grad_(True)
            
            # Forward pass to get image embeddings
            forward_output = self.model.vision_encoder.forward_features(image_tensor)
            x_norm = forward_output["x_norm_1st_clstoken"]
            image_embedding = x_norm.squeeze(0)
            image_embedding_norm = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
            
            # Compute similarity score
            similarity_score = torch.matmul(image_embedding_norm, text_embeddings_norm.T.to(device)).squeeze()

        self.model.zero_grad()
        similarity_score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Could not get gradients or activations")

        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations.detach()

        # The number of channels in the activations
        num_channels = activations.shape[1]

        # Reshape pooled_gradients to be broadcastable
        pooled_gradients = pooled_gradients.view(1, num_channels, 1, 1)

        # Multiply activations by the reshaped gradients
        activations *= pooled_gradients
            
        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        heatmap /= torch.max(heatmap)
        
        self.model.eval() # Switch back to eval mode
        return heatmap.cpu().numpy()

def generate_grad_cam_overlay(image, query, model, transform):
    """Generates a Grad-CAM overlay for a given image and query."""
    # The target layer might need adjustment depending on the exact model architecture
    # For TIPS-v2, the last block of the vision encoder is a good choice.
    target_layer = model.vision_encoder.blocks[-1].norm1
    
    grad_cam = GradCAM(model, target_layer)
    
    image_tensor = transform(image).unsqueeze(0)
    
    heatmap = grad_cam.generate_cam(image_tensor, query)
    
    # Create overlay
    original_img_np = np.array(image)
    heatmap_resized = cv2.resize(heatmap, (original_img_np.shape[1], original_img_np.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    
    overlayed_image = cv2.addWeighted(cv2.cvtColor(original_img_np, cv2.COLOR_RGB2BGR), 0.6, heatmap_colored, 0.4, 0)
    overlayed_image = cv2.cvtColor(overlayed_image, cv2.COLOR_BGR2RGB)
    
    return Image.fromarray(overlayed_image)
