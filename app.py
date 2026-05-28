import streamlit as st
import os
import numpy as np
from PIL import Image
import torch
from transformers import AutoModel
from torchvision import transforms
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
from shapely.geometry import LineString, Point
import cv2

from Search.search import search_images
from Search.image_search import search_images_by_image
from utils.text_search_explanations.saliency_map import generate_saliency_map

from utils.image_search_explanations.image_saliency_map import generate_saliency_map_for_image
from utils.Image_Composition.ICC import extract_icc, draw_icc_overlay
from utils.feature_visualization.color_histogram import compute_histogram, draw_histogram

warnings.filterwarnings('ignore')

# --- Session State Initialization ---
if 'search_triggered' not in st.session_state:
    st.session_state.search_triggered = False
if 'results' not in st.session_state:
    st.session_state.results = []
if 'show_explanation' not in st.session_state:
    st.session_state.show_explanation = None # Will store the index of the image to explain
if 'show_grad_cam' not in st.session_state:
    st.session_state.show_grad_cam = None
if 'search_type' not in st.session_state:
    st.session_state.search_type = None
if 'show_image_explanation' not in st.session_state:
    st.session_state.show_image_explanation = None
if 'show_icc' not in st.session_state:
    st.session_state.show_icc = None
if 'show_color_histogram' not in st.session_state:
    st.session_state.show_color_histogram = None

# --- Part 1: Initial Loading of Website ---

@st.cache_resource
def load_model_and_data():
    """Loads the TIPSv2 model, image embeddings, and image paths."""
    st.write("Loading TIPSv2 model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained("google/tipsv2-b14", trust_remote_code=True, dtype="auto").to(device)
    model.eval() # Set model to evaluation mode
    st.write("Model loaded successfully!")

    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
    ])

    # Load embeddings from cache
    cache_path = "Data/embeddings_cache.pt"
    if os.path.exists(cache_path):
        st.write("Loading embeddings from cache...")
        cache = torch.load(cache_path, weights_only=False)
        image_embeddings = cache["embeddings"]
        image_paths = cache["image_paths"]
        st.write(f"Loaded {len(image_embeddings)} embeddings.")
    else:
        st.error("Embeddings cache not found. Please make sure 'Data/embeddings_cache.pt' exists.")
        return None, None, None, None

    return model, transform, image_embeddings, image_paths

model, transform, image_embeddings, image_paths = load_model_and_data()

st.title("XAI iArt")
st.set_page_config(layout="wide")

# --- Sidebar for search options ---
st.sidebar.title("Search Options")
search_type = st.sidebar.radio("Choose search type:", ("Text", "Image"))

query = None
uploaded_file = None

if search_type == "Text":
    query = st.sidebar.text_input("Enter your search query:", "")
    if st.sidebar.button("Search by Text"):
        st.session_state.search_triggered = True
        st.session_state.search_type = "text"
        st.session_state.show_explanation = None
        st.session_state.show_grad_cam = None
else:
    uploaded_file = st.sidebar.file_uploader("Upload an image for search:", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        st.session_state.search_triggered = True
        st.session_state.search_type = "image"
        st.session_state.show_explanation = None
        st.session_state.show_grad_cam = None
        st.session_state.show_image_explanation = None
        st.session_state.show_icc = None
        st.session_state.show_color_histogram = None

# --- Part 2: Search Query and Result Loading ---

if st.session_state.search_triggered and model:
    if st.session_state.search_type == "text" and query:
        st.session_state.results = search_images(query, image_embeddings, image_paths, model)
        st.subheader(f"Search Results for: '{query}'")
    elif st.session_state.search_type == "image" and uploaded_file:
        query_image = Image.open(uploaded_file).convert("RGB")
        st.sidebar.image(query_image, caption="Uploaded Image", width=200)
        st.session_state.results = search_images_by_image(query_image, image_embeddings, image_paths, model)
        st.subheader("Search Results for Uploaded Image")

    if not st.session_state.results:
        st.write("No results found.")
    else:
        cols = st.columns(5)
        for i, result in enumerate(st.session_state.results):
            with cols[i % 5]:
                img = Image.open(result['image_path']).convert("RGB")
                img = img.resize((300, 300))
                st.image(img, caption=f"{result['filename']}\nScore: {result['score']:.3f}")

                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"ICC Explanation", key=f"icc_{i}"):
                        st.session_state.show_icc = i
                
                with col2:
                    if st.button(f"Color Histogram", key=f"color_hist_{i}"):
                        st.session_state.show_color_histogram = i

                if st.session_state.search_type == "text":
                    if st.button(f"Saliency Map", key=f"explain_{i}"):
                        st.session_state.show_explanation = i
                    # if st.button(f"Grad-CAM", key=f"grad_cam_{i}"):
                    #     st.session_state.show_grad_cam = i
                elif st.session_state.search_type == "image":
                    if st.button(f"Image Saliency Map", key=f"image_explain_{i}"):
                        st.session_state.show_image_explanation = i

# --- Part 3: Explanation Loading ---

@st.dialog("Saliency Map Explanation")
def show_explanation_dialog(result_to_explain, query, model, transform):
    image = Image.open(result_to_explain['image_path']).convert("RGB")
    image_tensor_data = transform(image).unsqueeze(0)
    
    saliency_map = generate_saliency_map(image_tensor_data, query, model)
    
    if saliency_map is not None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Original Image
        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        axes[0].axis('off')

        # Saliency Map
        axes[1].imshow(saliency_map, cmap='hot')
        axes[1].set_title("Saliency Map")
        axes[1].axis('off')
        
        st.pyplot(fig)
    else:
        st.error("Could not generate Saliency Map.")

    if st.button("Back to results"):
        st.session_state.show_explanation = None
@st.dialog("Grad-CAM Explanation")
def show_grad_cam_dialog(result_to_explain, query, model, transform):
    image = Image.open(result_to_explain['image_path']).convert("RGB")
    overlay_image = generate_grad_cam_overlay(image, query, model, transform)
    
    st.image(overlay_image, caption="Grad-CAM Overlay", use_column_width=True)

    if st.button("Back to results"):
        st.session_state.show_grad_cam = None
        st.rerun()

if st.session_state.show_explanation is not None:
    result_to_explain = st.session_state.results[st.session_state.show_explanation]
    show_explanation_dialog(result_to_explain, query, model, transform)

if st.session_state.show_grad_cam is not None:
    result_to_explain = st.session_state.results[st.session_state.show_grad_cam]
    show_grad_cam_dialog(result_to_explain, query, model, transform)

@st.dialog("ICC Explanation")
def show_icc_dialog(result_to_explain):
    image = Image.open(result_to_explain['image_path']).convert("RGB")
    
    # Resize image for display, maintaining aspect ratio
    max_width = 800
    width, height = image.size
    if width > max_width:
        new_height = int(max_width * height / width)
        display_image = image.resize((max_width, new_height))
    else:
        display_image = image

    image_np = np.array(image)
    display_image_np = np.array(display_image)
    
    # Extract ICC from original full-res image for accuracy
    icc_data = extract_icc(image_np)

    # Scale ICC data to match the display image size
    scale_x = display_image.width / image.width
    scale_y = display_image.height / image.height

    scaled_icc_data = {
        'poselines': [LineString([(p[0] * scale_x, p[1] * scale_y) for p in line.coords]) for line in icc_data['poselines']],
        'action_lines': [LineString([(p[0] * scale_x, p[1] * scale_y) for p in line.coords]) for line in icc_data['action_lines']],
        'action_centers': [Point(p.x * scale_x, p.y * scale_y) for p in icc_data['action_centers']]
    }

    overlay_image = draw_icc_overlay(display_image_np, scaled_icc_data)
    
    st.image(overlay_image, caption="ICC Overlay")

    if st.button("Back to results"):
        st.session_state.show_icc = None

if st.session_state.show_icc is not None:
    result_to_explain = st.session_state.results[st.session_state.show_icc]
    show_icc_dialog(result_to_explain)

@st.dialog("Color Histogram Explanation")
def show_color_histogram_dialog(result_to_explain):
    image_path = result_to_explain['image_path']
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    # Convert RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    # Get image metadata
    filename = Path(image_path).name
    file_size = os.path.getsize(image_path) // 1024  # Size in KB
    image_size = image.size  # (width, height)
    
    hist_r, hist_g, hist_b, stats = compute_histogram(image_bgr)
    fig = draw_histogram(hist_r, hist_g, hist_b, stats, filename, file_size, image_size)
    
    st.pyplot(fig)

    if st.button("Back to results"):
        st.session_state.show_color_histogram = None

if st.session_state.show_color_histogram is not None:
    result_to_explain = st.session_state.results[st.session_state.show_color_histogram]
    show_color_histogram_dialog(result_to_explain)

@st.dialog("Image Saliency Map Explanation")
def show_image_explanation_dialog(result_to_explain, query_image, model):
    target_image = Image.open(result_to_explain['image_path']).convert("RGB")
    
    saliency_map = generate_saliency_map_for_image(target_image, query_image, model)
    
    if saliency_map is not None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Query Image
        axes[0].imshow(query_image)
        axes[0].set_title("Query Image")
        axes[0].axis('off')

        # Original Result Image
        axes[1].imshow(target_image)
        axes[1].set_title("Result Image")
        axes[1].axis('off')

        # Saliency Map
        axes[2].imshow(saliency_map, cmap='hot')
        axes[2].set_title("Saliency Map on Result")
        axes[2].axis('off')
        
        st.pyplot(fig)
    else:
        st.error("Could not generate Saliency Map.")

    if st.button("Back to results"):
        st.session_state.show_image_explanation = None

if st.session_state.show_image_explanation is not None:
    result_to_explain = st.session_state.results[st.session_state.show_image_explanation]
    query_image = Image.open(uploaded_file).convert("RGB")
    show_image_explanation_dialog(result_to_explain, query_image, model)
