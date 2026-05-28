import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os

def compute_histogram(image):
    """Computes the color histogram for an image and returns statistics."""
    hist_b = cv2.calcHist([image], [0], None, [25], [0, 256]).flatten()
    hist_g = cv2.calcHist([image], [1], None, [25], [0, 256]).flatten()
    hist_r = cv2.calcHist([image], [2], None, [25], [0, 256]).flatten()
    
    # Compute statistics
    pixels = image.reshape(-1, 3)
    count = len(pixels)
    r_mean = np.mean(pixels[:, 2])
    g_mean = np.mean(pixels[:, 1])
    b_mean = np.mean(pixels[:, 0])
    r_std = np.std(pixels[:, 2])
    g_std = np.std(pixels[:, 1])
    b_std = np.std(pixels[:, 0])
    
    return hist_r, hist_g, hist_b, {
        'count': count,
        'r_mean': r_mean,
        'g_mean': g_mean,
        'b_mean': b_mean,
        'r_std': r_std,
        'g_std': g_std,
        'b_std': b_std
    }

def draw_histogram(hist_r, hist_g, hist_b, stats, filename, file_size, image_size):
    """Draws the color histogram with the reference image layout."""
    fig = plt.figure(figsize=(10, 10))
    fig.patch.set_facecolor('white')
    
    # Title with image metadata
    title_text = f"{filename}\n{image_size[0]}x{image_size[1]} pixels; RGB; {file_size}K"
    fig.suptitle(title_text, fontsize=10, fontweight='bold')
    
    channels = [hist_r, hist_g, hist_b]
    colors = ['red', 'green', 'blue']
    labels = ['R', 'G', 'B']
    
    for idx, (hist, color, label) in enumerate(zip(channels, colors, labels)):
        # Create subplot
        ax = fig.add_subplot(3, 1, idx + 1)
        ax.set_facecolor('white')
        
        # Plot histogram as filled black area with 25 bins
        bin_width = 256 / 25
        bin_edges = np.arange(0, 256 + bin_width, bin_width)
        bin_centers = bin_edges[:-1] + bin_width / 2
        
        ax.bar(bin_centers, hist, width=bin_width * 0.9, color='black', edgecolor='black')
        
        # Set axis properties
        ax.set_xlim(0, 255)
        ax.set_ylim(0, np.max(hist) * 1.1)
        ax.set_xlabel('Intensity')
        ax.set_ylabel('Frequency')
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add gradient bar under the histogram
        gradient = np.linspace(0, 1, 256).reshape(1, -1)
        if color == 'red':
            cmap = LinearSegmentedColormap.from_list('red_gradient', ['black', 'red'])
        elif color == 'green':
            cmap = LinearSegmentedColormap.from_list('green_gradient', ['black', 'green'])
        else:  # blue
            cmap = LinearSegmentedColormap.from_list('blue_gradient', ['black', 'blue'])
        
        # Create a small axis for the gradient bar
        ax_gradient = ax.twinx()
        ax_gradient.imshow(gradient, aspect='auto', extent=[0, 255, 0, 1], cmap=cmap, alpha=0.7)
        ax_gradient.set_yticks([])
        ax_gradient.set_xlim(0, 255)
    
    # Add statistics text
    stats_text = f"Count: {int(stats['count']):,}\nrMean: {stats['r_mean']:.2f}\t\trStdDev: {stats['r_std']:.2f}\ngMean: {stats['g_mean']:.2f}\t\tgStdDev: {stats['g_std']:.2f}\nbMean: {stats['b_mean']:.2f}\t\tbStdDev: {stats['b_std']:.2f}"
    fig.text(0.1, 0.02, stats_text, fontsize=9, verticalalignment='bottom', fontfamily='monospace')
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    
    return fig
