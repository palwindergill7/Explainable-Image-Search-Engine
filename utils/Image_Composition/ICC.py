import cv2
import numpy as np
from ultralytics import YOLO
from shapely.geometry import LineString, Point

# Load the YOLOv8-Pose model
model = YOLO('yolov8n-pose.pt')

def extract_icc(image):
    """
    Extracts Image Composition Canvas (ICC) data from an image.

    Args:
        image (np.ndarray): The input image in BGR format.

    Returns:
        dict: A dictionary containing poselines, action_lines, and action_centers.
    """
    h, w, _ = image.shape
    results = model(image)
    keypoints = results[0].keypoints.xy.cpu().numpy()
    
    poselines = []
    if keypoints.shape[0] > 0:
        for person_kps in keypoints:
            # Define body axis from nose to mid-hip
            nose = person_kps[0]
            left_hip = person_kps[11]
            right_hip = person_kps[12]
            
            if np.all(nose > 0) and (np.all(left_hip > 0) or np.all(right_hip > 0)):
                if np.all(left_hip > 0) and np.all(right_hip > 0):
                    mid_hip = (left_hip + right_hip) / 2
                elif np.all(left_hip > 0):
                    mid_hip = left_hip
                else:
                    mid_hip = right_hip
                
                poselines.append(LineString([nose, mid_hip]))

    action_lines = []
    action_centers = []

    # Calculate global action lines and action centers
    if len(poselines) > 1:
        for i in range(len(poselines)):
            for j in range(i + 1, len(poselines)):
                line1 = poselines[i]
                line2 = poselines[j]

                if line1.boundary.is_empty or line2.boundary.is_empty:
                    continue
                
                # Extend lines to get a better representation of action lines
                p1, p2 = list(line1.boundary.geoms)
                p3, p4 = list(line2.boundary.geoms)
                
                # Create extended lines for intersection
                extended_line1 = LineString([p1.coords[0], p2.coords[0]])
                extended_line2 = LineString([p3.coords[0], p4.coords[0]])

                # Find intersection (action center)
                intersection = extended_line1.intersection(extended_line2)
                if isinstance(intersection, Point):
                    # Check if intersection is within image bounds
                    if 0 <= intersection.x < w and 0 <= intersection.y < h:
                        action_centers.append(intersection)
                    
                    # Global action line connects the midpoints of the poselines
                    midpoint1 = line1.interpolate(0.5, normalized=True)
                    midpoint2 = line2.interpolate(0.5, normalized=True)
                    action_lines.append(LineString([midpoint1, midpoint2]))

    return {
        'poselines': poselines,
        'action_lines': action_lines,
        'action_centers': action_centers
    }

def draw_icc_overlay(image, icc_data):
    """
    Draws the ICC overlay on an image.

    Args:
        image (np.ndarray): The input image in BGR format.
        icc_data (dict): A dictionary with ICC data.

    Returns:
        np.ndarray: The image with the ICC overlay.
    """
    overlay = image.copy()
    h, w, _ = image.shape

    # Draw poselines (green)
    for line in icc_data['poselines']:
        x1, y1 = map(int, line.coords[0])
        x2, y2 = map(int, line.coords[1])
        cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 4)

    # Draw global action lines (yellow)
    for line in icc_data['action_lines']:
        x1, y1 = map(int, line.coords[0])
        x2, y2 = map(int, line.coords[1])
        # Clip line to image boundaries
        points = np.array([[x1, y1], [x2, y2]])
        cv2.line(overlay, tuple(points[0]), tuple(points[1]), (0, 255, 255), 4)


    # Draw action centers (cyan dots)
    for center in icc_data['action_centers']:
        x, y = map(int, center.coords[0])
        cv2.circle(overlay, (x, y), 8, (255, 255, 0), -1)

    return overlay
