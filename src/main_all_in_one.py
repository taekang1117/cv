"""
All-in-One Gesture Recognition System (Laptop Version)

Handles:
- Kinect depth capture
- Depth thresholding
- Feature extraction
- Template saving/loading
- Gesture recognition
- Debug visualization

Press:
    'r' → record template
    'q' → quit
"""

import cv2
import numpy as np
import os
import freenect
import time


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DEPTH_MIN = 500
DEPTH_MAX = 1200
MATCH_THRESHOLD = 0.5
TEMPLATE_FOLDER = "templates"


# --------------------------------------------------
# KINECT FUNCTIONS
# --------------------------------------------------

def get_depth_frame():
    depth_frame, _ = freenect.sync_get_depth()

    if depth_frame is None:
        return None

    return depth_frame.astype(np.uint16)


# --------------------------------------------------
# PREPROCESSING
# --------------------------------------------------

def isolate_hand(depth_frame):

    mask = np.logical_and(
        depth_frame > DEPTH_MIN,
        depth_frame < DEPTH_MAX
    )

    mask = mask.astype(np.uint8) * 255

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask


# --------------------------------------------------
# FEATURE EXTRACTION
# --------------------------------------------------

def extract_features(mask):

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return None

    largest = max(contours, key=cv2.contourArea)

    area = cv2.contourArea(largest)
    perimeter = cv2.arcLength(largest, True)

    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None

    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]

    x, y, w, h = cv2.boundingRect(largest)
    aspect_ratio = w / float(h) if h != 0 else 0

    hull = cv2.convexHull(largest)
    hull_area = cv2.contourArea(hull)
    solidity = area / float(hull_area) if hull_area != 0 else 0

    edges = cv2.Canny(mask, 100, 200)
    edge_density = np.sum(edges > 0) / float(np.sum(mask > 0) + 1)

    features = np.array([
        cx,
        cy,
        area,
        perimeter,
        aspect_ratio,
        solidity,
        edge_density
    ], dtype=np.float32)

    # Normalize feature vector
    norm = np.linalg.norm(features)
    if norm != 0:
        features = features / norm

    return features


# --------------------------------------------------
# TEMPLATE MANAGEMENT
# --------------------------------------------------

def load_templates():

    templates = []

    if not os.path.exists(TEMPLATE_FOLDER):
        os.makedirs(TEMPLATE_FOLDER)
        return templates

    for file in os.listdir(TEMPLATE_FOLDER):
        if file.endswith(".npy"):
            label = file.replace(".npy", "")
            features = np.load(os.path.join(TEMPLATE_FOLDER, file))
            templates.append({
                "label": label,
                "features": features
            })

    print("Loaded templates:", [t["label"] for t in templates])
    return templates


def save_template(mask, label):

    features = extract_features(mask)

    if features is None:
        print("Failed to extract features.")
        return

    if not os.path.exists(TEMPLATE_FOLDER):
        os.makedirs(TEMPLATE_FOLDER)

    np.save(os.path.join(TEMPLATE_FOLDER, f"{label}.npy"), features)

    print("Template saved for:", label)


# --------------------------------------------------
# RECOGNITION
# --------------------------------------------------

def match_gesture(features, templates):

    if features is None:
        return "unknown"

    best_label = "unknown"
    best_score = float("inf")

    for template in templates:

        template_features = template["features"]

        score = np.linalg.norm(features - template_features)

        if score < best_score:
            best_score = score
            best_label = template["label"]

    if best_score < MATCH_THRESHOLD:
        return best_label
    else:
        return "unknown"


# --------------------------------------------------
# MAIN LOOP
# --------------------------------------------------

print("Starting Gesture Recognition System")
print("Press 'r' to record a new template")
print("Press 'q' to quit")

templates = load_templates()

while True:

    depth_frame = get_depth_frame()

    if depth_frame is None:
        print("No Kinect detected.")
        time.sleep(0.5)
        continue

    mask = isolate_hand(depth_frame)
    features = extract_features(mask)

    gesture = match_gesture(features, templates)

    print("Detected:", gesture)

    # Display depth and mask
    depth_display = cv2.normalize(
        depth_frame, None, 0, 255, cv2.NORM_MINMAX
    ).astype("uint8")

    cv2.imshow("Depth", depth_display)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    if key == ord('r'):
        label = input("Enter gesture label: ")
        save_template(mask, label)
        templates = load_templates()

cv2.destroyAllWindows()
