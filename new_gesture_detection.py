"""
gesture_detection.py
--------------------
Webcam gesture detection using Canny edge detection + template matching.

Computer Vision techniques used:
  1. Skin colour masking    — isolates the hand by filtering HSV skin tones
  2. Canny Edge Detection   — extracts the hand silhouette from the masked ROI
  3. Template Matching      — compares live edges to saved gesture templates

Requirements:
    pip install opencv-python numpy

Controls:
    [1]  Capture OPEN PALM  (press 3x for 3 samples)
    [2]  Capture FIST       (press 3x for 3 samples)
    [3]  Capture THUMBS UP  (press 3x for 3 samples)
    [d]  Toggle edge/mask debug view
    [r]  Delete all saved templates and start over
    [q]  Quit

Tips for best results:
  - Capture each gesture 3 times (press the key 3x) from slightly
    different angles — the best score across all samples is used
  - Use consistent lighting when capturing and detecting
  - Keep your hand fully inside the green box
  - Plain background helps but is no longer required thanks to skin masking
"""

import cv2
import numpy as np
import os
import glob

# ============================================================
# SETTINGS
# ============================================================

WEBCAM_INDEX      = 1        # try 0 if wrong camera opens
TEMPLATE_DIR      = "gesture_templates"
ROI_SIZE          = 128      # all edge images resized to this before matching
SAMPLES_PER_GESTURE = 3      # how many captures per gesture key press

CANNY_LOW         = 20
CANNY_HIGH        = 60

# Lower = more detections (may false-positive)
# Raise if wrong gesture keeps triggering
MIN_CONFIDENCE    = 0.30

# ROI box — bottom right, away from face
ROI_X, ROI_Y, ROI_W, ROI_H = 390, 230, 240, 240

# HSV skin colour range — covers most skin tones under typical webcam lighting
# Hue 0-20 catches warm skin; 160-180 catches the wrap-around red end
# Tighter skin ranges — reduce S minimum and V range to cut background
# Fine-tune using the HSV readout crosshair:
#   WITH hand:    note H, S, V  -> set ranges ~+/-10 around those values
#   WITHOUT hand: confirm those values fall outside the range
SKIN_LOWER_1 = np.array([0,   50, 80],  dtype=np.uint8)   # H, S, V minimums
SKIN_UPPER_1 = np.array([20, 200, 230], dtype=np.uint8)   # H, S, V maximums
SKIN_LOWER_2 = np.array([160, 50,  80],  dtype=np.uint8)
SKIN_UPPER_2 = np.array([180, 200, 230], dtype=np.uint8)

GESTURES = {
    ord('1'): ("open_palm", "Open Palm"),
    ord('2'): ("fist",      "Fist"),
    ord('3'): ("thumbs_up", "Thumbs Up"),
}

# ============================================================
# CV TECHNIQUE 1: SKIN COLOUR MASKING
# ============================================================

def get_skin_mask(roi_bgr: np.ndarray) -> np.ndarray:
    """
    CV TECHNIQUE 1: Skin Colour Masking (HSV thresholding)
    -------------------------------------------------------
    Converts the ROI to HSV colour space and thresholds for skin-tone
    hues. This isolates the hand and removes background edges so Canny
    only sees the hand silhouette.

    Two ranges are used because skin hue wraps around 0/180 in OpenCV's
    HSV scale (red end of the spectrum appears at both 0 and 180).
    """
    hsv    = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    mask1  = cv2.inRange(hsv, SKIN_LOWER_1, SKIN_UPPER_1)
    mask2  = cv2.inRange(hsv, SKIN_LOWER_2, SKIN_UPPER_2)
    mask   = cv2.bitwise_or(mask1, mask2)

    # Clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask   = cv2.GaussianBlur(mask, (5, 5), 0)
    return mask


# ============================================================
# CV TECHNIQUE 2: CANNY EDGE DETECTION
# ============================================================

def get_edges(roi_bgr: np.ndarray) -> np.ndarray:
    """
    CV TECHNIQUE 2: Canny Edge Detection
    --------------------------------------
    Applies skin masking first so only the hand is visible, then runs
    Canny to extract the hand's edge silhouette.

    Pipeline:
      1. Skin colour mask  — zero out all non-skin pixels
      2. Grayscale + blur  — reduce noise
      3. Canny             — find strong edge contours
      4. Resize            — normalise to ROI_SIZE x ROI_SIZE
    """
    skin_mask = get_skin_mask(roi_bgr)

    # Zero out background pixels using the skin mask
    masked = cv2.bitwise_and(roi_bgr, roi_bgr, mask=skin_mask)

    gray    = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges   = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)
    resized = cv2.resize(edges, (ROI_SIZE, ROI_SIZE), interpolation=cv2.INTER_AREA)
    return resized, skin_mask


# ============================================================
# CV TECHNIQUE 3: TEMPLATE MATCHING
# ============================================================

def match_gesture(live_edges: np.ndarray, templates: dict) -> tuple:
    """
    CV TECHNIQUE 3: Template Matching
    -----------------------------------
    Compares the live edge image against every saved sample for each
    gesture using normalised cross-correlation (TM_CCOEFF_NORMED).

    For each gesture class, the BEST score across all its samples is
    used — this makes detection robust to slight variation in hand angle.

    Returns (label, score) or (None, score) if below MIN_CONFIDENCE.
    """
    if not templates:
        return None, 0.0

    best_label = None
    best_score = -1.0

    for label, sample_list in templates.items():
        # Score every saved sample, keep the highest
        class_best = max(
            float(cv2.matchTemplate(
                live_edges.astype(np.float32),
                tmpl.astype(np.float32),
                cv2.TM_CCOEFF_NORMED
            )[0, 0])
            for tmpl in sample_list
        )
        if class_best > best_score:
            best_score = class_best
            best_label = label

    if best_score < MIN_CONFIDENCE:
        return None, best_score

    return best_label, best_score


# ============================================================
# TEMPLATE SAVE / LOAD  (supports multiple samples per gesture)
# ============================================================

def save_template(key: str, edges: np.ndarray, index: int):
    os.makedirs(TEMPLATE_DIR, exist_ok=True)
    path = os.path.join(TEMPLATE_DIR, f"{key}_{index}.png")
    cv2.imwrite(path, edges)
    print(f"  [Template] Saved sample {index+1} for '{key}' -> {path}")


def load_templates() -> dict:
    """Load all saved samples. Returns {label: [edge_img, ...]}."""
    templates = {}
    if not os.path.exists(TEMPLATE_DIR):
        return templates
    for _, (key, label) in GESTURES.items():
        files = sorted(glob.glob(os.path.join(TEMPLATE_DIR, f"{key}_*.png")))
        samples = []
        for f in files:
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                samples.append(cv2.resize(img, (ROI_SIZE, ROI_SIZE)))
        if samples:
            templates[label] = samples
            print(f"  [Template] Loaded '{label}': {len(samples)} sample(s)")
    return templates


def delete_templates():
    if not os.path.exists(TEMPLATE_DIR):
        return
    for f in glob.glob(os.path.join(TEMPLATE_DIR, "*.png")):
        os.remove(f)
    print("  [Template] All templates deleted")


def template_sample_count(templates: dict, label: str) -> int:
    return len(templates.get(label, []))


# ============================================================
# DISPLAY HELPERS
# ============================================================

def draw_roi_box(frame, gesture):
    color = (0, 255, 100) if gesture else (0, 255, 0)
    cv2.rectangle(frame, (ROI_X, ROI_Y),
                  (ROI_X + ROI_W, ROI_Y + ROI_H), color, 2)
    cv2.putText(frame, "Place hand here",
                (ROI_X, ROI_Y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def draw_status_panel(frame, templates, gesture, score, show_edges):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (420, 175), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    for i, (_, (key, label)) in enumerate(GESTURES.items()):
        count = template_sample_count(templates, label)
        if count >= SAMPLES_PER_GESTURE:
            color  = (0, 220, 0)
            marker = f"OK ({count})"
        elif count > 0:
            color  = (0, 180, 255)
            marker = f"{count}/{SAMPLES_PER_GESTURE} — press [{i+1}] again"
        else:
            color  = (80, 80, 80)
            marker = f"-- press [{i+1}] to capture"
        cv2.putText(frame, f"{label}: {marker}",
                    (10, 26 + i * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    if gesture:
        text  = f"Watching...  best={score:.2f}"
        color = (120, 120, 120)
    elif not templates:
        text  = "Capture each gesture first"
        color = (0, 180, 255)
    elif sum(len(v) for v in templates.values()) < len(GESTURES):
        text  = "Need at least 1 sample per gesture"
        color = (0, 180, 255)
    else:
        text  = f"Watching...  best={score:.2f}"
        color = (120, 120, 120)

    cv2.putText(frame, text, (10, 108),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

    hint = "[d] hide debug" if show_edges else "[d] show debug"
    cv2.putText(frame, f"{hint}   [r] reset   [q] quit",
                (10, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1)

    cv2.putText(frame, f"Confidence threshold: {MIN_CONFIDENCE}",
                (10, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 100, 100), 1)


def draw_edge_overlay(frame, edges, skin_mask):
    """Show skin mask in blue and edges in yellow over the ROI."""
    overlay = np.zeros((ROI_H, ROI_W, 3), dtype=np.uint8)

    # Skin mask in blue
    skin_small = cv2.resize(skin_mask, (ROI_W, ROI_H))
    overlay[skin_small > 0] = (180, 80, 0)

    # Edges in bright yellow
    edge_small = cv2.resize(edges, (ROI_W, ROI_H))
    overlay[edge_small > 0] = (0, 230, 255)

    roi_area = frame[ROI_Y:ROI_Y + ROI_H, ROI_X:ROI_X + ROI_W]
    cv2.addWeighted(overlay, 0.5, roi_area, 0.5, 0, roi_area)


def draw_popup(frame, gesture):
    """Centred popup showing detected gesture name."""
    popup_w, popup_h = 420, 90
    popup_x = (frame.shape[1] - popup_w) // 2
    popup_y = (frame.shape[0] - popup_h) // 2

    overlay = frame.copy()
    cv2.rectangle(overlay, (popup_x, popup_y),
                  (popup_x + popup_w, popup_y + popup_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.rectangle(frame, (popup_x, popup_y),
                  (popup_x + popup_w, popup_y + popup_h), (0, 255, 100), 3)
    cv2.putText(frame, "Detected:",
                (popup_x + 20, popup_y + 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (180, 180, 180), 2)
    cv2.putText(frame, gesture,
                (popup_x + 20, popup_y + 72),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 100), 3)


def build_debug_window(live_edges, skin_mask, templates, gesture):
    s     = ROI_SIZE
    cells = []

    # Live edges
    live_bgr = cv2.cvtColor(live_edges, cv2.COLOR_GRAY2BGR)
    live_bgr[live_edges > 0] = (0, 230, 255)
    cv2.putText(live_bgr, "LIVE edges", (4, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
    cells.append(live_bgr)

    # Skin mask
    mask_small  = cv2.resize(skin_mask, (s, s))
    mask_bgr    = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
    mask_bgr[mask_small > 0] = (180, 80, 0)
    cv2.putText(mask_bgr, "Skin mask", (4, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
    cells.append(mask_bgr)

    # One cell per gesture (show first sample)
    for _, (key, label) in GESTURES.items():
        samples = templates.get(label, [])
        if samples:
            cell = cv2.cvtColor(samples[0], cv2.COLOR_GRAY2BGR)
            cell[samples[0] > 0] = (0, 230, 255)
            border = (0, 255, 100) if label == gesture else (50, 50, 50)
            cv2.rectangle(cell, (0, 0), (s-1, s-1), border, 3)
            n = len(samples)
            cv2.putText(cell, f"{label.split()[0]} ({n})", (4, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
        else:
            cell = np.zeros((s, s, 3), dtype=np.uint8)
            cv2.putText(cell, f"{label} --", (4, s // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (60, 60, 60), 1)
        cells.append(cell)

    div   = np.full((s, 3, 3), 30, dtype=np.uint8)
    panel = cells[0]
    for c in cells[1:]:
        panel = np.hstack([panel, div, c])

    return cv2.resize(panel, (panel.shape[1] * 2, panel.shape[0] * 2),
                      interpolation=cv2.INTER_NEAREST)


# ============================================================
# MAIN
# ============================================================

def main():
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        alt = 0 if WEBCAM_INDEX == 1 else 1
        print(f"Camera {WEBCAM_INDEX} failed, trying {alt}...")
        cap = cv2.VideoCapture(alt)
        if not cap.isOpened():
            print("Could not open any webcam.")
            return

    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    cap.set(cv2.CAP_PROP_ZOOM, 0)
    cap.set(cv2.CAP_PROP_FOCUS, 50)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    templates  = load_templates()
    show_edges = True
    gesture    = None
    score      = 0.0

    # Track how many samples have been captured per gesture this session
    session_counts = {label: template_sample_count(templates, label)
                      for _, (_, label) in GESTURES.items()}

    print("\n" + "="*54)
    print("  GESTURE DETECTION  —  Skin Mask + Canny + Matching")
    print("="*54)
    print("  Hold each gesture in the box and press its key")
    print(f"  Press each key {SAMPLES_PER_GESTURE}x to capture multiple samples")
    print("  [1] Open Palm   [2] Fist   [3] Thumbs Up")
    print("  [d] Toggle debug view      [r] Reset      [q] Quit")
    print("="*54 + "\n")
    if templates:
        print(f"  Loaded templates: { {l: len(s) for l, s in templates.items()} }\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Webcam read failed")
            break

        # frame = cv2.flip(frame, 1)  # uncomment to mirror

        frame_roi  = frame[ROI_Y:ROI_Y + ROI_H, ROI_X:ROI_X + ROI_W].copy()

        # ── HSV readout — shows exact skin values under crosshair ────────
        cx, cy = ROI_W // 2, ROI_H // 2
        hsv_roi = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2HSV)
        h, s, v = hsv_roi[cy, cx]
        cv2.drawMarker(frame, (ROI_X + cx, ROI_Y + cy),
                       (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
        cv2.putText(frame, f"HSV at cross: H={h} S={s} V={v}",
                    (ROI_X, ROI_Y + ROI_H + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

        # ── CV TECHNIQUE 1 + 2: Skin mask then Canny ──────────────────────
        live_edges, skin_mask = get_edges(frame_roi)

        # ── CV TECHNIQUE 3: Template matching ─────────────────────────────
        gesture, score = match_gesture(live_edges, templates)

        # ── Draw ──────────────────────────────────────────────────────────
        if show_edges:
            draw_edge_overlay(frame, live_edges, skin_mask)

        draw_roi_box(frame, gesture)
        draw_status_panel(frame, templates, gesture, score, show_edges)

        if gesture:
            draw_popup(frame, gesture)

        cv2.imshow("Gesture Detection", frame)

        if show_edges:
            cv2.imshow("Debug View", build_debug_window(
                live_edges, skin_mask, templates, gesture))
        else:
            try:
                cv2.destroyWindow("Debug View")
            except:
                pass

        # ── Keys ──────────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key in GESTURES:
            gesture_key, label = GESTURES[key]
            idx = session_counts.get(label, 0)
            templates.setdefault(label, []).append(live_edges.copy())
            save_template(gesture_key, live_edges, idx)
            session_counts[label] = idx + 1
            total = len(templates[label])
            print(f"  Captured sample {total} for '{label}'", end="")
            if total >= SAMPLES_PER_GESTURE:
                print(f"  — {label} ready!")
            else:
                print(f"  — press [{list(GESTURES.keys()).index(key)+1}] "
                      f"{SAMPLES_PER_GESTURE - total} more time(s)")

        elif key == ord('d'):
            show_edges = not show_edges

        elif key == ord('r'):
            templates.clear()
            session_counts = {l: 0 for _, (_, l) in GESTURES.items()}
            delete_templates()

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
