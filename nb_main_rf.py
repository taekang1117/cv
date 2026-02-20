# main_rf.py — Bolt vs Screw inference with counting (webcam version)
# Run: python3 main_rf.py
import cv2
import numpy as np
import pickle
import os
import sys

# =========================
# Configuration
# =========================
WEBCAM_INDEX = 0        # change to 1 if wrong camera opens
FRAME_W, FRAME_H        = 960, 540
ROI_X, ROI_Y, ROI_W, ROI_H = 260, 90, 440, 360

BLUR_K      = 5
MORPH_K     = 5
OPEN_ITERS  = 2
CLOSE_ITERS = 2
MIN_AREA    = 800
MAX_AREA    = 40000

MODEL_FILE  = "bolt_screw_model.pkl"

BOLT_COLOR  = (0, 200, 255)   # yellow-gold for bolts
SCREW_COLOR = (80, 80, 80)    # dark grey for screws

# =========================
# Helpers
# =========================
def load_model():
    if not os.path.exists(MODEL_FILE):
        print(f"ERROR: {MODEL_FILE} not found!")
        print("Run collect_data.py then train_model.py first.")
        return None
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)
    print(f"Loaded model from {MODEL_FILE}")
    return model

def clamp_roi(x, y, w, h, W, H):
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))
    return x, y, w, h

def morph_cleanup(mask):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_K, MORPH_K))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=OPEN_ITERS)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=CLOSE_ITERS)
    return mask

def capture_background_gray(cap, roi_rect, n=20):
    rx, ry, rw, rh = roi_rect
    acc = None
    for _ in range(n):
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.resize(frame, (FRAME_W, FRAME_H))
        roi   = frame[ry:ry+rh, rx:rx+rw]
        g     = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
        g     = cv2.GaussianBlur(g, (BLUR_K, BLUR_K), 0)
        acc   = g if acc is None else acc + g
    return (acc / n).astype(np.uint8)

def get_object_mask(roi_bgr, bg_gray):
    g1 = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    g1 = cv2.GaussianBlur(g1, (BLUR_K, BLUR_K), 0)
    diff = cv2.absdiff(g1, bg_gray)
    _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = morph_cleanup(mask)
    return mask

def get_features_vector(cnt):
    # Order MUST match train_model.py:
    # ['area', 'aspect_ratio', 'circularity', 'solidity', 'perimeter']
    area  = float(cv2.contourArea(cnt))
    perim = float(cv2.arcLength(cnt, True))
    if perim == 0:
        return None
    circularity = (4.0 * np.pi * area) / (perim * perim)
    hull        = cv2.convexHull(cnt)
    hull_area   = float(cv2.contourArea(hull))
    solidity    = area / hull_area if hull_area > 0 else 0
    x, y, w, h  = cv2.boundingRect(cnt)
    aspect_ratio_invariant = float(max(w, h)) / (min(w, h) + 1e-9)
    return [area, aspect_ratio_invariant, circularity, solidity, perim]

# =========================
# Main
# =========================
def main():
    model = load_model()
    if model is None:
        sys.exit(1)

    roi_rect       = clamp_roi(ROI_X, ROI_Y, ROI_W, ROI_H, FRAME_W, FRAME_H)
    rx, ry, rw, rh = roi_rect

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    if not cap.isOpened():
        print("Could not open webcam. Check WEBCAM_INDEX.")
        sys.exit(1)

    bg_gray = None

    print("=" * 50)
    print("BOLT vs SCREW COUNTER  —  Inference Mode")
    print("  b : Capture background")
    print("  q : Quit")
    print("=" * 50)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Webcam read failed")
            break

        full_bgr = cv2.resize(frame, (FRAME_W, FRAME_H))
        cv2.rectangle(full_bgr, (rx, ry), (rx + rw, ry + rh), (0, 255, 255), 2)
        roi_bgr = full_bgr[ry:ry + rh, rx:rx + rw]
        vis_roi = roi_bgr.copy()

        if bg_gray is None:
            cv2.putText(full_bgr, "Press 'b' for BACKGROUND", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            mask        = get_object_mask(roi_bgr, bg_gray)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            bolts_count  = 0
            screws_count = 0
            feature_list   = []
            valid_contours = []
            bbox_list      = []

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if not (MIN_AREA < area < MAX_AREA):
                    continue
                vec = get_features_vector(cnt)
                if vec:
                    feature_list.append(vec)
                    valid_contours.append(cnt)
                    bbox_list.append(cv2.boundingRect(cnt))

            if feature_list:
                preds = model.predict(feature_list)
                probs = model.predict_proba(feature_list)

                for i, label in enumerate(preds):
                    x, y, w, h = bbox_list[i]
                    conf        = max(probs[i]) * 100

                    if label == 1:   # BOLT
                        color = BOLT_COLOR
                        text  = f"BOLT {conf:.0f}%"
                        bolts_count += 1
                    else:            # SCREW
                        color = SCREW_COLOR
                        text  = f"SCREW {conf:.0f}%"
                        screws_count += 1

                    cv2.rectangle(vis_roi, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(vis_roi, text, (x, y - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            cv2.putText(full_bgr,
                        f"Bolts: {bolts_count}   Screws: {screws_count}   "
                        f"Total: {bolts_count + screws_count}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.imshow("Mask", mask)

        cv2.imshow("Overview", full_bgr)
        cv2.imshow("Inference — Bolt vs Screw", vis_roi)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('b'):
            print("Capturing background...")
            bg_gray = capture_background_gray(cap, roi_rect)
            print("Background captured.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
