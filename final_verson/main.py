# main.py — Bolt vs NUT inference with counting (webcam version)
# Run: python3 main.py
import cv2
import numpy as np
import pickle
import os
import sys

# =========================
# Configuration
# =========================
WEBCAM_INDEX = 0
FRAME_W, FRAME_H            = 960, 540
ROI_X, ROI_Y, ROI_W, ROI_H = 260, 90, 440, 360

BLUR_K      = 5
MORPH_K     = 5
OPEN_ITERS  = 2
CLOSE_ITERS = 2
MIN_AREA    = 800
MAX_AREA    = 40000

RF_MODEL_FILE  = "bolt_nut_model.pkl"
KNN_MODEL_FILE = "bolt_nut_knn.pkl"
SVM_MODEL_FILE = "bolt_nut_svm.pkl"

BOLT_COLOR  = (0, 200, 255)   # yellow-gold
NUT_COLOR = (80, 80, 80)    # dark grey

CLASSIFIER_NAMES = ["Random Forest", "KNN (k=5)", "SVM (RBF)"]

# =========================
# Model Loading
# =========================
def load_model(filepath):
    if not os.path.exists(filepath):
        print(f"ERROR: {filepath} not found!")
        print("Run collect_data.py then train_model.py first.")
        return None, None
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    # RF is saved as just the model; KNN and SVM are saved as (model, scaler) tuples
    if isinstance(data, tuple):
        model, scaler = data
    else:
        model, scaler = data, None
    print(f"Loaded: {filepath}")
    return model, scaler

# =========================
# Helpers
# =========================
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

def run_classifier(model, scaler, feature_list):
    import pandas as pd
    col_names = ['area', 'aspect_ratio', 'circularity', 'solidity', 'perimeter']
    X = pd.DataFrame(feature_list, columns=col_names)
    if scaler is not None:
        X = scaler.transform(X)
    preds = model.predict(X)
    probs = model.predict_proba(X)
    return preds, probs

def draw_results(vis_roi, bbox_list, preds, probs):
    """Draw bounding boxes and labels on the ROI display copy."""
    bolts_count  = 0
    nuts_count = 0
    for i, label in enumerate(preds):
        x, y, w, h = bbox_list[i]
        conf = max(probs[i]) * 100
        if label == 1:
            color = BOLT_COLOR
            text  = f"BOLT {conf:.0f}%"
            bolts_count += 1
        else:
            color = NUT_COLOR
            text  = f"NUT {conf:.0f}%"
            nuts_count += 1
        cv2.rectangle(vis_roi, (x, y), (x + w, y + h), color, 2)
        cv2.putText(vis_roi, text, (x, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return bolts_count, nuts_count

# =========================
# Main
# =========================
def main():
    # Load all three models
    rf_model,  rf_scaler  = load_model(RF_MODEL_FILE)
    knn_model, knn_scaler = load_model(KNN_MODEL_FILE)
    svm_model, svm_scaler = load_model(SVM_MODEL_FILE)

    models  = [rf_model,  knn_model,  svm_model]
    scalers = [rf_scaler, knn_scaler, svm_scaler]

    if any(m is None for m in models):
        print("One or more models failed to load. Run train_model.py first.")
        sys.exit(1)

    roi_rect       = clamp_roi(ROI_X, ROI_Y, ROI_W, ROI_H, FRAME_W, FRAME_H)
    rx, ry, rw, rh = roi_rect

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    if not cap.isOpened():
        print("Could not open webcam. Check WEBCAM_INDEX.")
        sys.exit(1)

    bg_gray       = None
    active_clf    = 0   # 0 = RF, 1 = KNN, 2 = SVM

    print("=" * 50)
    print("BOLT vs NUT COUNTER  -  Inference Mode")
    print("  b         : Capture background")
    print("  1 / 2 / 3 : Switch classifier (RF / KNN / SVM)")
    print("  q         : Quit")
    print("=" * 50)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Webcam read failed")
            break

        full_bgr = cv2.resize(frame, (FRAME_W, FRAME_H))
        cv2.rectangle(full_bgr, (rx, ry), (rx + rw, ry + rh), (0, 255, 255), 2)
        roi_bgr  = full_bgr[ry:ry + rh, rx:rx + rw]
        vis_roi  = roi_bgr.copy()

        if bg_gray is None:
            cv2.putText(full_bgr, "Press 'b' for BACKGROUND", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            mask        = get_object_mask(roi_bgr, bg_gray)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

            feature_list = []
            bbox_list    = []

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if not (MIN_AREA < area < MAX_AREA):
                    continue
                vec = get_features_vector(cnt)
                if vec:
                    feature_list.append(vec)
                    bbox_list.append(cv2.boundingRect(cnt))

            bolts_count  = 0
            nuts_count = 0

            if feature_list:
                # Run only the active classifier for display
                preds, probs = run_classifier(
                    models[active_clf],
                    scalers[active_clf],
                    feature_list
                )
                bolts_count, nuts_count = draw_results(
                    vis_roi, bbox_list, preds, probs
                )

            # Show active classifier name, bolt/nut counts at top of frame
            clf_name = CLASSIFIER_NAMES[active_clf]
            cv2.putText(full_bgr,
                        f"[{clf_name}]  Bolts: {bolts_count}  "
                        f"Nuts: {nuts_count}  "
                        f"Total: {bolts_count + nuts_count}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Small key reminder at bottom of frame
            cv2.putText(full_bgr, "1=RF  2=KNN  3=SVM  b=Background  q=Quit",
                        (20, FRAME_H - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("Mask", mask)

        cv2.imshow("Overview", full_bgr)
        cv2.imshow("Inference - Bolt vs Nut", vis_roi)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('b'):
            print("Capturing background...")
            bg_gray = capture_background_gray(cap, roi_rect)
            print("Background captured.")
        if key == ord('1'):
            active_clf = 0
            print("Switched to: Random Forest")
        if key == ord('2'):
            active_clf = 1
            print("Switched to: KNN")
        if key == ord('3'):
            active_clf = 2
            print("Switched to: SVM")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
