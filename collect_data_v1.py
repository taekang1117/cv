# collect_data.py
# Run: python3 collect_data.py
import sys
import time
import cv2
import numpy as np
import pandas as pd

# =========================
# Configuration
# =========================
FRAME_W, FRAME_H = 960, 540
ROI_X, ROI_Y, ROI_W, ROI_H = 272, 44, 418, 417

BLUR_K = 5
MORPH_K = 5
OPEN_ITERS = 2
CLOSE_ITERS = 2
MIN_AREA = 800
MAX_AREA = 40000

DATA_FILE = "training_data.csv"

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
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=OPEN_ITERS)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=CLOSE_ITERS)
    return mask

def capture_background_gray(cap, roi_rect, n=20):
    rx, ry, rw, rh = roi_rect
    acc = None
    for _ in range(n):
        ret, frame_bgr = cap.read()
        if not ret:
            continue
        frame_bgr = cv2.resize(frame_bgr, (FRAME_W, FRAME_H))
        roi = frame_bgr[ry:ry+rh, rx:rx+rw]
        g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
        g = cv2.GaussianBlur(g, (BLUR_K, BLUR_K), 0)
        acc = g if acc is None else acc + g
    return (acc / n).astype(np.uint8)

def get_object_mask(roi_bgr, bg_gray):
    g1 = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    g1 = cv2.GaussianBlur(g1, (BLUR_K, BLUR_K), 0)
    diff = cv2.absdiff(g1, bg_gray)
    _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = morph_cleanup(mask)
    return mask

def get_features(cnt):
    area = float(cv2.contourArea(cnt))
    perim = float(cv2.arcLength(cnt, True))
    if perim == 0:
        return None

    circularity = (4.0 * np.pi * area) / (perim * perim)
    hull = cv2.convexHull(cnt)
    hull_area = float(cv2.contourArea(hull))
    solidity = area / hull_area if hull_area > 0 else 0
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio_invariant = float(max(w, h)) / (min(w, h) + 1e-9)

    return {
        "area": area,
        "aspect_ratio": aspect_ratio_invariant,
        "circularity": circularity,
        "solidity": solidity,
        "perimeter": perim
    }

# =========================
# Main
# =========================
def main():
    roi_rect = clamp_roi(ROI_X, ROI_Y, ROI_W, ROI_H, FRAME_W, FRAME_H)
    rx, ry, rw, rh = roi_rect

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    bg_gray = None
    samples_collected = []

    print("=" * 60)
    print("DATA COLLECTION MODE")
    print("1. Clear plate, press 'b' to capture BACKGROUND.")
    print("2. Place BEANS, press '1' to collect BEAN samples.")
    print("3. Place ROCKS, press '2' to collect ROCK samples.")
    print("4. Press 's' to SAVE to CSV.")
    print("5. Press 'q' to QUIT.")
    print("=" * 60)

    try:
        while True:
            ret, full_bgr = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            full_bgr = cv2.resize(full_bgr, (FRAME_W, FRAME_H))
            cv2.rectangle(full_bgr, (rx, ry), (rx + rw, ry + rh), (0, 255, 255), 2)
            roi_bgr = full_bgr[ry:ry + rh, rx:rx + rw]
            vis_roi = roi_bgr.copy()
            contours = []

            if bg_gray is None:
                cv2.putText(full_bgr, "Press 'b' for BACKGROUND", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                mask = get_object_mask(roi_bgr, bg_gray)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                count_visible = 0
                for cnt in contours:
                    a = cv2.contourArea(cnt)
                    if MIN_AREA < a < MAX_AREA:
                        count_visible += 1
                        x, y, w, h = cv2.boundingRect(cnt)
                        cv2.rectangle(vis_roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

                cv2.putText(full_bgr, f"Visible Objects: {count_visible}", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(full_bgr, f"Collected Total: {len(samples_collected)}", (20, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                cv2.imshow("Mask", mask)

            cv2.imshow("Data Collector", full_bgr)
            cv2.imshow("ROI", vis_roi)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            if key == ord('b'):
                print("Capturing background...")
                bg_gray = capture_background_gray(cap, roi_rect)
                print("Background captured.")

            if key == ord('1'):  # BEAN
                if bg_gray is not None:
                    added = 0
                    for cnt in contours:
                        if MIN_AREA < cv2.contourArea(cnt) < MAX_AREA:
                            feats = get_features(cnt)
                            if feats:
                                feats["label"] = 1
                                samples_collected.append(feats)
                                added += 1
                    print(f"Added {added} BEAN samples.")
                else:
                    print("!! Capture background first (b) !!")

            if key == ord('2'):  # ROCK
                if bg_gray is not None:
                    added = 0
                    for cnt in contours:
                        if MIN_AREA < cv2.contourArea(cnt) < MAX_AREA:
                            feats = get_features(cnt)
                            if feats:
                                feats["label"] = 0
                                samples_collected.append(feats)
                                added += 1
                    print(f"Added {added} ROCK samples.")
                else:
                    print("!! Capture background first (b) !!")

            if key == ord('s'):
                if len(samples_collected) > 0:
                    df = pd.DataFrame(samples_collected)
                    df.to_csv(DATA_FILE, index=False)
                    print(f"Saved {len(df)} samples to {DATA_FILE}")
                else:
                    print("No data to save!")

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()