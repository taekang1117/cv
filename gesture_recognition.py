"""
gesture_recognition.py
-----------------------
Drop-in gesture recognition module for your Kinect v1 project.

Techniques used:
  - Canny edge detection  → extract hand silhouettes from depth images
  - cv2.matchTemplate     → compare live hand ROI against dataset templates
  - Skeleton joint data   → crop a tight ROI around the hand so matching is fast & accurate

HOW TO INTEGRATE
----------------
1.  Point DATASET_ROOT at your depth image dataset folder.
    Expected structure (one sub-folder per gesture label):

        dataset/
          open_palm/
            img001.png
            img002.png
            ...
          fist/
            img001.png
            ...
          pointing/
            ...

2.  At the top of your main script add:
        from gesture_recognition import GestureRecognizer
        recognizer = GestureRecognizer(DATASET_ROOT)

3.  Inside depth_frame_ready(), after you have depth_mm, call:
        label, score = recognizer.classify(depth_mm, joint_positions)
        if label:
            current_gesture = f"{label}  ({score:.2f})"

    That's it — the rest of your code stays the same.
"""

import os
import cv2
import numpy as np
from pathlib import Path


# ── Tunable parameters ────────────────────────────────────────────────────────

# Path to your dataset root folder
DATASET_ROOT = r"C:\path\to\your\dataset"

# Size (pixels) of the square ROI cropped around the hand joint
ROI_SIZE = 96

# Canny thresholds — lower = more edges detected, higher = fewer/cleaner edges
CANNY_LOW  = 30
CANNY_HIGH = 90

# How many templates to load per gesture class (None = load all)
MAX_TEMPLATES_PER_CLASS = 20

# Minimum match score [0-1] to report a gesture (below this → "Unknown")
MIN_CONFIDENCE = 0.45

# ── End of tunable parameters ─────────────────────────────────────────────────


class GestureRecognizer:
    """
    Loads a depth-image dataset, extracts Canny-edge templates,
    and classifies live hand crops using template matching.
    """

    def __init__(self, dataset_root: str = DATASET_ROOT):
        self.roi_size  = ROI_SIZE
        self.templates = {}          # { label: [edge_img, ...] }
        self._load_templates(dataset_root)

    # ── Dataset loading ───────────────────────────────────────────────────────

    def _load_templates(self, root: str):
        """
        Walk dataset_root, treat every sub-folder name as a gesture label,
        load depth images, apply Canny, and store as templates.
        """
        root = Path(root)
        if not root.exists():
            print(f"[GestureRecognizer] ⚠  Dataset path not found: {root}")
            print("                       Set DATASET_ROOT to your dataset folder.")
            return

        extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
        loaded_total = 0

        for label_dir in sorted(root.iterdir()):
            if not label_dir.is_dir():
                continue

            label = label_dir.name
            templates = []
            image_files = [
                f for f in sorted(label_dir.iterdir())
                if f.suffix.lower() in extensions
            ]

            if MAX_TEMPLATES_PER_CLASS:
                image_files = image_files[:MAX_TEMPLATES_PER_CLASS]

            for img_path in image_files:
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue

                edge_template = self._to_edge_template(img)
                if edge_template is not None:
                    templates.append(edge_template)

            if templates:
                self.templates[label] = templates
                loaded_total += len(templates)
                print(f"[GestureRecognizer] ✓  '{label}': {len(templates)} templates loaded")

        print(f"[GestureRecognizer] ✓  Total: {loaded_total} templates across {len(self.templates)} classes\n")

    def _to_edge_template(self, gray_img: np.ndarray) -> np.ndarray | None:
        """
        Resize → smooth → Canny → return a fixed-size edge image.
        Works for both 8-bit and 16-bit depth images.
        """
        # Normalise to 8-bit if needed (16-bit depth PNGs from the Kinect)
        if gray_img.dtype == np.uint16:
            # Stretch to full 8-bit range so Canny sees contrast
            norm = cv2.normalize(gray_img, None, 0, 255, cv2.NORM_MINMAX)
            gray_img = norm.astype(np.uint8)

        if gray_img.size == 0:
            return None

        resized = cv2.resize(gray_img, (self.roi_size, self.roi_size),
                             interpolation=cv2.INTER_AREA)
        blurred = cv2.GaussianBlur(resized, (5, 5), 0)
        edges   = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)
        return edges

    # ── Live classification ───────────────────────────────────────────────────

    def classify(
        self,
        depth_mm: np.ndarray,
        joint_positions: dict,
        hand_key=None           # pass a Microsoft.Kinect.JointType if you like
    ) -> tuple[str | None, float]:
        """
        Crop the hand region from the live depth frame, apply edge detection,
        then run template matching against every loaded gesture class.

        Parameters
        ----------
        depth_mm        : float32 array (H×W) of per-pixel depth in mm
        joint_positions : dict produced by skeleton_frame_ready()
        hand_key        : the JointType key for the hand to inspect
                          (auto-selects HandRight if None)

        Returns
        -------
        (label, score)  label is None when no hand is visible or confidence too low
        """
        if not self.templates:
            return None, 0.0

        roi = self._extract_hand_roi(depth_mm, joint_positions, hand_key)
        if roi is None:
            return None, 0.0

        live_edges = self._to_edge_template(roi)
        if live_edges is None:
            return None, 0.0

        return self._match_against_templates(live_edges)

    def _extract_hand_roi(
        self,
        depth_mm: np.ndarray,
        joint_positions: dict,
        hand_key
    ) -> np.ndarray | None:
        """
        Use the skeleton hand-joint pixel position to crop a square ROI
        from the depth image centred on the hand.
        """
        # Auto-select hand joint key
        if hand_key is None:
            # Try to find HandRight or HandLeft by string matching the key repr
            for k in joint_positions:
                if "HandRight" in str(k):
                    hand_key = k
                    break
            if hand_key is None:
                for k in joint_positions:
                    if "HandLeft" in str(k):
                        hand_key = k
                        break

        if hand_key is None or hand_key not in joint_positions:
            return None

        hand = joint_positions[hand_key]
        cx, cy = hand["x"], hand["y"]
        half   = self.roi_size // 2

        H, W = depth_mm.shape
        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(W, cx + half)
        y2 = min(H, cy + half)

        if (x2 - x1) < 10 or (y2 - y1) < 10:
            return None

        crop = depth_mm[y1:y2, x1:x2].copy()

        # Normalise the cropped float32 depth to uint8 for edge detection
        valid = crop[crop > 0]
        if valid.size == 0:
            return None

        lo, hi = valid.min(), valid.max()
        if hi - lo < 1:
            return None

        norm = np.zeros_like(crop, dtype=np.uint8)
        mask = crop > 0
        norm[mask] = ((crop[mask] - lo) / (hi - lo) * 255).astype(np.uint8)
        return norm

    def _match_against_templates(
        self, live_edges: np.ndarray
    ) -> tuple[str | None, float]:
        """
        Compare live_edges against every template using TM_CCOEFF_NORMED.
        Returns the best-matching label and its normalised score [0-1].

        Because live_edges and every template are the same size (ROI_SIZE²),
        matchTemplate with TM_CCOEFF_NORMED returns a 1×1 result — i.e. a
        single similarity score — which is exactly what we want.
        """
        best_label = None
        best_score = -1.0

        for label, tmpl_list in self.templates.items():
            scores = []
            for tmpl in tmpl_list:
                # Both images are ROI_SIZE × ROI_SIZE — result is a 1×1 matrix
                result = cv2.matchTemplate(
                    live_edges.astype(np.float32),
                    tmpl.astype(np.float32),
                    cv2.TM_CCOEFF_NORMED
                )
                scores.append(float(result[0, 0]))

            # Use the mean of the top-5 scores for robustness
            top_scores = sorted(scores, reverse=True)[:5]
            class_score = float(np.mean(top_scores))

            if class_score > best_score:
                best_score = class_score
                best_label = label

        if best_score < MIN_CONFIDENCE:
            return None, best_score

        return best_label, best_score

    # ── Debug visualisation ───────────────────────────────────────────────────

    def debug_view(
        self,
        depth_mm: np.ndarray,
        joint_positions: dict,
        hand_key=None
    ) -> np.ndarray:
        """
        Returns a small debug image showing:
          left  — normalised hand ROI
          right — Canny edges of that ROI
        Useful for verifying that cropping and edge detection look correct.
        Call this inside depth_frame_ready() and display with cv2.imshow().
        """
        blank = np.zeros((self.roi_size, self.roi_size * 2 + 4, 3), dtype=np.uint8)

        roi = self._extract_hand_roi(depth_mm, joint_positions, hand_key)
        if roi is None:
            cv2.putText(blank, "No hand", (4, self.roi_size // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
            return blank

        edges = self._to_edge_template(roi)

        left  = cv2.cvtColor(roi,   cv2.COLOR_GRAY2BGR)
        right = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        divider = np.full((self.roi_size, 4, 3), 80, dtype=np.uint8)
        return np.hstack([left, divider, right])


# ── Quick standalone test (run this file directly to verify loading) ──────────

if __name__ == "__main__":
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else DATASET_ROOT
    print(f"Loading templates from: {root}\n")
    rec = GestureRecognizer(root)

    if not rec.templates:
        print("No templates loaded — check DATASET_ROOT path and folder structure.")
    else:
        print("Template loading OK.")
        print("Classes found:", list(rec.templates.keys()))
