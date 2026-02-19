"""
gesture_recognition.py  (MSR Gesture 3D edition)
-------------------------------------------------
Loads the MSR Gesture 3D dataset (.mat files), extracts per-class edge
templates, and classifies live Kinect depth frames using template matching.

Dataset format
--------------
Files: sub_depth_m_n.mat   (m = subject index, n = 1..36)
Label mapping (n -> gesture):
  1-3  -> ASL_Z       4-6  -> ASL_J       7-9  -> ASL_Where
  10-12-> ASL_Store   13-15-> ASL_Pig     16-18-> ASL_Past
  19-21-> ASL_Hungary 22-24-> ASL_Green   25-27-> ASL_Finish
  28-30-> ASL_Blue    31-33-> ASL_Bathroom 34-36-> ASL_Milk

Each .mat file contains a 3-D array  depth_part  of shape (W, H, nFrames).
The hand has already been segmented and cropped by the dataset authors.

HOW TO INTEGRATE INTO YOUR MAIN SCRIPT
---------------------------------------
  from gesture_recognition import GestureRecognizer

  # After sensor.Start():
  recognizer = GestureRecognizer(r"C:\\path\\to\\MSRGesture3D")

  # Inside depth_frame_ready(), after depth_mm is computed:
  cv_label, cv_score = recognizer.classify(depth_mm, joint_positions)
  if cv_label:
      current_gesture = f"CV: {cv_label}  ({cv_score:.2f})"

  # Optional debug window:
  debug_img = recognizer.debug_view(depth_mm, joint_positions)
  cv2.imshow("Hand ROI | Edges", debug_img)
"""

import re
import cv2
import numpy as np
from pathlib import Path
from scipy.io import loadmat


# ── Tunable parameters ────────────────────────────────────────────────────────

# Path to the folder containing sub_depth_m_n.mat files
DATASET_ROOT = r"C:\path\to\MSRGesture3D"

# Size (px) of the square ROI cropped around the hand joint in the live feed
# Dataset frames are small (~50-80px), so 64 is a good match
ROI_SIZE = 64

# Canny thresholds — lower = more edges, higher = cleaner/fewer edges
# Tune these by watching the debug window
CANNY_LOW  = 20
CANNY_HIGH = 60

# Frame sampling from each .mat sequence
#   "middle" -> 1 frame from the centre (fast, fine for static ASL poses)
#   "spread" -> FRAMES_PER_SEQUENCE evenly-spaced frames (better for dynamic
#               gestures like ASL_J and ASL_Z which have motion paths)
FRAME_SAMPLE_MODE    = "spread"
FRAMES_PER_SEQUENCE  = 5

# Minimum match score [0-1] to report a gesture (below this -> "Unknown")
MIN_CONFIDENCE = 0.40

# ── Label mapping (n index in filename -> gesture name) ──────────────────────

N_TO_LABEL = {}
_MAP = [
    (range(1,  4),  "ASL_Z"),
    (range(4,  7),  "ASL_J"),
    (range(7,  10), "ASL_Where"),
    (range(10, 13), "ASL_Store"),
    (range(13, 16), "ASL_Pig"),
    (range(16, 19), "ASL_Past"),
    (range(19, 22), "ASL_Hungary"),
    (range(22, 25), "ASL_Green"),
    (range(25, 28), "ASL_Finish"),
    (range(28, 31), "ASL_Blue"),
    (range(31, 34), "ASL_Bathroom"),
    (range(34, 37), "ASL_Milk"),
]
for _rng, _lbl in _MAP:
    for _n in _rng:
        N_TO_LABEL[_n] = _lbl


class GestureRecognizer:
    """
    Loads MSR Gesture 3D .mat files, builds Canny-edge templates per class,
    and classifies a live depth ROI via template matching.
    """

    def __init__(self, dataset_root: str = DATASET_ROOT):
        self.roi_size  = ROI_SIZE
        self.templates: dict = {}   # { label: [edge_img, ...] }
        self._load_templates(dataset_root)

    # ── Dataset loading ───────────────────────────────────────────────────────

    def _load_templates(self, root: str):
        root = Path(root)
        if not root.exists():
            print(f"[GestureRecognizer]  Dataset path not found: {root}")
            return

        mat_files = list(root.glob("sub_depth_*.mat"))
        if not mat_files:
            mat_files = list(root.rglob("sub_depth_*.mat"))

        if not mat_files:
            print(f"[GestureRecognizer]  No sub_depth_*.mat files found in {root}")
            return

        counts  = {}
        skipped = 0

        for mat_path in sorted(mat_files):
            # Extract n from "sub_depth_<m>_<n>"
            match = re.search(r"sub_depth_\d+_(\d+)", mat_path.stem)
            if not match:
                skipped += 1
                continue

            n     = int(match.group(1))
            label = N_TO_LABEL.get(n)
            if label is None:
                skipped += 1
                continue

            frames = self._load_mat_frames(mat_path)
            if not frames:
                skipped += 1
                continue

            edges = [self._to_edge_template(f) for f in frames]
            edges = [e for e in edges if e is not None]

            if edges:
                self.templates.setdefault(label, []).extend(edges)
                counts[label] = counts.get(label, 0) + len(edges)

        if skipped:
            print(f"[GestureRecognizer]    {skipped} files skipped")

        total = sum(counts.values())
        for label, cnt in sorted(counts.items()):
            print(f"[GestureRecognizer] ok '{label}': {cnt} templates")
        print(f"[GestureRecognizer] Total: {total} templates, {len(self.templates)} classes\n")

    def _load_mat_frames(self, mat_path: Path):
        """
        Read depth_part from a .mat file.
        Returns a list of normalised uint8 2-D arrays (one per sampled frame).
        """
        try:
            mat = loadmat(str(mat_path))
        except Exception as e:
            print(f"[GestureRecognizer]  Could not load {mat_path.name}: {e}")
            return []

        # Locate the depth array — key is "depth_part" per the dataset docs
        data = mat.get("depth_part")
        if data is None:
            # Fallback: find first 3-D numeric array
            for key, val in mat.items():
                if not key.startswith("_") and isinstance(val, np.ndarray) and val.ndim == 3:
                    data = val
                    break

        if data is None or data.ndim != 3:
            return []

        # MATLAB stores as (W, H, nFrames) — convert to (nFrames, H, W)
        data = np.transpose(data, (2, 1, 0)).astype(np.float32)
        n_frames = data.shape[0]

        if FRAME_SAMPLE_MODE == "middle" or n_frames == 1:
            indices = [n_frames // 2]
        else:
            indices = np.linspace(0, n_frames - 1, FRAMES_PER_SEQUENCE, dtype=int).tolist()

        frames = []
        for idx in indices:
            frame = data[idx]
            valid = frame[frame > 0]
            if valid.size == 0:
                continue
            lo, hi = valid.min(), valid.max()
            if hi - lo < 1:
                continue
            norm      = np.zeros_like(frame, dtype=np.uint8)
            mask      = frame > 0
            norm[mask] = ((frame[mask] - lo) / (hi - lo) * 255).astype(np.uint8)
            frames.append(norm)

        return frames

    def _to_edge_template(self, gray: np.ndarray):
        """Resize -> GaussianBlur -> Canny -> fixed-size edge image."""
        if gray is None or gray.size == 0:
            return None
        resized = cv2.resize(gray, (self.roi_size, self.roi_size),
                             interpolation=cv2.INTER_AREA)
        blurred = cv2.GaussianBlur(resized, (5, 5), 0)
        return cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)

    # ── Live classification ───────────────────────────────────────────────────

    def classify(self, depth_mm: np.ndarray, joint_positions: dict, hand_key=None):
        """
        Extract the hand ROI from the live depth frame, run Canny edge detection,
        and match against all templates.

        Returns (label, score).  label is None if score < MIN_CONFIDENCE.
        """
        if not self.templates:
            return None, 0.0

        roi = self._extract_hand_roi(depth_mm, joint_positions, hand_key)
        if roi is None:
            return None, 0.0

        live_edges = self._to_edge_template(roi)
        if live_edges is None:
            return None, 0.0

        return self._best_match(live_edges)

    def _extract_hand_roi(self, depth_mm: np.ndarray, joint_positions: dict, hand_key):
        """
        Use the skeleton hand-joint pixel coordinate to crop a square ROI
        from the live depth image, then normalise to uint8.
        """
        # Auto-detect hand joint key from the dict
        if hand_key is None:
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
        H, W   = depth_mm.shape

        x1, y1 = max(0, cx - half), max(0, cy - half)
        x2, y2 = min(W, cx + half), min(H, cy + half)

        if (x2 - x1) < 10 or (y2 - y1) < 10:
            return None

        crop  = depth_mm[y1:y2, x1:x2].astype(np.float32)
        valid = crop[crop > 0]
        if valid.size == 0:
            return None

        lo, hi = valid.min(), valid.max()
        if hi - lo < 1:
            return None

        norm       = np.zeros_like(crop, dtype=np.uint8)
        mask       = crop > 0
        norm[mask]  = ((crop[mask] - lo) / (hi - lo) * 255).astype(np.uint8)
        return norm

    def _best_match(self, live_edges: np.ndarray):
        """
        TM_CCOEFF_NORMED against every template.
        Both images are ROI_SIZE x ROI_SIZE so the result is a 1x1 score.
        Uses the mean of the top-5 per class for robustness.
        """
        best_label = None
        best_score = -1.0

        for label, tmpl_list in self.templates.items():
            scores = []
            for tmpl in tmpl_list:
                res = cv2.matchTemplate(
                    live_edges.astype(np.float32),
                    tmpl.astype(np.float32),
                    cv2.TM_CCOEFF_NORMED
                )
                scores.append(float(res[0, 0]))

            class_score = float(np.mean(sorted(scores, reverse=True)[:5]))
            if class_score > best_score:
                best_score = class_score
                best_label = label

        if best_score < MIN_CONFIDENCE:
            return None, best_score
        return best_label, best_score

    # ── Debug view ────────────────────────────────────────────────────────────

    def debug_view(self, depth_mm: np.ndarray, joint_positions: dict, hand_key=None):
        """
        Returns a side-by-side BGR image:
            [normalised hand depth crop]  |  [Canny edges]
        Useful for tuning CANNY_LOW / CANNY_HIGH.
        """
        h     = self.roi_size
        blank = np.zeros((h, h * 2 + 4, 3), dtype=np.uint8)

        roi = self._extract_hand_roi(depth_mm, joint_positions, hand_key)
        if roi is None:
            cv2.putText(blank, "No hand", (4, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)
            return blank

        edges   = self._to_edge_template(roi)
        left    = cv2.cvtColor(cv2.resize(roi,   (h, h)), cv2.COLOR_GRAY2BGR)
        right   = cv2.cvtColor(cv2.resize(edges, (h, h)), cv2.COLOR_GRAY2BGR)
        divider = np.full((h, 4, 3), 80, dtype=np.uint8)
        return np.hstack([left, divider, right])


# ── Quick standalone test ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else DATASET_ROOT
    print(f"Loading MSR Gesture 3D from: {root}\n")
    rec = GestureRecognizer(root)
    if rec.templates:
        print("Classes:", list(rec.templates.keys()))
    else:
        print("No templates loaded. Check DATASET_ROOT path.")
