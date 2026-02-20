import clr
import sys
import os
import numpy as np
import cv2
from datetime import datetime

# Load System
clr.AddReference("System")
from System import Array, Byte, UInt16

print("="*60)
print("KINECT V1 - GESTURE RECOGNITION")
print("="*60)
print("Capture templates then detect: Open Palm, Fist, Thumbs Up")
print("="*60)

# Add paths
sdk_path = r"C:\Program Files\Microsoft SDKs\Kinect\v1.8\Assemblies"
sys.path.append(sdk_path)

# Load Kinect
clr.AddReference("Microsoft.Kinect")
import Microsoft.Kinect as kinect

# Get sensor
sensor = kinect.KinectSensor.KinectSensors[0]
print(f"Kinect found: {sensor.DeviceConnectionId}")

# Enable streams
sensor.DepthStream.Enable(kinect.DepthImageFormat.Resolution640x480Fps30)
sensor.ColorStream.Enable(kinect.ColorImageFormat.RgbResolution640x480Fps30)
sensor.Start()
print("Depth + Color streams enabled")

# Enable skeleton stream
sensor.SkeletonStream.Enable()
sensor.SkeletonStream.TrackingMode = kinect.SkeletonTrackingMode.Default
print("Skeleton stream enabled")

# ============================================================
# CALIBRATION PARAMETERS
# ============================================================
MAX_RAW_CLOSE = 10000
MIN_RAW_FAR   = 50000
REAL_NEAR_MM  = 6000
REAL_FAR_MM   = 10000

SKELETON_MIN_MM = 1000
SKELETON_MAX_MM = 4000

# ============================================================
# GESTURE RECOGNITION PARAMETERS
# ============================================================

# Folder where captured templates are saved and loaded from
TEMPLATE_DIR = r"C:\kinect_templates"

# Size of the hand ROI crop (pixels) — must match what templates were captured at
ROI_SIZE = 96

# Canny edge detection thresholds — tune with the debug window
CANNY_LOW  = 20
CANNY_HIGH = 60

# Minimum match score [0-1] to report a gesture
MIN_CONFIDENCE = 0.40

# The 3 gestures — keys are used for filenames and display labels
GESTURE_KEYS = ["open_palm", "fist", "thumbs_up"]
GESTURE_LABELS = {
    "open_palm":  "Open Palm",
    "fist":       "Fist",
    "thumbs_up":  "Thumbs Up",
}

# Keyboard shortcuts to capture each template
CAPTURE_KEYS = {
    ord('1'): "open_palm",
    ord('2'): "fist",
    ord('3'): "thumbs_up",
}

# ============================================================
# GESTURE STATE
# ============================================================
current_gesture  = "None"
gesture_cooldown = 0
joint_positions  = {}
skeleton_tracked = False
snapshot_counter = 0

# Template storage: { gesture_key: edge_image (ROI_SIZE x ROI_SIZE uint8) }
templates = {}

# ============================================================
# JOINT CONNECTIONS FOR SKELETON DRAWING
# ============================================================
JOINT_CONNECTIONS = [
    (kinect.JointType.Head,            kinect.JointType.ShoulderCenter),
    (kinect.JointType.ShoulderCenter,  kinect.JointType.ShoulderLeft),
    (kinect.JointType.ShoulderCenter,  kinect.JointType.ShoulderRight),
    (kinect.JointType.ShoulderLeft,    kinect.JointType.ElbowLeft),
    (kinect.JointType.ShoulderRight,   kinect.JointType.ElbowRight),
    (kinect.JointType.ElbowLeft,       kinect.JointType.WristLeft),
    (kinect.JointType.ElbowRight,      kinect.JointType.WristRight),
    (kinect.JointType.WristLeft,       kinect.JointType.HandLeft),
    (kinect.JointType.WristRight,      kinect.JointType.HandRight),
    (kinect.JointType.ShoulderCenter,  kinect.JointType.Spine),
    (kinect.JointType.Spine,           kinect.JointType.HipCenter),
    (kinect.JointType.HipCenter,       kinect.JointType.HipLeft),
    (kinect.JointType.HipCenter,       kinect.JointType.HipRight),
    (kinect.JointType.HipLeft,         kinect.JointType.KneeLeft),
    (kinect.JointType.HipRight,        kinect.JointType.KneeRight),
    (kinect.JointType.KneeLeft,        kinect.JointType.AnkleLeft),
    (kinect.JointType.KneeRight,       kinect.JointType.AnkleRight),
    (kinect.JointType.AnkleLeft,       kinect.JointType.FootLeft),
    (kinect.JointType.AnkleRight,      kinect.JointType.FootRight),
]

# ============================================================
# CALIBRATION HELPERS
# ============================================================

def raw_to_mm(raw_value):
    if raw_value <= 0:
        return 0
    raw_clipped = np.clip(raw_value, MAX_RAW_CLOSE, MIN_RAW_FAR)
    slope = (REAL_FAR_MM - REAL_NEAR_MM) / (MIN_RAW_FAR - MAX_RAW_CLOSE)
    intercept = REAL_NEAR_MM - slope * MAX_RAW_CLOSE
    return slope * raw_clipped + intercept

def mm_to_raw(mm_value):
    if mm_value <= 0:
        return 0
    slope = (REAL_FAR_MM - REAL_NEAR_MM) / (MIN_RAW_FAR - MAX_RAW_CLOSE)
    intercept = REAL_NEAR_MM - slope * MAX_RAW_CLOSE
    raw = (mm_value - intercept) / slope
    return np.clip(raw, MAX_RAW_CLOSE, MIN_RAW_FAR)

# ============================================================
# GESTURE RECOGNITION HELPERS
# ============================================================

def depth_roi_to_edges(gray_uint8):
    """Resize -> GaussianBlur -> Canny. Returns ROI_SIZE x ROI_SIZE edge image."""
    if gray_uint8 is None or gray_uint8.size == 0:
        return None
    resized = cv2.resize(gray_uint8, (ROI_SIZE, ROI_SIZE), interpolation=cv2.INTER_AREA)
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)
    return cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)


def extract_hand_roi(depth_mm):
    """
    Crop a ROI_SIZE x ROI_SIZE region centred on the right hand joint
    (falls back to left hand if right is not tracked).
    Returns a normalised uint8 grayscale crop, or None.
    """
    global joint_positions

    hand_key = None
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
    half   = ROI_SIZE // 2
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


def save_template(gesture_key, depth_mm):
    """
    Capture the current hand ROI, run edge detection, and save as a template.
    Also writes the raw grayscale crop as a PNG so you can inspect it.
    """
    os.makedirs(TEMPLATE_DIR, exist_ok=True)

    roi = extract_hand_roi(depth_mm)
    if roi is None:
        print(f"[Template] Could not capture '{gesture_key}' — no hand visible")
        return False

    edges = depth_roi_to_edges(roi)
    if edges is None:
        return False

    # Store in memory
    templates[gesture_key] = edges

    # Save edge image and raw crop to disk so they persist between runs
    edge_path = os.path.join(TEMPLATE_DIR, f"{gesture_key}_edges.png")
    raw_path  = os.path.join(TEMPLATE_DIR, f"{gesture_key}_raw.png")
    cv2.imwrite(edge_path, edges)
    cv2.imwrite(raw_path,  roi)

    print(f"[Template] Saved '{GESTURE_LABELS[gesture_key]}' -> {edge_path}")
    return True


def load_templates_from_disk():
    """Load any previously saved edge templates from TEMPLATE_DIR on startup."""
    if not os.path.exists(TEMPLATE_DIR):
        return
    for key in GESTURE_KEYS:
        path = os.path.join(TEMPLATE_DIR, f"{key}_edges.png")
        if os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                templates[key] = cv2.resize(img, (ROI_SIZE, ROI_SIZE))
                print(f"[Template] Loaded '{GESTURE_LABELS[key]}' from disk")


def classify_gesture(depth_mm):
    """
    Extract hand ROI, apply edge detection, match against all loaded templates.
    Returns (label_string, score) or (None, 0) if no confident match.
    """
    if not templates:
        return None, 0.0

    roi = extract_hand_roi(depth_mm)
    if roi is None:
        return None, 0.0

    live_edges = depth_roi_to_edges(roi)
    if live_edges is None:
        return None, 0.0

    best_key   = None
    best_score = -1.0

    for key, tmpl in templates.items():
        # Both images are ROI_SIZE x ROI_SIZE — result is a single 1x1 score
        result = cv2.matchTemplate(
            live_edges.astype(np.float32),
            tmpl.astype(np.float32),
            cv2.TM_CCOEFF_NORMED
        )
        score = float(result[0, 0])
        if score > best_score:
            best_score = score
            best_key   = key

    if best_score < MIN_CONFIDENCE:
        return None, best_score

    return GESTURE_LABELS[best_key], best_score


def build_debug_roi_view(depth_mm):
    """
    Returns a small side-by-side image: [depth crop] | [Canny edges]
    Shown in a separate window so you can tune CANNY_LOW / CANNY_HIGH.
    """
    h     = ROI_SIZE
    blank = np.zeros((h, h * 2 + 4, 3), dtype=np.uint8)

    roi = extract_hand_roi(depth_mm)
    if roi is None:
        cv2.putText(blank, "No hand", (4, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)
        return blank

    edges   = depth_roi_to_edges(roi)
    left    = cv2.cvtColor(cv2.resize(roi,   (h, h)), cv2.COLOR_GRAY2BGR)
    right   = cv2.cvtColor(cv2.resize(edges, (h, h)), cv2.COLOR_GRAY2BGR)
    divider = np.full((h, 4, 3), 80, dtype=np.uint8)
    return np.hstack([left, divider, right])

# ============================================================
# SKELETON HANDLER
# ============================================================

def skeleton_frame_ready(sender, e):
    global current_gesture, gesture_cooldown, joint_positions, skeleton_tracked

    frame = e.OpenSkeletonFrame()
    if frame:
        skeletons = [None] * frame.SkeletonArrayLength
        frame.CopySkeletonDataTo(skeletons)

        joint_positions  = {}
        skeleton_tracked = False

        for skeleton in skeletons:
            if skeleton is not None:
                if skeleton.TrackingState == kinect.SkeletonTrackingState.Tracked:
                    skeleton_tracked = True

                    for joint_type in [
                        kinect.JointType.Head,
                        kinect.JointType.ShoulderCenter,
                        kinect.JointType.ShoulderLeft,
                        kinect.JointType.ShoulderRight,
                        kinect.JointType.ElbowLeft,
                        kinect.JointType.ElbowRight,
                        kinect.JointType.WristLeft,
                        kinect.JointType.WristRight,
                        kinect.JointType.HandLeft,
                        kinect.JointType.HandRight,
                        kinect.JointType.Spine,
                        kinect.JointType.HipCenter,
                        kinect.JointType.HipLeft,
                        kinect.JointType.HipRight,
                        kinect.JointType.KneeLeft,
                        kinect.JointType.KneeRight,
                        kinect.JointType.AnkleLeft,
                        kinect.JointType.AnkleRight,
                        kinect.JointType.FootLeft,
                        kinect.JointType.FootRight,
                    ]:
                        joint = skeleton.Joints[joint_type]
                        if joint.TrackingState != kinect.JointTrackingState.NotTracked:
                            try:
                                depth_pt = sensor.MapSkeletonPointToDepth(joint.Position)
                                if 0 <= depth_pt.X < 640 and 0 <= depth_pt.Y < 480:
                                    joint_positions[joint_type] = {
                                        'x':       int(depth_pt.X),
                                        'y':       int(depth_pt.Y),
                                        'z':       joint.Position.Z * 1000,
                                        'tracked': joint.TrackingState == kinect.JointTrackingState.Tracked,
                                    }
                            except:
                                pass

        frame.Dispose()

        if gesture_cooldown > 0:
            gesture_cooldown -= 1

# ============================================================
# SKELETON DRAWING
# ============================================================

def draw_skeleton(image, joint_positions):
    for joint1, joint2 in JOINT_CONNECTIONS:
        if joint1 in joint_positions and joint2 in joint_positions:
            pt1 = (joint_positions[joint1]['x'], joint_positions[joint1]['y'])
            pt2 = (joint_positions[joint2]['x'], joint_positions[joint2]['y'])
            if (0 <= pt1[0] < 640 and 0 <= pt1[1] < 480 and
                    0 <= pt2[0] < 640 and 0 <= pt2[1] < 480):
                cv2.line(image, pt1, pt2, (0, 255, 255), 2)

    for joint_type, pos in joint_positions.items():
        if 0 <= pos['x'] < 640 and 0 <= pos['y'] < 480:
            if 'Hand' in str(joint_type):
                color = (0, 255, 255)
            elif 'Head' in str(joint_type):
                color = (0, 0, 255)
            else:
                color = (255, 255, 255)
            radius = 5 if pos['tracked'] else 3
            cv2.circle(image, (pos['x'], pos['y']), radius, color, -1)

# ============================================================
# DEPTH FRAME HANDLER
# ============================================================

# Shared storage so the main loop can read the latest processed frames
current_depth_img = None
current_depth_mm  = None   # raw mm array needed for gesture capture
current_color_img = None

def depth_frame_ready(sender, e):
    global current_depth_img, current_depth_mm

    frame = e.OpenDepthImageFrame()
    if not frame:
        return

    depth_bytes = Array.CreateInstance(UInt16, frame.PixelDataLength)
    frame.CopyPixelDataTo(depth_bytes)

    depth_array = np.frombuffer(depth_bytes, dtype=np.uint16)
    depth_raw   = depth_array.reshape(frame.Height, frame.Width)

    has_data = depth_raw > 0

    # Convert raw to mm using calibration
    depth_mm = np.zeros_like(depth_raw, dtype=np.float32)
    if has_data.any():
        raw_values  = depth_raw[has_data].astype(np.float32)
        raw_clipped = np.clip(raw_values, MAX_RAW_CLOSE, MIN_RAW_FAR)
        slope       = (REAL_FAR_MM - REAL_NEAR_MM) / (MIN_RAW_FAR - MAX_RAW_CLOSE)
        intercept   = REAL_NEAR_MM - slope * MAX_RAW_CLOSE
        depth_mm[has_data] = slope * raw_clipped + intercept

    # Store mm array so the main loop can use it for gesture capture
    current_depth_mm = depth_mm

    # ── Gesture recognition (edge detection + template matching) ──────────
    cv_label, cv_score = classify_gesture(depth_mm)
    if cv_label and gesture_cooldown <= 0:
        global current_gesture
        current_gesture = f"{cv_label} ({cv_score:.2f})"

    # ── Build depth visualisation ─────────────────────────────────────────
    depth_display = np.zeros((480, 640, 3), dtype=np.uint8)

    if has_data.any():
        raw_min  = np.min(depth_raw[has_data])
        raw_max  = np.max(depth_raw[has_data])
        raw_mean = np.mean(depth_raw[has_data])

        mm_min   = np.min(depth_mm[has_data])
        mm_max   = np.max(depth_mm[has_data])
        mm_mean  = np.mean(depth_mm[has_data])

        display_norm = np.zeros(depth_raw.shape, dtype=np.uint8)
        scaled = np.clip((depth_mm * 255.0 / REAL_FAR_MM), 0, 255).astype(np.uint8)
        display_norm[has_data] = scaled[has_data]
        depth_display = cv2.applyColorMap(display_norm, cv2.COLORMAP_JET)

        # Calibration info
        cv2.putText(depth_display, "CALIBRATION:",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(depth_display,
                    f"Raw {MAX_RAW_CLOSE}->{REAL_NEAR_MM}mm | Raw {MIN_RAW_FAR}->{REAL_FAR_MM}mm",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(depth_display,
                    f"CURRENT: Raw {raw_mean:.0f} -> {mm_mean:.0f}mm",
                    (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(depth_display,
                    f"Range: {mm_min:.0f}-{mm_max:.0f}mm",
                    (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Skeleton range indicator
        in_range = SKELETON_MIN_MM < mm_mean < SKELETON_MAX_MM
        if in_range:
            cv2.putText(depth_display, "IN SKELETON RANGE",
                        (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(depth_display,
                        f"NEED {SKELETON_MIN_MM}-{SKELETON_MAX_MM}mm FOR SKELETON",
                        (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Skeleton overlay
        if skeleton_tracked and joint_positions:
            draw_skeleton(depth_display, joint_positions)

        # Gesture result
        gesture_color = (0, 255, 0) if cv_label else (100, 100, 100)
        cv2.putText(depth_display, f"GESTURE: {current_gesture}",
                    (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, gesture_color, 2)

        # Template status — show which of the 3 have been captured
        status_parts = []
        for key in GESTURE_KEYS:
            label  = GESTURE_LABELS[key]
            marker = "OK" if key in templates else "--"
            status_parts.append(f"{marker} {label}")
        cv2.putText(depth_display, "  |  ".join(status_parts),
                    (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 0), 1)

    frame.Dispose()
    current_depth_img = depth_display

# ============================================================
# COLOR FRAME HANDLER
# ============================================================

def color_frame_ready(sender, e):
    global current_color_img

    frame = e.OpenColorImageFrame()
    if not frame:
        return

    color_bytes = Array.CreateInstance(Byte, frame.PixelDataLength)
    frame.CopyPixelDataTo(color_bytes)

    color_array = np.frombuffer(color_bytes, dtype=np.uint8)
    color_image = color_array.reshape(frame.Height, frame.Width, 4)
    color_bgr   = color_image[:, :, :3].copy()

    cv2.putText(color_bgr, "COLOR VIEW",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(color_bgr, datetime.now().strftime("%H:%M:%S"),
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    if skeleton_tracked and joint_positions:
        draw_skeleton(color_bgr, joint_positions)

    frame.Dispose()
    current_color_img = color_bgr

# ============================================================
# REGISTER EVENT HANDLERS
# ============================================================

sensor.SkeletonFrameReady += skeleton_frame_ready
sensor.DepthFrameReady    += depth_frame_ready
sensor.ColorFrameReady    += color_frame_ready

# ============================================================
# LOAD ANY PREVIOUSLY SAVED TEMPLATES
# ============================================================

load_templates_from_disk()

# ============================================================
# STARTUP INSTRUCTIONS
# ============================================================

print("\n" + "="*60)
print("GESTURE RECOGNITION ACTIVE")
print("="*60)
print("\nSTEP 1 — Capture templates (do this once):")
print("   Hold OPEN PALM in front of Kinect, press  [1]")
print("   Hold FIST      in front of Kinect, press  [2]")
print("   Hold THUMBS UP in front of Kinect, press  [3]")
print("\nSTEP 2 — Once all 3 are captured, recognition starts automatically")
print("\nOTHER KEYS:")
print("   [d] toggle hand ROI debug window")
print("   [q] quit\n")
print(f"Templates saved to: {TEMPLATE_DIR}")
print("="*60 + "\n")

# ============================================================
# MAIN LOOP
# ============================================================

show_debug = False

try:
    while True:
        depth_img = current_depth_img
        color_img = current_color_img
        depth_mm  = current_depth_mm

        if depth_img is not None and color_img is not None:
            color_resized = cv2.resize(color_img, (640, 480))
            combined = np.hstack([depth_img, color_resized])
            cv2.line(combined, (640, 0), (640, 480), (255, 255, 255), 2)
            cv2.imshow('Kinect v1 - Gesture Recognition', combined)

        # Debug ROI window (toggled with 'd')
        if show_debug and depth_mm is not None:
            debug_img = build_debug_roi_view(depth_mm)
            # Scale up so it's easy to see
            debug_big = cv2.resize(debug_img, (ROI_SIZE * 4 + 16, ROI_SIZE * 2),
                                   interpolation=cv2.INTER_NEAREST)
            cv2.putText(debug_big, "Depth crop", (4, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            cv2.putText(debug_big, "Canny edges", (ROI_SIZE * 2 + 20, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            cv2.imshow('Hand ROI Debug', debug_big)
        elif not show_debug:
            cv2.destroyWindow('Hand ROI Debug')

        key = cv2.waitKey(10) & 0xFF

        # Template capture keys
        if key in CAPTURE_KEYS and depth_mm is not None:
            gesture_key = CAPTURE_KEYS[key]
            if save_template(gesture_key, depth_mm):
                print(f"  Captured: {GESTURE_LABELS[gesture_key]}")
                captured = [GESTURE_LABELS[k] for k in GESTURE_KEYS if k in templates]
                missing  = [GESTURE_LABELS[k] for k in GESTURE_KEYS if k not in templates]
                if captured:
                    print(f"  Have:    {', '.join(captured)}")
                if missing:
                    print(f"  Missing: {', '.join(missing)}")
                if len(templates) == len(GESTURE_KEYS):
                    print("  All 3 templates captured — recognition active!")

        elif key == ord('d'):
            show_debug = not show_debug
            print(f"Debug window {'ON' if show_debug else 'OFF'}")

        elif key == ord('q'):
            break

except KeyboardInterrupt:
    pass

sensor.Stop()
cv2.destroyAllWindows()
print("\nDone.")
