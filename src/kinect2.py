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
print("KINECT V1 - LIVE CALIBRATION MODE")
print("="*60)
print("Change parameters and SEE the difference!")
print("="*60)

# Add paths
sdk_path = r"C:\Program Files\Microsoft SDKs\Kinect\v1.8\Assemblies"
sys.path.append(sdk_path)

# Load Kinect
clr.AddReference("Microsoft.Kinect")
import Microsoft.Kinect as kinect

# Get sensor
sensor = kinect.KinectSensor.KinectSensors[0]
print(f"✓ Kinect found: {sensor.DeviceConnectionId}")

# Enable streams
sensor.DepthStream.Enable(kinect.DepthImageFormat.Resolution640x480Fps30)
sensor.ColorStream.Enable(kinect.ColorImageFormat.RgbResolution640x480Fps30)
sensor.Start()
print("✓ Depth + Color streams enabled")

# Enable skeleton stream
sensor.SkeletonStream.Enable()
sensor.SkeletonStream.TrackingMode = kinect.SkeletonTrackingMode.Default
print("✓ Skeleton stream enabled")

# ============================================================
# CALIBRATION PARAMETERS - CHANGE THESE AND SEE RESULTS!
# ============================================================
# Your Kinect's behavior: CLOSE = SMALL number, FAR = LARGE number
MAX_RAW_CLOSE = 10000     # Raw value when object is VERY CLOSE
MIN_RAW_FAR = 50000      # Raw value when object is VERY FAR
REAL_NEAR_MM = 6000      # Actual distance in mm when VERY CLOSE
REAL_FAR_MM = 10000      # Actual distance in mm when VERY FAR

# Skeleton tracking range (fixed by Kinect hardware)
SKELETON_MIN_MM = 1000   # Minimum distance for skeleton tracking
SKELETON_MAX_MM = 4000  # Maximum distance for skeleton tracking

# For saving snapshots
snapshot_counter = 0

# Gesture detection variables
current_gesture = "None"
gesture_cooldown = 0
joint_positions = {}
skeleton_tracked = False

# Define joint connections for drawing skeleton
JOINT_CONNECTIONS = [
    (kinect.JointType.Head, kinect.JointType.ShoulderCenter),
    (kinect.JointType.ShoulderCenter, kinect.JointType.ShoulderLeft),
    (kinect.JointType.ShoulderCenter, kinect.JointType.ShoulderRight),
    (kinect.JointType.ShoulderLeft, kinect.JointType.ElbowLeft),
    (kinect.JointType.ShoulderRight, kinect.JointType.ElbowRight),
    (kinect.JointType.ElbowLeft, kinect.JointType.WristLeft),
    (kinect.JointType.ElbowRight, kinect.JointType.WristRight),
    (kinect.JointType.WristLeft, kinect.JointType.HandLeft),
    (kinect.JointType.WristRight, kinect.JointType.HandRight),
    (kinect.JointType.ShoulderCenter, kinect.JointType.Spine),
    (kinect.JointType.Spine, kinect.JointType.HipCenter),
    (kinect.JointType.HipCenter, kinect.JointType.HipLeft),
    (kinect.JointType.HipCenter, kinect.JointType.HipRight),
    (kinect.JointType.HipLeft, kinect.JointType.KneeLeft),
    (kinect.JointType.HipRight, kinect.JointType.KneeRight),
    (kinect.JointType.KneeLeft, kinect.JointType.AnkleLeft),
    (kinect.JointType.KneeRight, kinect.JointType.AnkleRight),
    (kinect.JointType.AnkleLeft, kinect.JointType.FootLeft),
    (kinect.JointType.AnkleRight, kinect.JointType.FootRight),
]

def raw_to_mm(raw_value):
    """Convert raw depth value to millimeters using current calibration"""
    if raw_value <= 0:
        return 0
    
    # Clip to calibration range
    raw_clipped = np.clip(raw_value, MAX_RAW_CLOSE, MIN_RAW_FAR)
    
    # Direct linear mapping (raw increases with distance)
    slope = (REAL_FAR_MM - REAL_NEAR_MM) / (MIN_RAW_FAR - MAX_RAW_CLOSE)
    intercept = REAL_NEAR_MM - slope * MAX_RAW_CLOSE
    
    return slope * raw_clipped + intercept

def mm_to_raw(mm_value):
    """Convert millimeters to raw value (inverse of raw_to_mm)"""
    if mm_value <= 0:
        return 0
    
    slope = (REAL_FAR_MM - REAL_NEAR_MM) / (MIN_RAW_FAR - MAX_RAW_CLOSE)
    intercept = REAL_NEAR_MM - slope * MAX_RAW_CLOSE
    
    # Inverse: raw = (mm - intercept) / slope
    raw = (mm_value - intercept) / slope
    return np.clip(raw, MAX_RAW_CLOSE, MIN_RAW_FAR)

def skeleton_frame_ready(sender, e):
    global current_gesture, gesture_cooldown, joint_positions, skeleton_tracked
    
    frame = e.OpenSkeletonFrame()
    if frame:
        skeletons = [None] * frame.SkeletonArrayLength
        frame.CopySkeletonDataTo(skeletons)
        
        joint_positions = {}
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
                        kinect.JointType.FootRight
                    ]:
                        joint = skeleton.Joints[joint_type]
                        if joint.TrackingState != kinect.JointTrackingState.NotTracked:
                            try:
                                depth_pt = sensor.MapSkeletonPointToDepth(joint.Position)
                                if 0 <= depth_pt.X < 640 and 0 <= depth_pt.Y < 480:
                                    joint_positions[joint_type] = {
                                        'x': int(depth_pt.X),
                                        'y': int(depth_pt.Y),
                                        'z': joint.Position.Z * 1000,  # Convert to mm
                                        'tracked': joint.TrackingState == kinect.JointTrackingState.Tracked
                                    }
                            except:
                                pass
                    
                    # Gesture detection
                    if gesture_cooldown <= 0 and len(joint_positions) > 5:
                        head = joint_positions.get(kinect.JointType.Head)
                        hand_right = joint_positions.get(kinect.JointType.HandRight)
                        hand_left = joint_positions.get(kinect.JointType.HandLeft)
                        
                        if head and hand_right and hand_right['y'] < head['y'] - 30:
                            current_gesture = "👆 HAND UP"
                            gesture_cooldown = 30
                        elif head and hand_right and hand_right['y'] > head['y'] + 100:
                            current_gesture = "👇 HAND DOWN"
                            gesture_cooldown = 30
        
        frame.Dispose()
        
        if gesture_cooldown > 0:
            gesture_cooldown -= 1

def draw_skeleton(image, joint_positions):
    """Draw skeleton lines and joints on image"""
    # Draw connections
    for joint1, joint2 in JOINT_CONNECTIONS:
        if joint1 in joint_positions and joint2 in joint_positions:
            pt1 = (joint_positions[joint1]['x'], joint_positions[joint1]['y'])
            pt2 = (joint_positions[joint2]['x'], joint_positions[joint2]['y'])
            
            if (0 <= pt1[0] < 640 and 0 <= pt1[1] < 480 and
                0 <= pt2[0] < 640 and 0 <= pt2[1] < 480):
                cv2.line(image, pt1, pt2, (0, 255, 255), 2)
    
    # Draw joints
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

def depth_frame_ready(sender, e):
    frame = e.OpenDepthImageFrame()
    if frame:
        # Get depth data
        depth_bytes = Array.CreateInstance(UInt16, frame.PixelDataLength)
        frame.CopyPixelDataTo(depth_bytes)
        
        # Convert to numpy
        depth_array = np.frombuffer(depth_bytes, dtype=np.uint16)
        depth_raw = depth_array.reshape(frame.Height, frame.Width)
        
        # Only consider pixels with data
        has_data = depth_raw > 0
        
        # CONVERT RAW TO MILLIMETERS USING CURRENT CALIBRATION
        depth_mm = np.zeros_like(depth_raw, dtype=np.float32)
        if has_data.any():
            # Apply calibration to EVERY pixel
            raw_values = depth_raw[has_data].astype(np.float32)
            
            # Clip to calibration range
            raw_clipped = np.clip(raw_values, MAX_RAW_CLOSE, MIN_RAW_FAR)
            
            # Direct linear mapping
            slope = (REAL_FAR_MM - REAL_NEAR_MM) / (MIN_RAW_FAR - MAX_RAW_CLOSE)
            intercept = REAL_NEAR_MM - slope * MAX_RAW_CLOSE
            
            mm_values = slope * raw_clipped + intercept
            depth_mm[has_data] = mm_values
        
        # Create visualization
        depth_display = np.zeros((480, 640, 3), dtype=np.uint8)
        
        if has_data.any():
            # Get statistics
            raw_min = np.min(depth_raw[has_data])
            raw_max = np.max(depth_raw[has_data])
            raw_mean = np.mean(depth_raw[has_data])
            
            mm_min = np.min(depth_mm[has_data])
            mm_max = np.max(depth_mm[has_data])
            mm_mean = np.mean(depth_mm[has_data])
            
            # Normalize for display (using current REAL_FAR_MM as max)
            display_norm = np.zeros(depth_raw.shape, dtype=np.uint8)
            scaled = np.clip((depth_mm * 255.0 / REAL_FAR_MM), 0, 255).astype(np.uint8)
            display_norm[has_data] = scaled[has_data]
            depth_display = cv2.applyColorMap(display_norm, cv2.COLORMAP_JET)
            
            # Show calibration info
            cv2.putText(depth_display, f"CALIBRATION:", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(depth_display, f"Raw {MAX_RAW_CLOSE}->{REAL_NEAR_MM}mm | Raw {MIN_RAW_FAR}->{REAL_FAR_MM}mm", 
                       (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Show current readings
            cv2.putText(depth_display, f"CURRENT: Raw {raw_mean:.0f} → {mm_mean:.0f}mm", 
                       (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(depth_display, f"Range: {mm_min:.0f}-{mm_max:.0f}mm", 
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Skeleton range indicator
            in_skeleton_range = (mm_mean > SKELETON_MIN_MM) and (mm_mean < SKELETON_MAX_MM)
            if in_skeleton_range:
                cv2.putText(depth_display, "✓ IN SKELETON RANGE", 
                           (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(depth_display, f"✗ NEED {SKELETON_MIN_MM}-{SKELETON_MAX_MM}mm FOR SKELETON", 
                           (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Draw skeleton
            global joint_positions, skeleton_tracked
            if skeleton_tracked and joint_positions:
                draw_skeleton(depth_display, joint_positions)
                cv2.putText(depth_display, f"GESTURE: {current_gesture}", 
                           (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        frame.Dispose()
        return depth_display

def color_frame_ready(sender, e):
    frame = e.OpenColorImageFrame()
    if frame:
        color_bytes = Array.CreateInstance(Byte, frame.PixelDataLength)
        frame.CopyPixelDataTo(color_bytes)
        
        color_array = np.frombuffer(color_bytes, dtype=np.uint8)
        color_image = color_array.reshape(frame.Height, frame.Width, 4)
        color_bgr = color_image[:, :, :3].copy()
        
        # Add text overlay
        cv2.putText(color_bgr, "COLOR VIEW", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(color_bgr, datetime.now().strftime("%H:%M:%S"), 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw skeleton
        global joint_positions, skeleton_tracked
        if skeleton_tracked and joint_positions:
            draw_skeleton(color_bgr, joint_positions)
        
        frame.Dispose()
        return color_bgr

# Register skeleton handler
sensor.SkeletonFrameReady += skeleton_frame_ready

# Storage for current frames
current_depth = None
current_color = None

def update_frames():
    global current_depth, current_color
    
    def depth_handler(sender, e):
        global current_depth
        current_depth = depth_frame_ready(sender, e)
    
    def color_handler(sender, e):
        global current_color
        current_color = color_frame_ready(sender, e)
    
    sensor.DepthFrameReady += depth_handler
    sensor.ColorFrameReady += color_handler

update_frames()

print("\n" + "="*60)
print("🎥 LIVE CALIBRATION ACTIVE!")
print("="*60)
print("\n📏 CURRENT PARAMETERS:")
print(f"   MAX_RAW_CLOSE = {MAX_RAW_CLOSE} (raw when close)")
print(f"   MIN_RAW_FAR = {MIN_RAW_FAR} (raw when far)")
print(f"   REAL_NEAR_MM = {REAL_NEAR_MM}mm (actual close distance)")
print(f"   REAL_FAR_MM = {REAL_FAR_MM}mm (actual far distance)")
print("\n🔧 TO CALIBRATE:")
print("   1. Put hand at 500mm - note RAW value → update MAX_RAW_CLOSE")
print("   2. Put hand at 2000mm - note RAW value → update MIN_RAW_FAR")
print("   3. Adjust REAL_NEAR_MM and REAL_FAR_MM to match actual distances")
print("\n⌨️ Press 'q' to quit\n")

try:
    while True:
        depth_img = current_depth
        color_img = current_color
        
        if depth_img is not None and color_img is not None:
            color_resized = cv2.resize(color_img, (640, 480))
            combined = np.hstack([depth_img, color_resized])
            cv2.line(combined, (640, 0), (640, 480), (255, 255, 255), 2)
            cv2.imshow('Kinect v1 - Live Calibration', combined)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass

sensor.Stop()
cv2.destroyAllWindows()
print("\n✅ CALIBRATION COMPLETE!")