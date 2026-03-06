#!/usr/bin/env python3
"""
Hand Gesture Recognition System - Webcam Version
MSR Gesture3D Dataset (Training) + Webcam (Real-time Recognition)

Team Members: [Add your names and IDs here]

This single file handles:
1. Webcam setup and verification
2. Loading MSR Gesture3D dataset (.mat files)
3. Training gesture templates
4. Real-time gesture recognition and matching

Usage:
    python main.py

Controls:
    Q - Quit
    T - Show template info
    ESC - Exit
"""

import numpy as np
import cv2
import os
import time
from pathlib import Path
from collections import deque
import pickle

# Check required libraries
try:
    from scipy.io import loadmat
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("WARNING: scipy not installed - Cannot load .mat files")
    print("Install: pip install scipy")


class MSRGesture3DHandler:
    """Handle MSR Gesture3D dataset loading and processing"""
    
    # Gesture name mapping (from MSR Gesture3D documentation)
    GESTURE_MAPPING = {
        1: "beat_both",
        2: "beat_long", 
        3: "boxing",
        4: "clap_hands",
        5: "wave",
        6: "throw_forward",
        7: "throw_left",
        8: "throw_right",
        9: "throw_up",
        10: "point",
        11: "draw_circle_CW",
        12: "draw_circle_CCW"
    }
    
    # Map to simpler gesture names for our system
    SIMPLE_MAPPING = {
        "wave": "open_palm",
        "clap_hands": "fist",
        "point": "pointing",
        "beat_both": "peace",
        "boxing": "thumbs_up"
    }
    
    def __init__(self, dataset_path="./MSRGesture3D"):
        """
        Initialize dataset handler
        
        Args:
            dataset_path: Path to extracted MSR Gesture3D folder
        """
        self.dataset_path = Path(dataset_path)
        self.loaded_templates = {}
        
    def load_mat_file(self, filepath):
        """
        Load a .mat file from MSR Gesture3D dataset
        
        Args:
            filepath: Path to .mat file
            
        Returns:
            Depth sequence as numpy array or None
        """
        if not SCIPY_AVAILABLE:
            print("Error: scipy not installed")
            return None
        
        try:
            mat_data = loadmat(filepath)
            
            # MSR Gesture3D stores depth data in 'Video' field
            if 'Video' in mat_data:
                depth_sequence = mat_data['Video']
                return depth_sequence
            
            # Fallback: try to find any array in the mat file
            for key, value in mat_data.items():
                if isinstance(value, np.ndarray) and value.ndim >= 2:
                    return value
            
            return None
            
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def get_gesture_type(self, n):
        """
        Get gesture type from sequence number
        
        Args:
            n: Sequence number (1-36)
            
        Returns:
            Gesture type (1-12)
        """
        # Each gesture has 3 sequences
        # n=1,2,3 → gesture 1
        # n=4,5,6 → gesture 2, etc.
        return ((n - 1) // 3) + 1
    
    def extract_features_from_frame(self, depth_frame):
        """
        Extract features from a single depth frame
        
        Args:
            depth_frame: 2D depth array
            
        Returns:
            5-dimensional feature vector or None
        """
        if depth_frame is None or depth_frame.size == 0:
            return None
        
        # Convert to uint8 for processing
        depth_normalized = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX)
        depth_uint8 = depth_normalized.astype(np.uint8)
        
        # Threshold to get hand region
        _, binary = cv2.threshold(depth_uint8, 30, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return None
        
        # Get largest contour (hand)
        contour = max(contours, key=cv2.contourArea)
        
        if cv2.contourArea(contour) < 100:
            return None
        
        # Extract features
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h if h != 0 else 0
        
        compactness = 4 * np.pi * area / (perimeter ** 2) if perimeter != 0 else 0
        
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area != 0 else 0
        
        # Convexity defects
        hull_indices = cv2.convexHull(contour, returnPoints=False)
        num_defects = 0
        if len(hull_indices) > 3 and len(contour) > 10:
            try:
                defects = cv2.convexityDefects(contour, hull_indices)
                num_defects = len(defects) if defects is not None else 0
            except:
                num_defects = 0
        
        features = np.array([
            area / 10000,
            aspect_ratio,
            compactness,
            solidity,
            num_defects / 10
        ])
        
        return features
    
    def train_templates_from_dataset(self, gestures_to_use=None):
        """
        Train gesture templates from MSR Gesture3D dataset
        
        Args:
            gestures_to_use: List of gesture IDs to train (1-12), or None for specific ones
            
        Returns:
            Dictionary of trained templates
        """
        print("\n" + "="*60)
        print("Training Templates from MSR Gesture3D Dataset")
        print("="*60)
        
        if not self.dataset_path.exists():
            print(f"ERROR: Dataset not found at {self.dataset_path}")
            print("Please extract MSRGesture3D.zip to this directory")
            return {}
        
        # Default: train gestures we mapped to simple names
        if gestures_to_use is None:
            gestures_to_use = [5, 4, 10, 1, 3]  # wave, clap, point, beat, boxing
        
        templates = {}
        
        for gesture_id in gestures_to_use:
            gesture_name = self.GESTURE_MAPPING.get(gesture_id, f"gesture_{gesture_id}")
            print(f"\nProcessing: {gesture_name} (ID {gesture_id})")
            
            features_list = []
            
            # Iterate through subjects (1-10)
            for subject in range(1, 11):
                # Calculate sequence numbers for this gesture
                # Gesture 1: n=1,2,3; Gesture 2: n=4,5,6, etc.
                base_n = (gesture_id - 1) * 3 + 1
                
                for offset in range(3):
                    n = base_n + offset
                    filename = f"sub_depth_{subject:02d}_{n:02d}.mat"
                    filepath = self.dataset_path / filename
                    
                    if not filepath.exists():
                        continue
                    
                    # Load .mat file
                    depth_sequence = self.load_mat_file(filepath)
                    
                    if depth_sequence is None:
                        continue
                    
                    # Extract features from middle frames of sequence
                    num_frames = depth_sequence.shape[2] if depth_sequence.ndim == 3 else 1
                    
                    # Sample 5 frames evenly across sequence
                    frame_indices = np.linspace(0, num_frames - 1, 5, dtype=int)
                    
                    for frame_idx in frame_indices:
                        if depth_sequence.ndim == 3:
                            frame = depth_sequence[:, :, frame_idx]
                        else:
                            frame = depth_sequence
                        
                        features = self.extract_features_from_frame(frame)
                        
                        if features is not None:
                            features_list.append(features)
            
            if len(features_list) > 0:
                # Average features
                avg_features = np.mean(features_list, axis=0)
                std_features = np.std(features_list, axis=0)
                
                # Map to simple name if available
                simple_name = self.SIMPLE_MAPPING.get(gesture_name, gesture_name)
                
                templates[simple_name] = {
                    'features': avg_features,
                    'std': std_features,
                    'original_name': gesture_name,
                    'num_samples': len(features_list)
                }
                
                print(f"  ✓ Trained '{simple_name}': {len(features_list)} samples")
                print(f"    Features: {avg_features}")
            else:
                print(f"  ✗ No samples found for {gesture_name}")
        
        # Save templates
        if templates:
            with open('trained_templates.pkl', 'wb') as f:
                pickle.dump(templates, f)
            print(f"\n✓ Saved {len(templates)} templates to trained_templates.pkl")
        
        print("="*60)
        
        self.loaded_templates = templates
        return templates


class WebcamGestureRecognizer:
    """Real-time gesture recognition using webcam"""
    
    def __init__(self, camera_id=0):
        """
        Initialize the recognizer
        
        Args:
            camera_id: Webcam device ID (usually 0)
        """
        self.templates = {}
        self.gesture_history = deque(maxlen=5)
        self.camera_id = camera_id
        self.cap = None
        
        # Background subtractor for hand isolation
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=True
        )
        
    def load_templates(self, templates_dict=None, template_file='trained_templates.pkl'):
        """
        Load gesture templates
        
        Args:
            templates_dict: Pre-loaded templates dictionary
            template_file: Path to saved templates file
        """
        if templates_dict:
            # Use provided templates
            for name, data in templates_dict.items():
                self.templates[name] = data['features']
            print(f"✓ Loaded {len(self.templates)} templates from provided dict")
        
        elif os.path.exists(template_file):
            # Load from file
            try:
                with open(template_file, 'rb') as f:
                    loaded = pickle.load(f)
                
                for name, data in loaded.items():
                    self.templates[name] = data['features']
                
                print(f"✓ Loaded {len(self.templates)} templates from {template_file}")
            except Exception as e:
                print(f"Error loading templates: {e}")
                self.use_default_templates()
        else:
            print("No trained templates found, using defaults")
            self.use_default_templates()
    
    def use_default_templates(self):
        """Use default hardcoded templates"""
        self.templates = {
            "open_palm": np.array([0.8, 1.0, 0.7, 0.9, 0.4]),
            "fist": np.array([0.5, 1.0, 0.9, 0.95, 0.0]),
            "thumbs_up": np.array([0.6, 0.4, 0.6, 0.85, 0.1]),
            "pointing": np.array([0.4, 0.3, 0.5, 0.7, 0.1]),
            "peace": np.array([0.6, 0.8, 0.6, 0.8, 0.2])
        }
        print("✓ Using default templates")
    
    def initialize_camera(self):
        """Initialize webcam"""
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            print(f"ERROR: Cannot open camera {self.camera_id}")
            return False
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print(f"✓ Camera {self.camera_id} initialized")
        return True
    
    def preprocess_frame(self, frame):
        """Isolate hand region from webcam frame"""
        if frame is None:
            return None
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        
        # Use adaptive threshold to create binary mask
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Clean up noise
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        return binary
    
    def extract_hand_contour(self, mask):
        """Extract hand contour from mask"""
        if mask is None:
            return None, None
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return None, None
        
        # Get largest contour (hand)
        contour = max(contours, key=cv2.contourArea)
        
        if cv2.contourArea(contour) < 1000:
            return None, None
        
        bbox = cv2.boundingRect(contour)
        
        return contour, bbox
    
    def extract_features(self, contour, bbox):
        """Extract feature vector from contour"""
        if contour is None:
            return None
        
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        x, y, w, h = bbox
        aspect_ratio = float(w) / h if h != 0 else 0
        
        compactness = 4 * np.pi * area / (perimeter ** 2) if perimeter != 0 else 0
        
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area != 0 else 0
        
        hull_indices = cv2.convexHull(contour, returnPoints=False)
        num_defects = 0
        if len(hull_indices) > 3 and len(contour) > 10:
            try:
                defects = cv2.convexityDefects(contour, hull_indices)
                num_defects = len(defects) if defects is not None else 0
            except:
                pass
        
        features = np.array([
            area / 10000,
            aspect_ratio,
            compactness,
            solidity,
            num_defects / 10
        ])
        
        return features
    
    def match_template(self, features):
        """Match features to templates"""
        if features is None or len(self.templates) == 0:
            return "Unknown", 0.0
        
        best_match = "Unknown"
        best_distance = float('inf')
        
        for gesture_name, template in self.templates.items():
            distance = np.linalg.norm(features - template)
            
            if distance < best_distance:
                best_distance = distance
                best_match = gesture_name
        
        confidence = max(0, 100 - (best_distance * 20))
        
        return best_match, confidence
    
    def smooth_prediction(self, gesture):
        """Smooth predictions over frames"""
        self.gesture_history.append(gesture)
        
        if len(self.gesture_history) > 0:
            return max(set(self.gesture_history), key=self.gesture_history.count)
        
        return gesture
    
    def visualize(self, frame, mask, contour, bbox, gesture, confidence):
        """Display results"""
        # Create display frame
        display = frame.copy()
        
        # Draw contour and bounding box if detected
        if contour is not None:
            cv2.drawContours(display, [contour], -1, (0, 255, 0), 2)
            
            if bbox is not None:
                x, y, w, h = bbox
                cv2.rectangle(display, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Add gesture text
        text = f"Gesture: {gesture} ({confidence:.1f}%)"
        cv2.putText(display, text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Add instructions
        cv2.putText(display, "Press Q to quit | T for template info | R to reset background",
                   (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Convert mask to color for display
        mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        # Stack side by side
        combined = np.hstack([display, mask_color])
        
        cv2.imshow('Hand Gesture Recognition (Webcam)', combined)
    
    def run(self):
        """Main recognition loop"""
        print("\n" + "="*60)
        print("Starting Real-Time Gesture Recognition (Webcam)")
        print("="*60)
        print(f"\nLoaded gestures: {list(self.templates.keys())}")
        print("\nPlace hand in front of webcam")
        print("Press Q to quit, T for template info, R to reset background")
        print("="*60 + "\n")
        
        if not self.initialize_camera():
            print("ERROR: Camera initialization failed!")
            return
        
        try:
            # Capture initial frames for background model
            print("Initializing background model... (keep hand out of frame)")
            for i in range(30):
                ret, frame = self.cap.read()
                if ret:
                    self.bg_subtractor.apply(frame)
                time.sleep(0.1)
            print("✓ Background model ready\n")
            
            while True:
                # Read frame
                ret, frame = self.cap.read()
                
                if not ret:
                    print("Failed to grab frame")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Process
                mask = self.preprocess_frame(frame)
                contour, bbox = self.extract_hand_contour(mask)
                features = self.extract_features(contour, bbox)
                gesture, confidence = self.match_template(features)
                gesture = self.smooth_prediction(gesture)
                
                # Display
                self.visualize(frame, mask, contour, bbox, gesture, confidence)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:  # Q or ESC
                    break
                
                elif key == ord('t'):
                    print("\n" + "="*40)
                    print("Loaded Templates:")
                    for name, template in self.templates.items():
                        print(f"  {name}: {template}")
                    print("="*40 + "\n")
                
                elif key == ord('r'):
                    print("Resetting background model...")
                    self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                        history=500, varThreshold=16, detectShadows=True
                    )
                    print("✓ Background reset")
        
        except KeyboardInterrupt:
            print("\nInterrupted")
        
        finally:
            if self.cap is not None:
                self.cap.release()
            cv2.destroyAllWindows()
            print("✓ Shutdown complete")


def verify_camera(camera_id=0):
    """Verify webcam is available"""
    print("\n" + "="*60)
    print("STEP 1: Webcam Verification")
    print("="*60)
    
    print(f"\nChecking camera {camera_id}...")
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"✗ Camera {camera_id} not available")
        print("\nTroubleshooting:")
        print("  1. Check if another application is using the camera")
        print("  2. Try a different camera_id (0, 1, 2, etc.)")
        print("  3. Check camera permissions")
        cap.release()
        return False
    
    ret, frame = cap.read()
    
    if not ret:
        print("✗ Cannot read from camera")
        cap.release()
        return False
    
    print(f"✓ Camera available!")
    print(f"  Frame shape: {frame.shape}")
    print(f"  Resolution: {frame.shape[1]}x{frame.shape[0]}")
    
    cap.release()
    return True


def main():
    """Main application entry point"""
    print("\n" + "="*60)
    print("HAND GESTURE RECOGNITION SYSTEM")
    print("MSR Gesture3D Dataset (Training) + Webcam (Recognition)")
    print("="*60)
    
    # Step 1: Verify webcam
    camera_ok = verify_camera()
    
    if not camera_ok:
        print("\nWebcam verification failed!")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Step 2: Train templates from dataset
    print("\n" + "="*60)
    print("STEP 2: Loading MSR Gesture3D Dataset")
    print("="*60)
    
    dataset_handler = MSRGesture3DHandler("./MSRGesture3D")
    
    # Check if templates already trained
    if os.path.exists('trained_templates.pkl'):
        print("\nFound existing trained_templates.pkl")
        response = input("Use existing templates? (y/n): ")
        
        if response.lower() == 'y':
            with open('trained_templates.pkl', 'rb') as f:
                templates = pickle.load(f)
            print(f"✓ Loaded {len(templates)} existing templates")
        else:
            templates = dataset_handler.train_templates_from_dataset()
    else:
        print("\nNo existing templates found")
        templates = dataset_handler.train_templates_from_dataset()
    
    if not templates:
        print("\nWARNING: No templates loaded!")
        print("Will use default templates")
    
    # Step 3: Start real-time recognition
    print("\n" + "="*60)
    print("STEP 3: Starting Real-Time Recognition")
    print("="*60)
    
    response = input("\nReady to start? (y/n): ")
    if response.lower() != 'y':
        print("Exiting...")
        return
    
    recognizer = WebcamGestureRecognizer(camera_id=0)
    recognizer.load_templates(templates)
    recognizer.run()
    
    print("\n" + "="*60)
    print("Session Complete")
    print("="*60)


if __name__ == "__main__":
    main()