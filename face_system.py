import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime

# Relaxed thresholds for real-world use
BLUR_THRESHOLD = 40
DARK_THRESHOLD = 35
BRIGHT_THRESHOLD = 220


class FaceSystem:

    def __init__(self):
        self.breach_dir = "images/unidentified_logs"
        os.makedirs(self.breach_dir, exist_ok=True)
        
        # For liveness detection
        self.prev_face_location = None
        self.movement_detected = False

    def log_breach(self, frame):
        """Log unidentified face with timestamp"""
        name = f"BREACH_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        path = os.path.join(self.breach_dir, name)
        cv2.imwrite(path, frame)
        return path

    def check_liveness(self, face_location):
        """Simple liveness detection based on movement"""
        if self.prev_face_location is None:
            self.prev_face_location = face_location
            return "Checking liveness..."
        
        # Calculate movement
        prev_top, prev_right, prev_bottom, prev_left = self.prev_face_location
        top, right, bottom, left = face_location
        
        movement = abs(top - prev_top) + abs(left - prev_left)
        
        self.prev_face_location = face_location
        
        if movement > 5:  # Threshold for movement
            self.movement_detected = True
            return "Face OK"
        else:
            if not self.movement_detected:
                return "Move slightly to verify"
            return "Face OK"

    def process(self, frame):
        """
        Process frame and return encoding, status, and face locations
        Returns: (encoding, status_message, face_locations)
        """
        if frame is None:
            return None, "No Frame", []

        # Resize for speed
        small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        # Lighting check
        brightness = gray.mean()
        if brightness < DARK_THRESHOLD:
            return None, "Too Dark (Increase light)", []
        
        if brightness > BRIGHT_THRESHOLD:
            return None, "Too Bright (Reduce light)", []

        # Blur check
        blur_val = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur_val < BLUR_THRESHOLD:
            return None, "Too Blurry (Hold still)", []

        # Detect faces
        faces = face_recognition.face_locations(rgb, model="hog")

        if len(faces) == 0:
            self.prev_face_location = None
            self.movement_detected = False
            return None, "No Face Detected", []

        if len(faces) > 1:
            return None, "Multiple Faces (Only one person)", faces

        # Pick the largest face (closest one)
        def face_area(face):
            top, right, bottom, left = face
            return (bottom - top) * (right - left)

        largest_face = max(faces, key=face_area)
        
        # Check if face is too small (too far)
        area = face_area(largest_face)
        if area < 5000:  # Minimum face size
            return None, "Come Closer", faces
        
        # Check if face is too large (too close)
        if area > 40000:
            return None, "Move Back", faces

        # Liveness detection
        liveness_status = self.check_liveness(largest_face)
        if liveness_status != "Face OK":
            return None, liveness_status, faces

        # Get encoding
        encodings = face_recognition.face_encodings(rgb, [largest_face])
        if not encodings:
            return None, "Face Encoding Failed", faces

        return encodings[0], "Face OK", faces
