import cv2
import numpy as np
import time
import os
from datetime import datetime

class InspectionCamera:
    def __init__(self, camera_id=0, resolution=(1920, 1080)):
        """Initialize camera for inspection system"""
        self.camera_id = camera_id
        self.resolution = resolution
        self.cap = None
        self.is_initialized = False
        
    def initialize_camera(self):
        """Initialize and configure camera settings"""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            
            if not self.cap.isOpened():
                raise Exception(f"Could not open camera {self.camera_id}")
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print(f"Camera initialized: {actual_width}x{actual_height}")
            self.is_initialized = True
            return True
            
        except Exception as e:
            print(f"Error initializing camera: {e}")
            return False
    
    def capture_image(self, save_path=None):
        """Capture a single image from camera"""
        if not self.is_initialized:
            print("Camera not initialized")
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to capture image")
            return None
        
        if save_path:
            cv2.imwrite(save_path, frame)
            print(f"Image saved to: {save_path}")
        
        return frame
    
    def capture_dataset(self, save_directory, num_images, prefix="image"):
        """Capture multiple images for dataset"""
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        
        print(f"Capturing {num_images} images. Press SPACE to capture, ESC to stop")
        count = 0
        
        while count < num_images:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            cv2.imshow('Dataset Capture', frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == 32:  # SPACE
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{prefix}_{count+1:03d}_{timestamp}.jpg"
                filepath = os.path.join(save_directory, filename)
                
                cv2.imwrite(filepath, frame)
                count += 1
                print(f"Captured {count}/{num_images}")
                time.sleep(0.5)
                
            elif key == 27:  # ESC
                break
        
        cv2.destroyAllWindows()
        print(f"Dataset capture complete: {count} images")
    
    def preprocess_for_inference(self, image, target_size=(224, 224)):
        """Preprocess image for CNN inference"""
        resized = cv2.resize(image, target_size)
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb_image.astype(np.float32) / 255.0
        batched = np.expand_dims(normalized, axis=0)
        return batched
    
    def release_camera(self):
        """Release camera resources"""
        if self.cap:
            self.cap.release()
            cv2.destroyAllWindows()
