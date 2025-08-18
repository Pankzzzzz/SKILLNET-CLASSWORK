import cv2
import numpy as np
import tensorflow as tf
from utils.camera import InspectionCamera
from utils.preprocessing import DataPreprocessor
from utils.training import CNNTrainer
from config import Config
import os
import argparse

class InspectionSystem:
    def __init__(self):
        self.camera = InspectionCamera(Config.CAMERA_ID, Config.CAMERA_RESOLUTION)
        self.model = None
        self.is_initialized = False
    
    def collect_data(self, num_good=100, num_bad=100):
        """Collect training data using camera"""
        print("=== Data Collection Mode ===")
        
        if not self.camera.initialize_camera():
            print("Failed to initialize camera")
            return
        
        Config.create_directories()
        
        print(f"Collecting {num_good} good samples...")
        input("Position good samples in view and press Enter...")
        self.camera.capture_dataset(Config.GOOD_DATA_DIR, num_good, "good")
        
        print(f"Collecting {num_bad} bad samples...")
        input("Position defective samples in view and press Enter...")
        self.camera.capture_dataset(Config.BAD_DATA_DIR, num_bad, "bad")
        
        self.camera.release_camera()
        print("Data collection completed!")
    
    def train_model(self):
        """Train the CNN model"""
        print("=== Model Training Mode ===")
        
        # Check if data exists
        if not os.path.exists(Config.GOOD_DATA_DIR) or not os.path.exists(Config.BAD_DATA_DIR):
            print("Training data not found. Please collect data first.")
            return
        
        # Load and preprocess data
        preprocessor = DataPreprocessor(Config.INPUT_SIZE)
        images, labels = preprocessor.load_dataset(Config.GOOD_DATA_DIR, Config.BAD_DATA_DIR)
        
        if len(images) < 10:
            print("Insufficient training data. Please collect more samples.")
            return
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_dataset(images, labels)
        
        # Create data generators
        train_gen, val_gen = preprocessor.create_data_generators(
            X_train, y_train, X_val, y_val, Config.BATCH_SIZE
        )
        
        # Create and train model
        trainer = CNNTrainer((*Config.INPUT_SIZE, 3))
        
        # Use transfer learning for better results with limited data
        trainer.create_transfer_learning_model()
        trainer.compile_model(Config.LEARNING_RATE)
        
        # Train model
        history = trainer.train_model(train_gen, val_gen, Config.EPOCHS, Config.MODEL_DIR)
        
        # Evaluate on test set
        trainer.evaluate_model(X_test, y_test)
        
        # Plot training history
        trainer.plot_training_history(os.path.join(Config.MODEL_DIR, 'training_history.png'))
        
        print("Model training completed!")
    
    def load_model(self, model_path=None):
        """Load trained model"""
        if model_path is None:
            model_path = os.path.join(Config.MODEL_DIR, 'best_model.h5')
        
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}")
            return False
        
        try:
            self.model = tf.keras.models.load_model(model_path)
            print(f"Model loaded from: {model_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def run_inspection(self):
        """Run real-time inspection"""
        print("=== Real-time Inspection Mode ===")
        
        # Load model
        if not self.load_model():
            print("Failed to load model. Please train model first.")
            return
        
        # Initialize camera
        if not self.camera.initialize_camera():
            print("Failed to initialize camera")
            return
        
        print("Starting real-time inspection...")
        print("Press 'q' to quit, 'SPACE' to inspect current frame")
        
        while True:
            ret, frame = self.camera.capture_image()
            
            if not ret:
                continue
            
            # Preprocess for display
            display_frame = frame.copy()
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == 32:  # SPACE bar
                # Perform inspection
                result = self.inspect_image(frame)
                
                # Display result on frame
                if result['prediction'] == 1:
                    color = (0, 0, 255)  # Red for defective
                    status = "DEFECTIVE"
                else:
                    color = (0, 255, 0)  # Green for good
                    status = "GOOD"
                
                confidence = result['confidence']
                
                # Add text overlay
                cv2.putText(display_frame, f"Status: {status}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(display_frame, f"Confidence: {confidence:.2f}", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                
                print(f"Inspection Result: {status} (Confidence: {confidence:.2f})")
            
            # Show instructions
            cv2.putText(display_frame, "Press SPACE to inspect, 'q' to quit", 
                       (10, display_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Inspection System', display_frame)
        
        self.camera.release_camera()
    
    def inspect_image(self, image):
        """Inspect a single image"""
        if self.model is None:
            return {'prediction': 0, 'confidence': 0.0}
        
        # Preprocess image
        processed = self.camera.preprocess_for_inference(image, Config.INPUT_SIZE)
        
        # Make prediction
        prediction = self.model.predict(processed, verbose=0)[0][0]
        
        # Convert to binary prediction
        binary_pred = 1 if prediction > Config.CONFIDENCE_THRESHOLD else 0
        
        return {
            'prediction': binary_pred,
            'confidence': prediction
        }

def main():
    parser = argparse.ArgumentParser(description='Computer Vision Inspection System')
    parser.add_argument('--mode', choices=['collect', 'train', 'inspect'], 
                       required=True, help='Operation mode')
    parser.add_argument('--good-samples', type=int, default=100, 
                       help='Number of good samples to collect')
    parser.add_argument('--bad-samples', type=int, default=100, 
                       help='Number of bad samples to collect')
    
    args = parser.parse_args()
    
    system = InspectionSystem()
    
    if args.mode == 'collect':
        system.collect_data(args.good_samples, args.bad_samples)
    elif args.mode == 'train':
        system.train_model()
    elif args.mode == 'inspect':
        system.run_inspection()

if __name__ == "__main__":
    # If no command line arguments, show interactive menu
    import sys
    if len(sys.argv) == 1:
        system = InspectionSystem()
        
        print("=== Computer Vision Inspection System ===")
        print("1. Collect Data")
        print("2. Train Model") 
        print("3. Run Inspection")
        
        choice = input("Select option (1-3): ")
        
        if choice == "1":
            system.collect_data()
        elif choice == "2":
            system.train_model()
        elif choice == "3":
            system.run_inspection()
        else:
            print("Invalid choice")
    else:
        main()
