import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self, input_size=(224, 224)):
        self.input_size = input_size
        
    def load_dataset(self, good_dir, bad_dir):
        """Load and prepare dataset from directories"""
        images = []
        labels = []
        
        # Load good images (label = 0)
        print("Loading good images...")
        for filename in os.listdir(good_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(good_dir, filename)
                image = cv2.imread(img_path)
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = cv2.resize(image, self.input_size)
                    images.append(image)
                    labels.append(0)  # Good = 0
        
        # Load bad images (label = 1)
        print("Loading bad images...")
        for filename in os.listdir(bad_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(bad_dir, filename)
                image = cv2.imread(img_path)
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = cv2.resize(image, self.input_size)
                    images.append(image)
                    labels.append(1)  # Bad = 1
        
        images = np.array(images, dtype=np.float32) / 255.0
        labels = np.array(labels)
        
        print(f"Dataset loaded: {len(images)} images")
        print(f"Good samples: {np.sum(labels == 0)}")
        print(f"Bad samples: {np.sum(labels == 1)}")
        
        return images, labels
    
    def split_dataset(self, images, labels, test_size=0.2, val_size=0.2):
        """Split dataset into train, validation, and test sets"""
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            images, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        # Second split: separate train and validation
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
        )
        
        print(f"Training set: {len(X_train)} images")
        print(f"Validation set: {len(X_val)} images")
        print(f"Test set: {len(X_test)} images")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def create_data_generators(self, X_train, y_train, X_val, y_val, batch_size=32):
        """Create data generators with augmentation"""
        # Ensure labels are the correct shape for binary classification
        y_train = y_train.reshape(-1, 1)
        y_val = y_val.reshape(-1, 1)

        train_datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1,
            brightness_range=[0.8, 1.2]
        )
        
        val_datagen = ImageDataGenerator()
        
        train_generator = train_datagen.flow(
            X_train, y_train, batch_size=batch_size, shuffle=True
        )
        
        val_generator = val_datagen.flow(
            X_val, y_val, batch_size=batch_size, shuffle=False
        )
        
        return train_generator, val_generator
