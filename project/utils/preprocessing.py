import numpy as np
from sklearn.model_selection import train_test_split
def split_dataset(self, images, labels, test_size=0.2, val_size=0.2):
    """Split dataset with proper shape validation"""
    # Ensure images have correct shape (N, H, W, C)
    if len(images.shape) == 3:
        # Add channel dimension if missing
        images = np.expand_dims(images, axis=-1)
    
    print(f"Images shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Ensure labels are 1D
    if len(labels.shape) > 1:
        labels = labels.flatten()
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        images, labels, test_size=test_size, random_state=42, stratify=labels
    )
    
    # Second split: separate train and validation
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
    )
    
    # Validate shapes
    print(f"Training set: {X_train.shape} images, {y_train.shape} labels")
    print(f"Validation set: {X_val.shape} images, {y_val.shape} labels")
    print(f"Test set: {X_test.shape} images, {y_test.shape} labels")
    
    return X_train, X_val, X_test, y_train, y_val, y_test
