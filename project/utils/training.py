import numpy as np
import tensorflow as tf 
def evaluate_model(self, test_data, test_labels):
    """Evaluate model on test data with proper shape handling"""
    if self.model is None:
        print("No model available for evaluation")
        return
    
    try:
        # Ensure test_data has the correct shape
        print(f"Test data shape before reshape: {test_data.shape}")
        print(f"Test labels shape before reshape: {test_labels.shape}")
        
        # Reshape test_data if needed (should be 4D for CNN)
        if len(test_data.shape) == 3:
            # Add channel dimension if missing
            test_data = np.expand_dims(test_data, axis=-1)
        elif len(test_data.shape) != 4:
            raise ValueError(f"Invalid test data shape: {test_data.shape}")
        
        # Ensure test_labels are properly shaped for binary classification
        if len(test_labels.shape) > 1:
            test_labels = test_labels.flatten()
        
        print(f"Test data shape after reshape: {test_data.shape}")
        print(f"Test labels shape after reshape: {test_labels.shape}")
        
        # Convert to float32 to ensure compatibility
        test_data = test_data.astype(np.float32)
        test_labels = test_labels.astype(np.float32)
        
        # Evaluate the model
        test_loss, test_accuracy = self.model.evaluate(
            test_data, test_labels, 
            verbose=1,
            batch_size=32
        )
        
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        
        return test_accuracy, test_loss
        
    except Exception as e:
        print(f"Error during model evaluation: {e}")
        print(f"Test data shape: {test_data.shape if test_data is not None else 'None'}")
        print(f"Test labels shape: {test_labels.shape if test_labels is not None else 'None'}")
        return None, None
