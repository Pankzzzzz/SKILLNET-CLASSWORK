import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D # type: ignore
from tensorflow.keras.applications import MobileNetV2 # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau # type: ignore
import matplotlib.pyplot as plt
import os

class CNNTrainer:
    def __init__(self, input_size=(224, 224, 3), num_classes=2):
        self.input_size = input_size
        self.num_classes = num_classes
        self.model = None
        self.history = None
    
    def create_simple_cnn(self):
        """Create a simple CNN model"""
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=self.input_size),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')  # Binary classification
        ])
        
        self.model = model
        return model
    
    def create_transfer_learning_model(self):
        """Create a transfer learning model using MobileNetV2"""
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_size
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(128, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        
        self.model = model
        return model
    
    def compile_model(self, learning_rate=0.001):
        """Compile the model"""
        if self.model is None:
            raise ValueError("Model not created. Call create_simple_cnn() or create_transfer_learning_model() first")
        
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        print("Model compiled successfully")
        print(f"Total parameters: {self.model.count_params():,}")
    
    def train_model(self, train_generator, val_generator, epochs=50, model_save_path='models/'):
        """Train the model"""
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=os.path.join(model_save_path, 'best_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=0.0001,
                verbose=1
            )
        ]
        
        print("Starting training...")
        self.history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model
        final_model_path = os.path.join(model_save_path, 'final_model.h5')
        self.model.save(final_model_path)
        print(f"Final model saved to: {final_model_path}")
        
        return self.history
    
    def plot_training_history(self, save_path=None):
        """Plot training history"""
        if self.history is None:
            print("No training history available")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Training plots saved to: {save_path}")
        
        plt.show()
    
    def evaluate_model(self, test_data, test_labels):
        """Evaluate model on test data"""
        if self.model is None:
            print("No model available for evaluation")
            return
        
        test_loss, test_accuracy = self.model.evaluate(test_data, test_labels, verbose=0)
        
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        
        return test_accuracy, test_loss
