import os

# Project configuration
class Config:
    # Paths
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
    GOOD_DATA_DIR = os.path.join(DATA_DIR, 'good')
    BAD_DATA_DIR = os.path.join(DATA_DIR, 'bad')
    MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
    
    # Camera settings
    CAMERA_ID = 0
    CAMERA_RESOLUTION = (1920, 1080)
    
    # Image processing
    INPUT_SIZE = (224, 224)
    BATCH_SIZE = 32
    
    # Training parameters
    EPOCHS = 50
    LEARNING_RATE = 0.001
    VALIDATION_SPLIT = 0.2
    
    # Model settings
    MODEL_NAME = 'inspection_model.h5'
    CONFIDENCE_THRESHOLD = 0.7
    
    # Create directories if they don't exist
    @classmethod
    def create_directories(cls):
        os.makedirs(cls.DATA_DIR, exist_ok=True)
        os.makedirs(cls.GOOD_DATA_DIR, exist_ok=True)
        os.makedirs(cls.BAD_DATA_DIR, exist_ok=True)
        os.makedirs(cls.MODEL_DIR, exist_ok=True)
