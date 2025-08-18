from flask import Flask, render_template, request, jsonify, Response
import tensorflow as tf
import numpy as np
import cv2
import os
import base64
import traceback
import time
import threading
from datetime import datetime
from io import BytesIO
import logging

# Import your existing utilities
try:
    from utils.camera import InspectionCamera
    from utils.preprocessing import DataPreprocessor
    from config import Config
except ImportError as e:
    print(f"Warning: Could not import utilities: {e}")
    print("Make sure utils/camera.py, utils/preprocessing.py, and config.py exist")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

class WebInspectionSystem:
    def __init__(self):
        self.model = None
        self.model_loaded = False
        self.camera = None
        self.camera_initialized = False
        self.load_model()
    
    def load_model(self):
        """Load the trained CNN model with error handling"""
        try:
            model_path = os.path.join(Config.MODEL_DIR, 'best_model.h5')
            
            if not os.path.exists(model_path):
                logger.warning(f"Model not found at {model_path}")
                return False
            
            self.model = tf.keras.models.load_model(model_path)
            self.model_loaded = True
            logger.info(f"Model loaded successfully from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model_loaded = False
            return False
    
    def initialize_camera(self):
        """Initialize camera with error handling"""
        try:
            if self.camera is None:
                self.camera = InspectionCamera(Config.CAMERA_ID, Config.CAMERA_RESOLUTION)
            
            if not self.camera_initialized:
                self.camera_initialized = self.camera.initialize_camera()
            
            return self.camera_initialized
            
        except Exception as e:
            logger.error(f"Error initializing camera: {e}")
            self.camera_initialized = False
            return False
    
    def preprocess_image(self, image_data):
        """Preprocess uploaded image for inference with error handling"""
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_data, np.uint8)
            
            # Decode image
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("Could not decode image")
            
            # Resize to model input size
            resized = cv2.resize(image, Config.INPUT_SIZE)
            
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            # Normalize pixel values
            normalized = rgb_image.astype(np.float32) / 255.0
            
            # Add batch dimension
            batched = np.expand_dims(normalized, axis=0)
            
            return batched
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return None
    
    def inspect_image(self, preprocessed_image):
        """Perform inspection on preprocessed image with error handling"""
        try:
            if not self.model_loaded:
                return {
                    'success': False,
                    'error': 'Model not loaded',
                    'prediction': 0,
                    'confidence': 0.0
                }
            
            # Make prediction
            prediction_prob = self.model.predict(preprocessed_image, verbose=0)[0][0]
            
            # Convert to binary prediction
            binary_prediction = 1 if prediction_prob > Config.CONFIDENCE_THRESHOLD else 0
            
            # Determine status
            status = "DEFECTIVE" if binary_prediction == 1 else "GOOD"
            
            return {
                'success': True,
                'prediction': int(binary_prediction),
                'confidence': float(prediction_prob),
                'status': status,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
        except Exception as e:
            logger.error(f"Error during inspection: {e}")
            return {
                'success': False,
                'error': str(e),
                'prediction': 0,
                'confidence': 0.0
            }

# Initialize the inspection system
try:
    inspection_system = WebInspectionSystem()
except Exception as e:
    logger.error(f"Failed to initialize inspection system: {e}")
    inspection_system = None

# Error handlers - MUST return JSON
@app.errorhandler(404)
def not_found_error(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'status': 404
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error',
        'status': 500
    }), 500

@app.errorhandler(400)
def bad_request_error(error):
    return jsonify({
        'success': False,
        'error': 'Bad request',
        'status': 400
    }), 400

@app.errorhandler(413)
def file_too_large_error(error):
    return jsonify({
        'success': False,
        'error': 'File too large. Maximum size is 16MB.',
        'status': 413
    }), 413

# Main routes
@app.route('/')
def index():
    """Serve the main HTML page"""
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error serving index page: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to load main page'
        }), 500

@app.route('/status')
def status():
    """Check system status - Always returns JSON"""
    try:
        if inspection_system is None:
            return jsonify({
                'success': False,
                'error': 'Inspection system not initialized',
                'model_loaded': False,
                'camera_initialized': False
            })
        
        return jsonify({
            'success': True,
            'model_loaded': inspection_system.model_loaded,
            'camera_initialized': inspection_system.camera_initialized,
            'model_path': os.path.join(Config.MODEL_DIR, 'best_model.h5'),
            'confidence_threshold': Config.CONFIDENCE_THRESHOLD,
            'input_size': Config.INPUT_SIZE
        })
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/inspect', methods=['POST'])
def inspect():
    """Handle image inspection requests - Always returns JSON"""
    try:
        # Check if inspection system is available
        if inspection_system is None:
            return jsonify({
                'success': False,
                'error': 'Inspection system not initialized'
            }), 500
        
        # Check if model is loaded
        if not inspection_system.model_loaded:
            return jsonify({
                'success': False,
                'error': 'Model not loaded. Please train the model first.'
            }), 500
        
        # Check if file was uploaded
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image file provided'
            }), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        # Validate file type
        allowed_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
        file_ext = os.path.splitext(file.filename.lower())[1]
        
        if file_ext not in allowed_extensions:
            return jsonify({
                'success': False,
                'error': 'Invalid file type. Please upload an image file.'
            }), 400
        
        # Read and preprocess image
        image_data = file.read()
        
        if len(image_data) == 0:
            return jsonify({
                'success': False,
                'error': 'Empty file uploaded'
            }), 400
        
        preprocessed_image = inspection_system.preprocess_image(image_data)
        
        if preprocessed_image is None:
            return jsonify({
                'success': False,
                'error': 'Failed to preprocess image'
            }), 400
        
        # Perform inspection
        result = inspection_system.inspect_image(preprocessed_image)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in inspect endpoint: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': f'Inspection failed: {str(e)}'
        }), 500

@app.route('/batch-inspect', methods=['POST'])
def batch_inspect():
    """Handle multiple image inspection - Always returns JSON"""
    try:
        if inspection_system is None or not inspection_system.model_loaded:
            return jsonify({
                'success': False,
                'error': 'Model not loaded'
            }), 500
        
        files = request.files.getlist('images')
        
        if not files or len(files) == 0:
            return jsonify({
                'success': False,
                'error': 'No images provided'
            }), 400
        
        results = []
        
        for i, file in enumerate(files):
            if file.filename == '':
                continue
            
            try:
                image_data = file.read()
                
                if len(image_data) == 0:
                    results.append({
                        'filename': file.filename,
                        'index': i,
                        'success': False,
                        'error': 'Empty file'
                    })
                    continue
                
                preprocessed = inspection_system.preprocess_image(image_data)
                
                if preprocessed is not None:
                    result = inspection_system.inspect_image(preprocessed)
                    result['filename'] = file.filename
                    result['index'] = i
                    results.append(result)
                else:
                    results.append({
                        'filename': file.filename,
                        'index': i,
                        'success': False,
                        'error': 'Failed to preprocess image'
                    })
                    
            except Exception as e:
                results.append({
                    'filename': file.filename,
                    'index': i,
                    'success': False,
                    'error': str(e)
                })
        
        return jsonify({
            'success': True,
            'results': results,
            'total_processed': len(results)
        })
        
    except Exception as e:
        logger.error(f"Error in batch inspect: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': f'Batch processing failed: {str(e)}'
        }), 500

@app.route('/camera-capture', methods=['POST'])
def camera_capture():
    """Capture image from camera and inspect - Always returns JSON"""
    try:
        if inspection_system is None:
            return jsonify({
                'success': False,
                'error': 'Inspection system not initialized'
            }), 500
        
        if not inspection_system.model_loaded:
            return jsonify({
                'success': False,
                'error': 'Model not loaded'
            }), 500
        
        # Initialize camera if needed
        if not inspection_system.initialize_camera():
            return jsonify({
                'success': False,
                'error': 'Failed to initialize camera'
            }), 500
        
        # Capture image
        image = inspection_system.camera.capture_image()
        
        if image is None:
            return jsonify({
                'success': False,
                'error': 'Failed to capture image from camera'
            }), 500
        
        # Preprocess for inference
        preprocessed = inspection_system.camera.preprocess_for_inference(
            image, Config.INPUT_SIZE
        )
        
        # Perform inspection
        result = inspection_system.inspect_image(preprocessed)
        
        if not result.get('success', False):
            return jsonify(result), 500
        
        # Convert image to base64 for web display
        try:
            _, buffer = cv2.imencode('.jpg', image)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            result['image'] = f"data:image/jpeg;base64,{image_base64}"
        except Exception as e:
            logger.warning(f"Could not encode image to base64: {e}")
            # Don't fail the whole request if image encoding fails
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in camera capture: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': f'Camera capture failed: {str(e)}'
        }), 500

@app.route('/training-info')
def training_info():
    """Get information about training data and model - Always returns JSON"""
    try:
        info = {
            'success': True,
            'model_exists': False,
            'good_data_count': 0,
            'bad_data_count': 0,
            'training_history_exists': False,
            'total_data': 0,
            'data_ready': False
        }
        
        # Check if model exists
        model_path = os.path.join(Config.MODEL_DIR, 'best_model.h5')
        info['model_exists'] = os.path.exists(model_path)
        
        # Count training data
        try:
            if os.path.exists(Config.GOOD_DATA_DIR):
                info['good_data_count'] = len([f for f in os.listdir(Config.GOOD_DATA_DIR) 
                                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        except Exception as e:
            logger.warning(f"Could not count good data: {e}")
        
        try:
            if os.path.exists(Config.BAD_DATA_DIR):
                info['bad_data_count'] = len([f for f in os.listdir(Config.BAD_DATA_DIR) 
                                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        except Exception as e:
            logger.warning(f"Could not count bad data: {e}")
        
        # Check training history
        history_path = os.path.join(Config.MODEL_DIR, 'training_history.png')
        info['training_history_exists'] = os.path.exists(history_path)
        
        info['total_data'] = info['good_data_count'] + info['bad_data_count']
        info['data_ready'] = info['total_data'] >= 20  # Minimum viable dataset
        
        return jsonify(info)
        
    except Exception as e:
        logger.error(f"Error getting training info: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/reload-model', methods=['POST'])
def reload_model():
    """Reload the model - Always returns JSON"""
    try:
        if inspection_system is None:
            return jsonify({
                'success': False,
                'error': 'Inspection system not initialized'
            }), 500
        
        success = inspection_system.load_model()
        
        return jsonify({
            'success': success,
            'model_loaded': inspection_system.model_loaded,
            'message': 'Model reloaded successfully' if success else 'Failed to reload model'
        })
        
    except Exception as e:
        logger.error(f"Error reloading model: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Health check endpoint
@app.route('/health')
def health():
    """Health check endpoint - Always returns JSON"""
    return jsonify({
        'success': True,
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'system_ready': inspection_system is not None
    })

# Create necessary directories on startup
def create_directories():
    """Create necessary directories"""
    try:
        Config.create_directories()
        logger.info("Directories created successfully")
    except Exception as e:
        logger.error(f"Error creating directories: {e}")

if __name__ == '__main__':
    print("=== Computer Vision Inspection Web System ===")
    
    # Create directories
    create_directories()
    
    # System status
    if inspection_system is None:
        print("❌ ERROR: Inspection system failed to initialize")
    else:
        print(f"📂 Model path: {os.path.join(Config.MODEL_DIR, 'best_model.h5')}")
        print(f"🤖 Model loaded: {inspection_system.model_loaded}")
        
        if not inspection_system.model_loaded:
            print("\n⚠️  WARNING: Model not found!")
            print("Please train your model first:")
            print("python main.py --mode train")
            print("\nThe web interface will still run, but inspections will fail until model is trained.")
    
    print(f"\n🚀 Starting web server...")
    print(f"📊 Web interface: http://127.0.0.1:5000")
    print(f"📈 System status: http://127.0.0.1:5000/status")
    print(f"🔍 Health check: http://127.0.0.1:5000/health")
    print(f"📚 Training info: http://127.0.0.1:5000/training-info")
    
    # Run Flask app
    try:
        app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
    except Exception as e:
        logger.error(f"Failed to start Flask app: {e}")
        print(f"❌ Server failed to start: {e}")
