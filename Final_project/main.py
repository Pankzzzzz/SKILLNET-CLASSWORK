import cv2
import numpy as np
from ultralytics import YOLO

# Load a pre-trained YOLOv8 model (you can fine-tune it with your dataset)
model = YOLO('yolov8n.pt')  # Use 'yolov8n.pt' for nano model; replace with your trained model

def detect_defects(image_path):
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found or unable to load.")
    
    # Preprocess: Resize and normalize
    img_resized = cv2.resize(img, (640, 640))  # YOLOv8 input size
    img_normalized = img_resized / 255.0  # Normalize to [0,1]
    
    # Run YOLO inference for defect detection
    results = model(img_normalized)  # Model predicts defects like cracks, breaks
    
    # Post-process results
    defects = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            conf = box.conf[0]  # Confidence score
            cls = int(box.cls[0])  # Class (e.g., 0: crack, 1: break)
            if conf > 0.5:  # Threshold for detection
                defects.append({
                    'class': cls,
                    'confidence': conf,
                    'bbox': (x1, y1, x2, y2)
                })
                # Draw bounding box on image
                cv2.rectangle(img_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img_resized, f'Defect {cls} ({conf:.2f})', (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Save or display the output
    cv2.imwrite('output_defects.jpg', img_resized)
    return defects

# Example usage
image_path = 'path/to/your/tablet_image.jpg'  # Replace with your image path
detected_defects = detect_defects(image_path)
print("Detected Defects:", detected_defects)