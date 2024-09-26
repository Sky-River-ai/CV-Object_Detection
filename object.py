
from ultralytics import YOLO

# Load the YOLOv8 model - you can choose 'yolov8n', 'yolov8s', 'yolov8m', etc.
model = YOLO('yolov8n.pt')  # Load a pretrained model

# Train the model on the custom dataset
model.train(data=r'C:\Users\user\Desktop\cv\data.yaml', epochs=50, imgsz=640, batch=16)
