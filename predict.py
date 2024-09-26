import cv2
from ultralytics import YOLO

# Load the trained YOLO model
model = YOLO(r'C:\Users\user\Desktop\cv\yolov8n.pt')  # Use your trained model's weight file

# Open the video file or webcam (use 0 for webcam)
cap = cv2.VideoCapture(r'C:\Users\user\Desktop\cv\4644521-uhd_2562_1440_30fps.mp4')

# Set up video writer to save output video with detections
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_with_detections.mp4', fourcc, 20.0, (640, 480))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame to match the YOLO input size (optional)
    resized_frame = cv2.resize(frame, (640, 640))

    # Run YOLO inference on the frame
    results = model(resized_frame)

    # Convert results to a format for OpenCV drawing
    annotated_frame = results[0].plot()  # Annotate the frame with bounding boxes

    # Display the frame with detections
    cv2.imshow('YOLOv8 Detection', annotated_frame)
    out.write(annotated_frame)

    # Break if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and writer objects
cap.release()
out.release()
cv2.destroyAllWindows()
