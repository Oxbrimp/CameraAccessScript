import cv2
from ultralytics import YOLO

# Load a pre-trained YOLOv8 model
model = YOLO("yolov8n.pt")  # 'n' is nano – fastest and lightest

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run object detection
    results = model(frame)

    # Visualise the results on the frame
    annotated_frame = results[0].plot()

    # Display the frame
    cv2.imshow("YOLOv8 Detection", annotated_frame)

    # To exit please just press 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release everything...
cap.release()
cv2.destroyAllWindows()
