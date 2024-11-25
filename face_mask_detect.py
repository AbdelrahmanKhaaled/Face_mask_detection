# from ultralytics import YOLO
# import cv2

# model_6 = YOLO('model_manually_6.pt')

# results_6 = model_6("dataset/face_1.png")

# cap = cv2.VideoCapture(0)

# while True:
#     _, frame = cap.read()

#     results_6 = model_6(frame)

#     # cv2.imshow('Real-Time Face Detection', results_6[0])
#     results_6[0].show()

#     # Break the loop if 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()
# results_6[0].show()

from ultralytics import YOLO
import cv2

model = YOLO('model_manually_6.pt')

# Start capturing from webcam or a video file
cap = cv2.VideoCapture(0)  # Use 0 for webcam or replace with video path

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to the required format
    results = model.predict(source=frame, show=True, conf=0.4, classes=[1])  # Display results

    # Optional: Access detections for further processing
    # for result in results[0].boxes:
    #     bbox = result.xyxy[0]  # Bounding box coordinates [x_min, y_min, x_max, y_max]
    #     confidence = result.conf  # Confidence score
    #     class_id = result.cls  # Class ID
    #     print(f"Detected {model.names[int(class_id)]} with confidence {confidence}")

    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
