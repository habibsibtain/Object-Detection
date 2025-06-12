from ultralytics import YOLO
import cv2
import numpy 
import matplotlib.pyplot as plt

model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture('test_video.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    result = model(frame)
    annomated_frame = result[0].plot()

    cv2.imshow('YOLO Detection', annomated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 