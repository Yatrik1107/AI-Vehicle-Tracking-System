import cvzone
from ultralytics import YOLO
import cv2
import math
import numpy as np
from sort import *

# Initialize video capture
cap = cv2.VideoCapture('../Videos/cars.mp4')
# cap = cv2.VideoCapture(0)

# Load mask and model
mask_image = cv2.imread('./mask-image.png')
model = YOLO('../YOLOWeights/yolov8l.pt')

# Initialize tracking
tracker = Sort(max_age=15, min_hits=3, iou_threshold=0.3)

# Vehicle classes
classes = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush'
]

# frame_resolution = (640, 480)
# Limits = [ 194, 200, 322, 200  ]
# frame_resolution_for_graphics_image = (280,70)
# total_count_position = (150, 45)
# totalCountFontSize = 5

# Frame resolution and limits for vehicle counting
frame_resolution = (1366, 768)
Limits = [415, 333, 688, 333]  # [x1, y1, x2, y2]
total_count_position = (320, 72)
totalCountFontSize = 7
frame_resolution_for_graphics_image = (599, 112)

totalCount = []

# Resize the mask image to match frame resolution
mask_image = cv2.resize(mask_image, frame_resolution)

if cap.isOpened():
    while True:
        success, img = cap.read()
        if not success:
            break

        img = cv2.resize(img, frame_resolution)

        # Load and overlay graphical image
        graphical_image = cv2.imread('./graphics.png', cv2.IMREAD_UNCHANGED)
        graphical_image = cv2.resize(graphical_image, frame_resolution_for_graphics_image)
        img = cvzone.overlayPNG(img, graphical_image, (0, 0))

        # Apply mask to the frame
        resized_img = cv2.bitwise_and(mask_image, img)

        detections = np.empty((0, 5))

        # Perform object detection
        results = model(resized_img, stream=True)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                vehicle_category = classes[cls]

                # Only consider specific vehicle classes with confidence > 0.4
                if vehicle_category in ['person', 'car', 'motorcycle', 'bus', 'truck'] and conf > 0.4:
                    temp_detected_array = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, temp_detected_array))

        # Update tracker with the new detections
        results_of_tracker = tracker.update(detections)

        for i in results_of_tracker:
            x1, y1, x2, y2, id = map(int, i)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
            cvzone.putTextRect(img, f"{id}", (max(x1, 35), max(y1 - 20, 35)), scale=2, thickness=3, offset=10)

            cx, cy = x1 + w // 2, y1 + h // 2
            cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            # Check if the object crosses the limits
            if (Limits[0] <= cx <= Limits[2]) and (Limits[1] - 70 <= cy <= Limits[1] + 70):
                if id not in totalCount:
                    totalCount.append(id)
                cv2.line(img, (Limits[0], Limits[1]), (Limits[2], Limits[3]), color=(0, 255, 0), thickness=3)

        # Display total count and draw the limit line
        cv2.putText(img, str(len(totalCount)), total_count_position, cv2.FONT_HERSHEY_PLAIN, totalCountFontSize, (50, 50, 255), 5)
        cv2.line(img, (Limits[0], Limits[1]), (Limits[2], Limits[3]), color=(0, 0, 255), thickness=1)

        # Show the processed frame
        cv2.imshow('Vehicle Counting', img)
        cv2.waitKey(1)

else:
    print('Error in opening the camera!')