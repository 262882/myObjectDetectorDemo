#!/usr/bin/env python3
"""Start object detector test"""

import time
import sys
import cv2
import numpy as np

print("Welcome to the Object Detector Tester")
run_time = time.perf_counter()

print("Initialising detector")
detector = cv2.dnn.readNetFromDarknet('./net/yolov3-tiny.cfg', './net/yolov3-tiny.weights')
detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
ln = detector.getLayerNames()  # layer names
ln = [ln[i - 1] for i in detector.getUnconnectedOutLayers()]  # layers: ['yolo_16', 'yolo_23']
print("Detector ready")

print("Loading video stream")
cap = cv2.VideoCapture(sys.argv[1])
font = cv2.FONT_HERSHEY_DUPLEX
color = (0, 0, 0)
print("Loaded video stream")

# Read until video is completed
while cap.isOpened():

  # Capture frame-by-frame
    ret, frame = cap.read()

    if ret:
        blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255,
                                      size = (416, 416), # Resolution multiple of 32
                                      swapRB=True, crop=False)
        detector.setInput(blob)
        layer_output = detector.forward(ln)

          # Record performance
        rate_fps = 1/(time.perf_counter()-run_time)
        run_time = time.perf_counter()
        frame = cv2.putText(frame, str(round(rate_fps))+" FPS",
                           ((frame.shape[1]//10)*7,(frame.shape[0]//10)*9), font, 1, color)

        # Add bounding boxes https://thinkinfi.com/yolo-object-detection-using-python-opencv/
        positive_predict = False  # Only consider the first prediction made
        for predictions in layer_output:
            for prediction in predictions:
                box = prediction[:4]*np.array([frame.shape[1], frame.shape[0],
                                               frame.shape[1], frame.shape[0]])
                (x, y, w, h) = np.copy(box).astype("int")
                if prediction[5]>0.2: # Only consider predictions of people
                    frame = cv2.rectangle(frame, (x-w//2, y-h//2), (x + w//2, y + h//2),
                                          (0, 255, 0), 3)
                    positive_predict = True
                    break
            if positive_predict:
                break

        # Display the resulting frame
        cv2.imshow('Frame',frame)

        # Press Q on keyboard to  exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Break the loop
    else:
        break

cap.release() # When everything done, release the video capture object
cv2.destroyAllWindows() # Closes all the frames

print("Complete")
