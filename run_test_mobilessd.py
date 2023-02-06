#!/usr/bin/env python3
"""Start object detector test"""

import time
import sys
import cv2
import numpy as np

print("Welcome to the Object Detector Tester")
run_time = time.perf_counter()

print("Initialising detector")
detector = cv2.dnn.readNetFromCaffe('net/MobileNetSSD_deploy.prototxt', 'net/MobileNetSSD_deploy.caffemodel')
#cv2.dnn.readNetFromDarknet('./net/yolov3-tiny.cfg', './net/yolov3-tiny.weights')
detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
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
                                      size = (300, 300), # Resolution multiple of 32
                                      swapRB=True, crop=True)
        detector.setInput(blob)
        layer_output = detector.forward()

          # Record performance
        rate_fps = 1/(time.perf_counter()-run_time)
        run_time = time.perf_counter()
        frame = cv2.putText(frame, str(round(rate_fps))+" FPS",
                           ((frame.shape[1]//10)*7,(frame.shape[0]//10)*9), font, 1, color)

        #Size of frame resize (300x300)
        cols = 300
        rows = 300

        for detection in layer_output[0,0]:
            confidence = detection[2] 
            class_id = int(detection[1])
            if confidence > 0.2 and class_id == 15:  # Threshold       
                 
                # Object location 
                xLeftBottom = int(detection[3] * cols) 
                yLeftBottom = int(detection[4] * rows)
                xRightTop   = int(detection[5] * cols)
                yRightTop   = int(detection[6] * rows)
                
                # Factor for scale to original size of frame
                heightFactor = frame.shape[0]/rows
                widthFactor = frame.shape[1]/cols
                # Scale object detection to frame
                xLeftBottom = int(widthFactor * xLeftBottom) 
                yLeftBottom = int(heightFactor * yLeftBottom)
                xRightTop   = int(widthFactor * xRightTop)
                yRightTop   = int(heightFactor * yRightTop)
                # Draw location of object  
                frame = cv2.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),
                              (0, 255, 0),3)

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
