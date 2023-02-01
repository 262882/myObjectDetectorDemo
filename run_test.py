#!/usr/bin/env python3
"""Start object detector test"""

import cv2
import time
import sys

print("Welcome to the Object Detector Tester")
run_time = time.perf_counter()

print("Initialising detector")
faceDetector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

print("Load video stream")
cap = cv2.VideoCapture(sys.argv[1])
font = cv2.FONT_HERSHEY_DUPLEX
color = (0, 0, 0) 
 
# Read until video is completed
while(cap.isOpened()):

  # Capture frame-by-frame
  ret, frame = cap.read()

  if ret == True:
    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    scale_factor = 1.1  # how much the image size is reduced at each image scale
    min_neighbours = 10 # how many neighbors each candidate rectangle should have to retain it
    detections = faceDetector.detectMultiScale(frameGray, scale_factor, min_neighbours)

    # Record performance
    rate_fps = 1/(time.perf_counter()-run_time)
    run_time = time.perf_counter()
    frame = cv2.putText(frame, str(round(rate_fps))+" FPS", ((frame.shape[1]//10)*7,(frame.shape[0]//10)*9), font, 1, color)

    # Add bounding boxes
    for (x, y, w, h) in detections:
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

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
