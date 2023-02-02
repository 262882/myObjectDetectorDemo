#!/usr/bin/env python3
"""Start object detector test"""

import cv2
import time
import sys
import numpy as np

print()
print("Welcome to the Object Detector Tester")
run_time = time.perf_counter()

print("Initialising detector")
detector = cv2.dnn.readNetFromDarknet('./net/yolov3-tiny.cfg', './net/yolov3-tiny.weights')
detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
ln = detector.getLayerNames()  # layer names
ln = [ln[i - 1] for i in detector.getUnconnectedOutLayers()]  #YOLOv3 is ['yolo_16', 'yolo_23']
print("Detector ready")

# Run from GPU
#detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
#detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

print("Load video stream")
cap = cv2.VideoCapture(sys.argv[1])
font = cv2.FONT_HERSHEY_DUPLEX
color = (0, 0, 0) 
 
# Read until video is completed
while(cap.isOpened()):

  # Capture frame-by-frame
  ret, frame = cap.read()

  if ret == True:
    blob = cv2.dnn.blobFromImage(frame, 1/255.0,
                                (160, 160), # Resolution multiple of 32
                                swapRB=True, crop=False) 
    detector.setInput(blob)
    layer_output = detector.forward(ln)

    # Record performance
    rate_fps = 1/(time.perf_counter()-run_time)
    run_time = time.perf_counter()
    frame = cv2.putText(frame, str(round(rate_fps))+" FPS", ((frame.shape[1]//10)*7,(frame.shape[0]//10)*9), font, 1, color)

    # Add bounding boxes https://thinkinfi.com/yolo-object-detection-using-python-opencv/
    for predictions in layer_output:
      for prediction in predictions:
          box = prediction[:4]*np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
          (x, y, w, h) = box.astype("int")
          print(prediction)
          #if prediction[7]>0.1:
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
