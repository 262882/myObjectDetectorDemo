#!/usr/bin/env python3
"""Start object detector test"""

import time
import sys
import cv2
import numpy as np
import onnxruntime as rt

print("Welcome to the Object Detector Tester")
run_time = time.perf_counter()

print("Initialising detector")
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if rt.get_device()=='GPU' else ['CPUExecutionProvider']
session = rt.InferenceSession('./net/yolox_nano.onnx', providers=providers)
outname = [i.name for i in session.get_outputs()] 
inname = [i.name for i in session.get_inputs()]
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
        blob = cv2.dnn.blobFromImage(frame, size = (416, 416), # Resolution multiple of 32
                                     swapRB=True, crop=False) 
        inp = {inname[0]:blob}
        layer_output = session.run(outname, inp)[0][0]  # Assume batch size of 1                 

        # Record performance
        rate_fps = 1/(time.perf_counter()-run_time)
        run_time = time.perf_counter()
        frame = cv2.putText(frame, str(round(rate_fps))+" FPS",
                            ((frame.shape[1]//10)*7,(frame.shape[0]//10)*9), font, 1, color)

        # Perform inference
        coco_ind = 0  # Person
        stride = 16  # Stride: 8,16,32
        cell_count = 416//stride

        max_inds = np.argwhere(layer_output[(416//8)**2:(416//8)**2+cell_count**2,coco_ind+5]>0.5)
        
        for ind in max_inds:
            ind = ind - (416//8)**2
            # Find cell representation
            x = (ind%(cell_count**2))%cell_count
            y = (ind%(cell_count**2))//cell_count
            
            # Find max prediction as result
            p_x, p_y, p_w, p_h = layer_output[ind,:4][0]

            # Find pixel representation
            x = int((x+0.5)/cell_count*frame.shape[1])
            y = int((y+0.5)/cell_count*frame.shape[0])
            l_w, l_h = (stride*np.exp(np.array([p_w, p_h]))).astype("int")
            l_x, l_y = np.array([p_x,p_y]).astype("int")

            frame = cv2.circle(frame, (x+l_x,y+l_y), 10,  (0, 255, 0), 3)            
            frame = cv2.rectangle(frame, (x-l_w//2, y-l_h//2), (x+l_w//2, y+l_h//2), (0, 255, 0), 3)
        
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
