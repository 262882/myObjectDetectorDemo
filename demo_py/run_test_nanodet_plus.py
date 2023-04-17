#!/usr/bin/env python3
"""Start object detector test"""

import time
import sys
import cv2
import numpy as np
import onnxruntime as rt

def _normalize(img): 
    img = img.astype(np.float32) / 255
    MEAN = np.array([0.406, 0.456, 0.485], dtype=np.float32)
    STD = np.array([0.225, 0.224, 0.229], dtype=np.float32)
    img = img - MEAN / STD
    return img

print("Welcome to the Object Detector Tester")
run_time = time.perf_counter()

print("Initialising detector")
providers = ['CPUExecutionProvider']
sess_options = rt.SessionOptions()  # https://onnxruntime.ai/docs/performance/tune-performance.html
sess_options.intra_op_num_threads = 1
sess_options.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
session = rt.InferenceSession('../models/nanodet-plus-m_416.onnx', sess_options=sess_options, providers=providers)
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
        blob = cv2.dnn.blobFromImage(_normalize(frame), 
                                        size = (416, 416),
                                        swapRB=True, crop=False) 
        inp = {inname[0]:blob}
        layer_output = session.run(outname, inp)[0][0]                        

        # Record performance
        rate_fps = 1/(time.perf_counter()-run_time)
        run_time = time.perf_counter()
        frame = cv2.putText(frame, str(round(rate_fps))+" FPS",
                            ((frame.shape[1]//10)*7,(frame.shape[0]//10)*9), font, 1, color)

        coco_ind = 0  # Person
        stride = 8  # Stride: 8,16,32
        cell_count = 416//stride

        max_inds = np.argwhere(layer_output[:(cell_count)**2, coco_ind] > 0.2)

        for ind in max_inds:
            ind=ind[0]

            # Find pixel representation
            x = int((ind%cell_count+0.5)*frame.shape[0]/cell_count)-1  # Compensate for different size input image
            y = int((ind//cell_count+0.5)*frame.shape[0]/cell_count)-1

            # Find max prediction as result
            box = np.max(np.reshape(layer_output[ind,-32:],(4,8)),axis=1)
            l,t,r,b = (box*frame.shape[0]/cell_count).astype("int")
          
            frame = cv2.rectangle(frame, (x-l, y-t), (x+r, y+b),(0, 255, 0), 2)

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
