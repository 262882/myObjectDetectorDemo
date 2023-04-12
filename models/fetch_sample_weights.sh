#!/bin/bash
echo "Creating directory"
mkdir net
cd net

echo "Retrieving Viola Jones weights"
wget https://github.com/kipr/opencv/raw/master/data/haarcascades/haarcascade_frontalface_default.xml

echo "Retrieving mobile ssd files"
wget https://github.com/PINTO0309/MobileNet-SSD-RealSense/raw/master/caffemodel/MobileNetSSD/MobileNetSSD_deploy.prototxt
wget https://github.com/PINTO0309/MobileNet-SSD-RealSense/raw/master/caffemodel/MobileNetSSD/MobileNetSSD_deploy.caffemodel

echo "Retrieving yolox nano"
wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx

echo "Retrieving nanodet files"
wget https://github.com/hpc203/nanodet-opncv-dnn-cpp-python/raw/main/nanodet.onnx

echo "Retrieving yolo-tiny files"
wget https://github.com/pjreddie/darknet/raw/master/cfg/yolov3-tiny.cfg
wget https://pjreddie.com/media/files/yolov3-tiny.weights
