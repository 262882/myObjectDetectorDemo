#!/bin/bash
echo "Creating directory"
mkdir net
cd net

echo "Retrieving Viola Jones weights"
wget https://github.com/kipr/opencv/raw/master/data/haarcascades/haarcascade_frontalface_default.xml

echo "Retrieving config files"
wget https://github.com/pjreddie/darknet/raw/master/cfg/yolov3-tiny.cfg

echo "Retrieving yolo-tiny weights"
wget https://pjreddie.com/media/files/yolov3-tiny.weights
