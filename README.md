# myDetectorEvaluation

Apply the Viola Jones and Tiny Yolov3 detectors on an input video to detect human faces. A frames per second measurement is reported to evaluate realtime performance. Both Python and c++ implementations exist for comparrison

## OpenCV installation for Ubuntu via repo
```
sudo apt update
sudo apt install libopencv-dev python3-opencv
```

## Build instructions for c++ implmentation
```
cmake .
make
```

## Download trained weights
```
fetch_weights.sh
```

## Example usage (any of the following)
```
run_test_tyolo.py <video_filename>
run_test_vj.py <video_filename>
run_test_vj <video_filename>
run_test_tyolo <video_filename>
```
