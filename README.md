# myDetectorEvaluation

Apply various object detectors on an input video to detect people. A frames per second measurement is reported to evaluate realtime performance. NMS and a measure of quality of inference is skipped as the intension is only to get a feel for the realtime performance and methods of deployment. Both Python and c++ implementations for VJ and Tiny yolo exist for comparrison.

## OpenCV installation for Ubuntu via repo:
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
