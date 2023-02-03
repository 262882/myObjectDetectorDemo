#include "opencv2/opencv.hpp"
#include <iostream>
#include <chrono>

using namespace std;
using namespace cv;
using namespace std::chrono;
using namespace cv::dnn;

int main(int argc, char *argv[]){
    
    cout << "Loading video" << endl;
    // Create a VideoCapture object and open the input file
    // If the input is the web camera, pass 0 instead of the video file name
    VideoCapture video(argv[1]);  //Name must include a number
    auto current_time = steady_clock::now();

    // Check if camera opened successfully
    if(!video.isOpened()){
        cout << "Error opening video stream or file" << endl;
        return -1;
        }
    else{
        cout << "Video open" << endl;
        }

    // Load pretrained tiny yolo
    Net detector; 
    detector = readNetFromDarknet("./net/yolov3-tiny.cfg","./net/yolov3-tiny.weights");
    vector<String> outLayers;
    vector<int> outLayers_ind = detector.getUnconnectedOutLayers(); //indices of the output layer
    vector<String> layersNames = detector.getLayerNames(); //names of all the layers
    for (int layer_no : outLayers_ind){
        outLayers.push_back(layersNames[layer_no-1]);
        }
    

    // Main loop
    while(true){
 
        Mat frame;  // OpenCV Matrix
        video >> frame; // Retrieve frame-by-frame

        // If the frame is empty, break
        if (frame.empty())
          break;

        // Preprocess
        Mat blob;
        blobFromImage(frame, blob, 1./255., Size(416, 416), Scalar(), true, true);

        // Perform detection
        vector<Mat> layer_outputs;
        detector.setInput(blob);
        detector.forward(layer_outputs, outLayers);

        bool positive_predict = false;
        for (int layer_no = 0; layer_no < layer_outputs.size(); ++layer_no){  // iterate through the two output layers
            for (int detection_no = 0; detection_no < layer_outputs[layer_no].rows; ++detection_no){ // iterate through all detections
                //continue; // By skipping block we can test if is efficent enough
                Mat data = layer_outputs[layer_no].row(detection_no);
                if (data.at<float>(5) > 0.2){

                    // Box dimensions
                    int cx = static_cast<int>(data.at<float>(0)*frame.cols);
                    int cy = static_cast<int>(data.at<float>(1)*frame.rows);
                    int w = static_cast<int>(data.at<float>(2)*frame.cols);
                    int h = static_cast<int>(data.at<float>(3)*frame.rows);

                    rectangle(frame, Point(cx-w/2, cy-h/2), Point(cx + w/2, cy + h/2), Scalar(0, 255, 0), 3);
                    positive_predict = true;
                    break;
                    }
                if (positive_predict == true){
                    break;
                    }
                }
            }  

        // Record performance
        int perform_fps =  1/(duration<double, std::milli>(steady_clock::now()-current_time).count()/1e+3);
        current_time = steady_clock::now();
        putText(frame, //target image
                to_string(perform_fps)+" FPS", //text
                cv::Point(frame.cols/10*7, frame.rows/10*9), //top-left position
                cv::FONT_HERSHEY_DUPLEX,
                1,
                CV_RGB(0, 0, 0), //font color
                2);
    
        // Display the resulting frame
        imshow("Frame", frame );
    
        // Press  ESC on keyboard to exit
        char c=(char)waitKey(1);
        if(c==27){
            break;
            }
        } 
      
        video.release(); // When everything done, release the video capture object
        destroyAllWindows(); // Closes all the frames
      
    return 0;
    }

