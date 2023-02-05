#include <chrono>
#include <iostream>
#include "opencv2/opencv.hpp"

using std::chrono::steady_clock;
using std::chrono::duration;

int main(int argc, char *argv[]) {
    std::cout << "Loading video" << std::endl;
    // Create a VideoCapture object and open the input file
    // If the input is the web camera, pass 0 instead of the video file name
    cv::VideoCapture video(argv[1]);  // Name must include a number
    auto start_time = steady_clock::now();

    // Check if camera opened successfully
    if (!video.isOpened()) {
        std::cout << "Error opening video stream or file" << std::endl;
        return -1;
    } else {
        std::cout << "Video open" << std::endl;
    }

    // Load pretrained tiny yolo
    cv::dnn::Net detector;
    detector = cv::dnn::readNetFromDarknet("./net/yolov3-tiny.cfg",
                                           "./net/yolov3-tiny.weights");
    std::vector<cv::String> outLayers;
    std::vector<int> outLayers_ind = detector.getUnconnectedOutLayers();
    std::vector<cv::String> layersNames = detector.getLayerNames();
    for (int layer_no : outLayers_ind) {
        outLayers.push_back(layersNames[layer_no-1]);
        }

    // Main loop
    while (true) {
        cv::Mat frame;  // OpenCV Matrix
        video >> frame;  // Retrieve frame-by-frame

        // If the frame is empty, break
        if (frame.empty())
          break;

        // Preprocess
        cv::Mat blob;
        cv::dnn::blobFromImage(frame, blob, 1./255., cv::Size(416, 416),
                               cv::Scalar(), true, true);

        // Perform detection
        std::vector<cv::Mat> layer_outputs;
        detector.setInput(blob);
        detector.forward(layer_outputs, outLayers);

        bool positive_predict = false;
        // iterate through the two output layers
        for (int layer_no = 0; layer_no < layer_outputs.size(); ++layer_no) {
            // iterate through all detections
            for (int detection_no = 0;
                 detection_no < layer_outputs[layer_no].rows; ++detection_no) {
                // continue;  // Skip block check efficiency
                cv::Mat data = layer_outputs[layer_no].row(detection_no);
                if (data.at<float>(5) > 0.2) {
                    // Box dimensions
                    int cx = static_cast<int>(data.at<float>(0)*frame.cols);
                    int cy = static_cast<int>(data.at<float>(1)*frame.rows);
                    int w = static_cast<int>(data.at<float>(2)*frame.cols);
                    int h = static_cast<int>(data.at<float>(3)*frame.rows);

                    cv::rectangle(frame, cv::Point(cx-w/2, cy-h/2),
                                  cv::Point(cx + w/2, cy + h/2),
                                  cv::Scalar(0, 255, 0), 3);
                    positive_predict = true;
                    break;
                    }
                if (positive_predict == true) {
                    break;
                    }
                }
            }

        // Record performance
        int perform_fps =  1/(duration<double, std::milli>(steady_clock::now()-start_time).count()/1e+3);
        start_time = steady_clock::now();
        putText(frame,  // target image
                std::to_string(perform_fps)+" FPS",  // text
                cv::Point(frame.cols/10*7, frame.rows/10*9),  // top-left
                cv::FONT_HERSHEY_DUPLEX,
                1,
                CV_RGB(0, 0, 0),  // font color
                2);

        // Display the resulting frame
        imshow("Frame", frame);

        // Press  ESC on keyboard to exit
        char c = static_cast<char>(cv::waitKey(1));
        if (c == 27) {
            break;
            }
        }

        video.release();
        cv::destroyAllWindows();

    return 0;
    }
