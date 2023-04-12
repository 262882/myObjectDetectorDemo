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
    auto current_time = steady_clock::now();

    // Check if camera opened successfully
    if (!video.isOpened()) {
        std::cout << "Error opening video stream or file" << std::endl;
        return -1;
    } else {
        std::cout << "Video open" << std::endl;
        }

    // Load pretrained XML classifier
    cv::CascadeClassifier detector;
    detector.load("../models/haarcascade_frontalface_default.xml");

    // Main loop
    while (true) {
        cv::Mat frame;  // OpenCV Matrix
        video >> frame;  // Retrieve frame-by-frame

        // If the frame is empty, break
        if (frame.empty())
          break;

        cv::Mat frame_gray;
        cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);

        // Perform detection
        std::vector<cv::Rect> faces;
        detector.detectMultiScale(frame_gray, faces, 1.1, 10);
        for (size_t i = 0; i < faces.size(); i++) {
            rectangle(frame, faces[i], cv::Scalar(0, 255, 0), 3);
            }

        // Record performance
        int perform_fps =  1/(duration<double, std::milli>(steady_clock::now()-current_time).count()/1e+3);
        current_time = steady_clock::now();
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

