#include "opencv2/opencv.hpp"
#include <iostream>
#include <chrono>

using namespace std;
using namespace cv;
using namespace std::chrono;

int main(){
    
    cout << "Loading video" << endl;
    // Create a VideoCapture object and open the input file
    // If the input is the web camera, pass 0 instead of the video file name
    VideoCapture video("./tracking_test0.avi");  //Name must include a number
    auto current_time = steady_clock::now();

    // Check if camera opened successfully
    if(!video.isOpened()){
        cout << "Error opening video stream or file" << endl;
        return -1;
        }
    else{
        cout << "Video open" << endl;
        }

    // Load pretrained XML classifier
    CascadeClassifier detector; 
    double scale=1;
    detector.load("haarcascade_frontalface_default.xml") ; 

    // Main loop
    while(true){
 
        Mat frame;  // OpenCV Matrix
        video >> frame; // Retrieve frame-by-frame
      
        // If the frame is empty, break
        if (frame.empty())
          break;
        
        Mat frame_gray;
        cvtColor( frame, frame_gray, COLOR_BGR2GRAY );

        // Perform detection
        vector<Rect> faces;
        detector.detectMultiScale( frame_gray, faces,1.1,10);
        for(size_t i = 0; i < faces.size(); i++) {
            rectangle(frame, faces[i], Scalar(0, 255, 0), 3);
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

