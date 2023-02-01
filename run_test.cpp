#include "opencv2/opencv.hpp"
#include <iostream>
#include <ctime>

using namespace std;
using namespace cv;

int main(){
    
    cout << "Loading video" << endl;
    // Create a VideoCapture object and open the input file
    // If the input is the web camera, pass 0 instead of the video file name
    VideoCapture video("./tracking_test0.avi");  //Name must include a number
    time_t current_time = clock();

    // Check if camera opened successfully
    if(!video.isOpened()){
        cout << "Error opening video stream or file" << endl;
        return -1;
        }
    else{
        cout << "Video open" << endl;
        }

    // Main loop
    while(true){
 
    Mat frame;  // OpenCV Matrix
    video >> frame; // Retrieve frame-by-frame
  
    // If the frame is empty, break
    if (frame.empty())
      break;

    // Record performance
    int perform_fps =  1/((clock()-current_time)/1e+6);
    current_time = clock();
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
    if(c==27)
      break;
    } 
  
    video.release(); // When everything done, release the video capture object
    destroyAllWindows(); // Closes all the frames
   
    return 0;
    }

