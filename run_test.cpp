#include "opencv2/opencv.hpp"
#include <iostream>

using namespace std;
using namespace cv;

int main(){
    
    cout << "Hello";
    // Create a VideoCapture object and open the input file
    // If the input is the web camera, pass 0 instead of the video file name
    VideoCapture video("./tracking_test0.avi");  //Name must include a number
    cout<<"Open";

    // Check if camera opened successfully
    //if(!video.isOpened()){
    //    cout << "Error opening video stream or file" << endl;
    //    return -1;
    //    }
   
  /*while(true){
 
    Mat frame;
    // Capture frame-by-frame
    cap >> frame;
  
    // If the frame is empty, break immediately
    if (frame.empty())
      break;
 
    // Display the resulting frame
    imshow( "Frame", frame );
 
    // Press  ESC on keyboard to exit
    char c=(char)waitKey(25);
    if(c==27)
      break;
  }*/
  
    video.release(); // When everything done, release the video capture object
    //destroyAllWindows(); // Closes all the frames
   
    return 0;
    }

