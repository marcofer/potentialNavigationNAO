// System includes
#include <iostream>
#include <signal.h>
#include <pthread.h>


// OpenCV includes
#include <opencv2/opencv.hpp>

// Joystick class include
#include "cJoystick.h"

// of_driving class include
#include "of_driving.h"

#include "utils.h"

using namespace std;
using namespace cv;
using namespace AL;


namespace AL {
    class ALBroker;
}



class potentialNavigationNAO : public AL::ALModule{
private:

    //flag stating if the algorithm is running online or offline [offline handling is to be implemented]
    bool online;
    //flag stating if the  algorithm is running on a real NAO or on simulated one (Choregraphe)
    bool realNAO;
    //flag stating if the NAO is manually or automatically driven
    bool manual;

    bool record;

    //online structures
    string pip;
    int pport;
    int camera_flag;

    //offline structures
    string video_path;

    // ALModule variables
    // Aldebaran built-in modules
    ALMotionProxy motionProxy;
    ALMotionProxy* motPtr;
    ALVideoDeviceProxy cameraProxy;
    void* recorderProxy;

    //Opencv Structures
    cv::VideoCapture vc;
    int frame_counter;

    double pan, tilt_cmd;
    double camera_tilt, camera_height;
    Mat cameraOrientation, cameraOrientationT;
    Mat cameraPosition;
    std::vector<float> cameraFrame;
    std::vector<float> headYawFrame;

    bool headset, firstKeyIgnored;
    string cameraName;
    Mat OCVimage;
    ALValue ALimg, config;

    //needed structures
    Size imgSize;
    Mat img, prev_img;
    cJoystick* joy;
    js_event* jse;
    joystick_state* js;

    bool move_robot;

    //of_driving object
    of_driving drive;
    timeval start_tod, end_tod, tic,toc;
    double elapsed_tod;

    ofstream cycle_f, cameraRate_f;

    double v, vx, vy, vmax;
    double w, wz, wmax; //TODO: fix names (wz is used for centroids control law, w for potential)
    double x_f;

    void updateTcAndLowPass();
    void updateTilt(int);
    void updateCameraPose();
    void enableRecording();
    void cleanAllActivities();
    void printTiltInfo();

    short int catchState(char);
    void applyControlInputs();
    double current_imageTime, previous_imageTime;
    double curtime;

    string full_path;
    unsigned int fileSeq;
    void openFiles();
    void closeFiles();

public:

    potentialNavigationNAO(boost::shared_ptr<AL::ALBroker> broker, const string &name);

    virtual ~potentialNavigationNAO();
    virtual void init();

    void getVideoPath(const std::string& vp);
    void setCaptureMode(const bool &on);


    void setTiltHead(char);
    void chooseCamera(int);
    void run();

};
