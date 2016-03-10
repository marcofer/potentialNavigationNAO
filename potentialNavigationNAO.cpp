#include <signal.h>
#include "potentialNavigationNAO.h"
#include <alvision/alvisiondefinitions.h>
#include <alerror/alerror.h>
#include <pthread.h>
#include <qi/os.hpp>
#include<sys/stat.h>
#include<sys/types.h>
#include <fstream>

#define LEFT_UD_AXIS 1
#define RIGHT_LR_AXIS 2
#define X_BUTTON 2
#define SQUARE_BUTTON 3
#define TRIANGLE_BUTTON 0
#define CIRCLE_BUTTON 1
#define START_BUTTON 9
#define JOY_MAXVAL 32767


#define TILT_JOINT "HeadPitch"
#define PAN_JOINT "HeadYaw"

#define MINPITCH -0.671951
#define MAXPITCH 0.515047
#define LINEAR_VEL_MAX 0.0476 //(half max vel)
#define ANGULAR_VEL_MAX 0.42 //0.83

#define SAVE_VIDEO true


potentialNavigationNAO::potentialNavigationNAO(
    boost::shared_ptr<AL::ALBroker> broker, const string& name)
      : AL::ALModule(broker, name)
      , motionProxy(AL::ALMotionProxy(broker))
      , cameraProxy(AL::ALVideoDeviceProxy(broker))
{
  // Describe the module here. This will appear on the webpage
  setModuleDescription("Potential Navigation Module.");
}


potentialNavigationNAO::~potentialNavigationNAO(){
    cout << "Destroyer called. " << endl ;
    cleanAllActivities();
}

void potentialNavigationNAO::init(){

    cout << "[potentialNavigationNAO] Initializing ... " << endl;
    cout << cv::getBuildInformation() << endl ;
    online = true;

    // Choose NAO camera
    camera_flag = 1 ;//1: top - 0: bottom
    chooseCamera(camera_flag);

    // Call getImageRemote just once before the main loop to obtain NAOimage characteristics
    ALimg = cameraProxy.getImageRemote(cameraName);
    /// compute camera framerate
    current_imageTime = (int)ALimg[4]          // second
                        + (int)ALimg[5] * 1e-6;  // usec to seconds
    previous_imageTime = current_imageTime;


    imgSize = Size((int)ALimg[0],(int)ALimg[1]);
    OCVimage.create(imgSize.height,imgSize.width,CV_8UC1);
    OCVimage.data = (uchar*)ALimg[6].GetBinary();
    img = Mat::zeros(imgSize,CV_8UC1);
    prev_img = Mat::zeros(imgSize,CV_8UC1);

    // Joystick variables
    joy = new cJoystick;
    jse = joy->getJsEventPtr();
    js = joy->getJsStatePtr();

    // Head pan-tilt variables
    pan = 0.0;
    tilt = 0.0;
    headset = false;

    // of_driving object
    drive.set_imgSize(imgSize.width,imgSize.height);
    drive.initFlows(SAVE_VIDEO);

    full_path = "/home/ubu1204/Documents/Software/qi_work_tree/potentialNavigationNAO/plot";

    try{
        openFiles();
    }
    catch(...){
        cerr << "Error! Some files cannot be opened!!! " << endl;
        std::exit(1);
    }
    gettimeofday(&start_tod,NULL);
    elapsed_tod = 0.0;

    run();

    return;

}


void potentialNavigationNAO::chooseCamera(int camera_flag){

    // ALVideoDeviceProxy::subscribe(const std::string& vmName, const int& resolution, const int& colorSpace, const int& fps)
    // vmName     – Name of the subscribing V.M.
    // resolution – Resolution requested. { 0 = kQQVGA, 1 = kQVGA, 2 = kVGA, 3 = k4VGA }
    // colorSpace – Colorspace requested. { 0 = kYuv, 9 = kYUV422, 10 = kYUV, 11 = kRGB, 12 = kHSY, 13 = kBGR }
    // fps        – Fps (frames per second) requested to the video source.

    cameraName = (camera_flag) ? ("bottom") : ("top");
    cameraName = cameraProxy.subscribeCamera(cameraName, camera_flag,  AL::kQVGA, AL::kYuvColorSpace, 30);
    //cameraName = cameraProxy.subscribe(cameraName, kQVGA, kBGRColorSpace, 30);
    //cameraProxy.setActiveCamera(cameraName,camera_flag);
    cout << "Camera chosen: " << cameraName << endl;
    cout << "Frame rate: " << cameraProxy.getFrameRate(cameraName) << endl;
}




void potentialNavigationNAO::cleanAllActivities(){
    delete joy;

    closeFiles();

    cameraProxy.unsubscribe(cameraName);

    if(SAVE_VIDEO && headset){
        recorderProxy.stopRecording();
    }

    motionProxy.stopMove();
    qi::os::sleep(2);

    cout << "Resetting initial posture ... " << endl;
    motionProxy.moveInit();


    cv::destroyAllWindows();
    cout << "all active works have been stopped. " << endl ;

}


void potentialNavigationNAO::setTiltHead(char key){

    int dtilt = 0;
    int res;


    // Capture image from subscribed camera
    ALimg = cameraProxy.getImageRemote(cameraName);
    printTiltInfo();

    switch(key){
    case('u'):
        dtilt = 1;
        break;
    case('d'):
        dtilt = -1;
        break;
    case('f'):
        headset = true;
        cv::destroyAllWindows();
        drive.createWindowAndTracks();
        if(SAVE_VIDEO){
            enableRecording();
        }
        break;
    default:
        dtilt = 0;
        break;
    }
    updateTilt(dtilt);

    AL::ALValue names = AL::ALValue::array(PAN_JOINT,TILT_JOINT);
    AL::ALValue angles = AL::ALValue::array(pan,tilt);
    double fracSpeed = 0.2;

    motionProxy.setAngles(names,angles,fracSpeed);
}


void potentialNavigationNAO::enableRecording(){
    string folder_path = "/home/nao/recordings/cameras/NAO_potentialNavigation/video";

    recorderProxy.setColorSpace(11);//AL::kRGBColorSpace : buffer contains triplet on the format 0xBBGGRR, equivalent to three unsigned char
    recorderProxy.setResolution(1);//kQVGA
    recorderProxy.setVideoFormat("MJPG");
    recorderProxy.setFrameRate(30);
    recorderProxy.startRecording(folder_path,"NAOvideo",true);//*/
}

void potentialNavigationNAO::updateTilt(int dtilt){
    double delta = 0.1;
    tilt += (double)dtilt*delta;

    tilt = (tilt < MAXPITCH) ? ((tilt > MINPITCH) ? (tilt) : (MINPITCH)) : (MAXPITCH) ;
}

void potentialNavigationNAO::run(){

    char key;
    double camera_rate;
    cout << "Running ... " << endl;
    curtime = getTickCount();

    motionProxy.stiffnessInterpolation("Body", 1.0, 1.0); //// DOES THE STIFFNESS HAVE TO BE  1.0?????
    qi::os::sleep(2);
    motionProxy.moveInit();

    // Command variables
    manual = false;
    cout << "MANUAL CONTROL" << endl ;


    while(true){

        key = cv::waitKey(1);

        //Catch keyboard input to select the mode operation (<Manual>/<Autonomous + Stop/Go>)
        if(catchState(key)==-1)
            break;


        if(online){
            if(!headset){
                setTiltHead(key);
            }
            else{

                // capture image from subscribed camera
                ALimg = cameraProxy.getImageRemote(cameraName);

                /// compute camera framerate
                current_imageTime = (int)ALimg[4]          // second
                                    + (int)ALimg[5] * 1e-6;  // usec to seconds

                camera_rate = 1.0/( current_imageTime - previous_imageTime);

                cout << std::fixed << "\n[LABROB] camera frame rate:  " << camera_rate << " Hz" << endl;
                /*cout << std::fixed << "[LABROB] current_imageTime:  " << current_imageTime << endl;
                cout << std::fixed << "[LABROB] previous_imageTime:  " << previous_imageTime << endl;//*/

                previous_imageTime = current_imageTime;


                if(camera_rate!=INFINITY){
                    double tic = getTickCount();
                    cameraRate_f << camera_rate << ";" << endl;

                    // update working images
                    img.copyTo(prev_img);
                    OCVimage.copyTo(img);

                    //make the algorithm run
                    updateTcAndLowPass();

                    try{
                        drive.run(img,prev_img,SAVE_VIDEO);
                    }
                    catch(...){
                        cerr << "Problem in drive.run. " << endl;
                        std::exit(1);
                    }



                    getVelocityCommands();


                    //command NAO
                    if((move_robot) || (manual)){
                        motionProxy.move(v,0.0f,-w);
                    }
                    else{
                        motionProxy.stopMove();
                    }
                    double toc = getTickCount();
                    double tictoc = (toc - tic)/getTickFrequency();
                    //cout << "tictoc: " << tictoc << endl;
                }//*/
            }
        }

    }

    cleanAllActivities();

}

short int potentialNavigationNAO::catchState(char key){
    if((int)key == 27){
      cout << "[potentialNavigationNAO] Caught ESC! Waiting for closing/deleting running activities ... "<< endl;
      return -1;
    }
    else if(key=='g'){
        cout << "Go NAO!" << endl;
        if(!manual) move_robot = true;
    }
    else if(key=='s'){
        cout << "Stop NAO!" << endl;
        if(!manual) move_robot = false;
    }
    else if(key=='m'){
        manual = !manual;
        v = 0.0;
        w = 0.0;
        cout << ((manual) ? ("MANUAL CONTROL") : ("AUTONOMOUS CONTROL")) << endl ;

    }
    return 1;
}

void potentialNavigationNAO::getVelocityCommands(){
    //if manual then read joystick values
    if(manual){
        int res = joy->readEv(); //<--- don't print this value, or the joystick won't answer
        if(res != -1){
            if(jse->type & JS_EVENT_AXIS){
                if((int)jse->number == LEFT_UD_AXIS){//tilt down
                    v = -(jse->value)*LINEAR_VEL_MAX/(double)JOY_MAXVAL;
                    cout << "v: " << v << endl;
                }
                else if((int)jse->number == RIGHT_LR_AXIS){//tilt up
                    w = (jse->value)*ANGULAR_VEL_MAX/(double)JOY_MAXVAL;
                    cout << "w: " << w << endl;
                }
            }
        }
    }
    else{
        v = drive.get_linearVel();
        double norm_w = drive.get_angularVel();
        w = norm_w*ANGULAR_VEL_MAX;
    }


}

void potentialNavigationNAO::updateTcAndLowPass(){
    gettimeofday(&end_tod,NULL);

    elapsed_tod = (end_tod.tv_sec + (double)end_tod.tv_usec /1000000.0)
          - (start_tod.tv_sec + (double)start_tod.tv_usec/1000000.0);

    start_tod = end_tod;
    double fc_ratio = 0.25;
    double cutoff_f = 1.0/elapsed_tod*fc_ratio;
    double freq = 1.0/elapsed_tod;

    double loop_time;
    double now = getTickCount();
    loop_time = (now - curtime)/getTickFrequency();
    curtime = now;

    //cout << "[LABROB] Loop rate:    " << 1.0/loop_time << "Hz" << endl;

    cycle_f << 1.0/loop_time << "; " << endl;


    /*drive.setTc(elapsed_tod);
    drive.setImgLowPassFrequency(cutoff_f);
    drive.setBarLowPassFrequency(cutoff_f*fc_ratio*0.5);//*/

    //If under Ethernet connection, the framerate is around 15Hz


    //drive.setTc(0.04);//25Hz
    drive.setTc(1.0/15.0);
    drive.setImgLowPassFrequency(7.0);
    drive.setBarLowPassFrequency(7.0);

    //cout << "elapsed_time: " << elapsed_tod << endl;
    /*cout << "image cut-off frequency: " << drive.getImgLowPassFrequency() << endl;
    cout << "control cut-off frequency: " << drive.getBarLowPassFrequency() << endl << endl;//*/

}

void potentialNavigationNAO::printTiltInfo(){

    string text_str;
    ostringstream convert;
    Size t_size, value_size;
    double font_scale = 0.9;
    Mat infoImg;
    OCVimage.copyTo(infoImg);

    text_str = "";
    text_str = "TILT [rad]= ";
    t_size = getTextSize(text_str,1,font_scale,1,0);
    putText(infoImg, text_str,Point(10,infoImg.rows - 10 - t_size.height),1,font_scale,Scalar(255,255,255),1,CV_AA);

    text_str = "";
    convert.str(""); convert.clear();
    convert << setprecision(4) << tilt;
    text_str = convert.str();
    value_size = getTextSize(text_str,1,font_scale,1,0);
    putText(infoImg, text_str,Point(t_size.width + 10,infoImg.rows - 10 - t_size.height),1,font_scale,Scalar(255,255,255),1,CV_AA);

    imshow("camera",infoImg);
    waitKey(1);

}


void potentialNavigationNAO::openFiles(){

    ifstream seqFileIn;

    seqFileIn.open("sequeceFile.txt", ios::in);

    // If "sequenceFile.txt" exists, read the last sequence from it and increment it by 1.
    if (seqFileIn.is_open())
    {
        seqFileIn >> fileSeq;
        fileSeq++;
    }
    else{
        fileSeq = 1; // if it does not exist, start from sequence 1.
    }

    full_path = full_path + to_string(fileSeq) + "/";

    mkdir(full_path.c_str(),0777);//creating a directory


    string cycle_path = full_path + "full_cycle_time.txt";
    string camera_path = full_path + "cameraFPS.txt";

    cycle_f.open(cycle_path.c_str(),ios::app);
    cameraRate_f.open(camera_path.c_str(),ios::app);

    cout << "teeeeeeeest" << endl;
    drive.openFiles(full_path);
}


void potentialNavigationNAO::closeFiles(){

    ofstream seqFileOut;

    drive.closeFiles();
    cycle_f.close();
    cameraRate_f.close();

    seqFileOut.open("sequeceFile.txt", ios::out);
    seqFileOut << fileSeq;
    seqFileOut.close();

}
