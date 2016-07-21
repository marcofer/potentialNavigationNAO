#include <signal.h>
#include "potentialNavigationNAO.h"
#include <alvision/alvisiondefinitions.h>
#include <alerror/alerror.h>
#include <pthread.h>
#include <qi/os.hpp>
#include<sys/stat.h>
#include<sys/types.h>
#include <fstream>


potentialNavigationNAO::potentialNavigationNAO(
    boost::shared_ptr<AL::ALBroker> broker, const string& name)
      : AL::ALModule(broker, name)
      , motionProxy(AL::ALMotionProxy(broker))
      , cameraProxy(AL::ALVideoDeviceProxy(broker))

{
    // If the running NAOqi instance is not local, instantiate the VideoRecorder Proxy to record video from camera
    if(broker->getParentIP() != "127.0.0.1" && SAVE_VIDEO){
        recorderProxy = static_cast<AL::ALVideoRecorderProxy*>(recorderProxy);
        recorderProxy = new AL::ALVideoRecorderProxy(broker);
    }

    // Describe the module here. This will appear on the webpage
    setModuleDescription("Potential Navigation Module.");

    functionName("getVideoPath", getName(), "Store the input video path for offline processing.");
    addParam("video_path", "The input video path.");
    BIND_METHOD(potentialNavigationNAO::getVideoPath);

    functionName("setCaptureMode", getName(), "Set proper structures according to the capture mode (online/offline).");
    addParam("online", "Boolean variable stating capture mode");
    BIND_METHOD(potentialNavigationNAO::setCaptureMode);

    functionName("run", getName(), "Run the main loop.");
    BIND_METHOD(potentialNavigationNAO::run);

}

void potentialNavigationNAO::getVideoPath(const string &vp){
    video_path = vp;

    cout << "Video path correctly acquired: " << video_path << endl;
}


void potentialNavigationNAO::setCaptureMode(const bool& onoff){
    online = onoff;
    cout << "Is the algorithm running online? " << ((online) ? ("YES") : ("NO")) << endl;

    if(online){
        // Choose NAO camera
        camera_flag = 0 ;//1: bottom - 0: top
        chooseCamera(camera_flag);

        // Call getImageRemote just once before the main loop to obtain NAOimage characteristics
        ALimg = cameraProxy.getImageRemote(cameraName);
        /// compute camera framerate
        current_imageTime = (int)ALimg[4]          // second
                            + (int)ALimg[5] * 1e-6;  // usec to seconds
        previous_imageTime = current_imageTime;

        //imgSize = Size((int)ALimg[0],(int)ALimg[1]);
        //OCVimage.create(imgSize.height,imgSize.width,CV_8UC1);
        //OCVimage.data = (uchar*)ALimg[6].GetBinary();

        workingImage.create((int)ALimg[1],(int)ALimg[0],CV_8UC1);
        workingImage.data = (uchar*)ALimg[6].GetBinary();

    }
    else{

        vc.open(video_path);

        if(!vc.isOpened()){
            cout << "ERROR:Input video not found." << endl;
            std::exit(1);
        }

        vc >> workingImage;
        resize(workingImage,workingImage,cv::Size(320,240));
        cvtColor(workingImage,workingImage,CV_BGR2GRAY);

    }


    ROI_x = 0.0;
    ROI_y = workingImage.rows/2;
    ROI_width = workingImage.cols - ROI_x;
    ROI_height = workingImage.rows - ROI_y;

    OCVimage = workingImage(Rect(ROI_x,ROI_y,ROI_width,ROI_height));
    imgSize = OCVimage.size();

    //img = Mat::zeros(imgSize,CV_8UC1);
    //prev_img = Mat::zeros(imgSize,CV_8UC1);

    //OCVimage.copyTo(img);
    //OCVimage.copyTo(prev_img);
    img = OCVimage.clone();
    prev_img = OCVimage.clone();


    // of_driving object
    drive.set_imgSize(imgSize.width,imgSize.height, ROI_x, ROI_y);
    drive.initFlows(SAVE_VIDEO);

    motPtr = new AL::ALMotionProxy;
    *motPtr = motionProxy;
    drive.set_ALMotionPtr(motPtr);

}

potentialNavigationNAO::~potentialNavigationNAO(){
    cout << "Destroyer called. " << endl ;
    cleanAllActivities();
}

void potentialNavigationNAO::init(){

    cout << "[potentialNavigationNAO] Initializing ... " << endl;

    // Joystick variables
    joy = new cJoystick;
    jse = joy->getJsEventPtr();
    js = joy->getJsStatePtr();

    // Head pan-tilt variables
    pan = 0.0;
    tilt_cmd = 0.0;

    headset = false;
    frame_counter = 0;

    vmax = drive.get_linVelMax();
    wmax = drive.get_angVelMax();

    //full_path = "/home/ubu1204//Documents/Software/NAO/qi_work_tree/potentialNavigationNAO/plot";
    char curDirAndFile[1024];
    getcwd(curDirAndFile, sizeof(curDirAndFile));
    full_path = std::string(curDirAndFile);
    full_path += "/potentialNavigationNAO/plot";
    std::cout << "full_path: " << full_path << std::endl;

    try{
        openFiles();
    }
    catch(...){
        cerr << "Error! Some files cannot be opened!!! " << endl;
        std::exit(1);
    }
    gettimeofday(&start_tod,NULL);
    elapsed_tod = 0.0;

    record = true;

    changeRefTheta = false;
}


void potentialNavigationNAO::chooseCamera(int camera_flag){

    // ALVideoDeviceProxy::subscribe(const std::string& vmName, const int& resolution, const int& colorSpace, const int& fps)
    // vmName     – Name of the subscribing V.M.
    // resolution – Resolution requested. { 0 = kQQVGA, 1 = kQVGA, 2 = kVGA, 3 = k4VGA }
    // colorSpace – Colorspace requested. { 0 = kYuv, 9 = kYUV422, 10 = kYUV, 11 = kRGB, 12 = kHSY, 13 = kBGR }
    // fps        – Fps (frames per second) requested to the video source.

    cameraName = (camera_flag) ? ("bottom") : ("top");
    cameraName = cameraProxy.subscribeCamera(cameraName, camera_flag,  AL::kQVGA, AL::kYuvColorSpace, 30);
    cout << "Camera chosen: " << cameraName << endl;
    cout << "Frame rate: " << cameraProxy.getFrameRate(cameraName) << endl;
}




void potentialNavigationNAO::cleanAllActivities(){
    delete joy;
    delete motPtr;

    closeFiles();

    if(online)
        cameraProxy.unsubscribe(cameraName);

    if(SAVE_VIDEO && headset){
        AL::ALVideoRecorderProxy* recProxy = (AL::ALVideoRecorderProxy*)recorderProxy;
        recProxy->stopRecording();
        delete (AL::ALVideoRecorderProxy*)recorderProxy;
    }//*/

    motionProxy.stopMove();
    qi::os::sleep(2);

    cout << "Resetting initial posture ... " << endl;
    motionProxy.moveInit();


    cv::destroyAllWindows();

    vc.release();

    cout << "all active works have been stopped. " << endl ;

}


void potentialNavigationNAO::setTiltHead(char key){

    int dtilt = 0;
    int res;
    string cameraFrameName;

    // Capture image from subscribed camera
    if(online){
        ALimg = cameraProxy.getImageRemote(cameraName);
        OCVimage = workingImage(Rect(ROI_x,ROI_y,ROI_width,ROI_height));

    }
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


        updateCameraPose();
        //Get camera parameters
        /*cameraFrameName = (camera_flag) ? ("CameraBottom") : ("CameraTop");
        cameraFrame = motionProxy.getTransform(cameraFrameName,FRAME_ROBOT,true);
        camera_tilt = atan2(-cameraFrame.at(8),sqrt(cameraFrame.at(9)*cameraFrame.at(9) + cameraFrame.at(10)*cameraFrame.at(10)));
        camera_tilt = M_PI/2.0 - camera_tilt;
        camera_height = cameraFrame.at(11);
        cout << "camera_tilt: " << camera_tilt << endl;
        cout << "camera height: " << camera_height << endl;
        drive.set_tilt(camera_tilt);
        drive.set_cameraHeight(camera_height);
        drive.set_cameraPose(cameraFrame);//*/

        break;
    default:
        dtilt = 0;
        break;
    }

    updateTilt(dtilt);

    AL::ALValue names = AL::ALValue::array(PAN_JOINT,TILT_JOINT);
    AL::ALValue angles = AL::ALValue::array(pan,tilt_cmd);
    double fracSpeed = 0.2;

    motionProxy.setAngles(names,angles,fracSpeed);

    updateCameraPose();

    /*cameraFrameName = (camera_flag) ? ("CameraBottom") : ("CameraTop");
    cameraFrame = motionProxy.getTransform(cameraFrameName,FRAME_ROBOT,true);
    camera_tilt = atan2(-cameraFrame.at(8),sqrt(cameraFrame.at(9)*cameraFrame.at(9) + cameraFrame.at(10)*cameraFrame.at(10)));
    camera_tilt = M_PI/2.0 - camera_tilt;
    cout << "camera_tilt: " << camera_tilt*180.0/M_PI << endl;
    camera_height = cameraFrame.at(11);
    drive.set_tilt(camera_tilt);
    drive.set_cameraHeight(camera_height);
    drive.set_cameraPose(cameraFrame);//*/

}


void potentialNavigationNAO::enableRecording(){
    string folder_path = "/home/nao/recordings/cameras/NAO_potentialNavigation/video";
    AL::ALVideoRecorderProxy* recProxy = (AL::ALVideoRecorderProxy*)recorderProxy;

    recProxy->setColorSpace(11);//AL::kRGBColorSpace : buffer contains triplet on the format 0xBBGGRR, equivalent to three unsigned char
    recProxy->setResolution(1);//kQVGA
    recProxy->setVideoFormat("MJPG");
    recProxy->setFrameRate(30);
    recProxy->startRecording(folder_path,"NAOvideo",true);
}//*/

void potentialNavigationNAO::updateTilt(int dtilt){
    double delta = 0.1;
    tilt_cmd += (double)dtilt*delta;

    tilt_cmd = (tilt_cmd < MAXPITCH) ? ((tilt_cmd > MINPITCH) ? (tilt_cmd) : (MINPITCH)) : (MAXPITCH) ;


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
    cout << "AUTONOMOUS CONTROL" << endl ;

    if(!online){
        cv::namedWindow("Phantom image",WINDOW_AUTOSIZE);
    }

    double tic = getTickCount();

    std::thread regulateHeadYaw(&of_driving::applyPanCmdonNAOqi, &drive, &move_robot, &key);

    while(true){

        //motionProxy.move(-0.05,0.6,0.1);

        key = cv::waitKey(1);

        //Catch keyboard input to select the mode operation (<Manual>/<Autonomous + Stop/Go>)
        if(catchState(key)==-1)
            break;


        if(!headset){
            setTiltHead(key);
        }
        else{
            if(online){

                // capture image from subscribed camera
                ALimg = cameraProxy.getImageRemote(cameraName);
                const unsigned char* ptr = static_cast<const unsigned char*>(ALimg[6].GetBinary());
                workingImage.data = (uchar*)ptr;

                if(VREP_SIM){
                    flip(workingImage,workingImage,1);
                }
                OCVimage = workingImage(Rect(ROI_x,ROI_y,ROI_width,ROI_height));

                /// compute camera framerate
                current_imageTime = (int)ALimg[4]          // second
                                    + (int)ALimg[5] * 1e-6;  // usec to seconds


                //std::cout << "camera time interval: " << current_imageTime - previous_imageTime << std::endl;
                camera_rate = 1.0/( current_imageTime - previous_imageTime);

                previous_imageTime = current_imageTime;
                cameraRate_f << camera_rate << ";" << endl;//*/
            }
            else{

                frame_counter += 1;

                if (frame_counter == vc.get(CV_CAP_PROP_FRAME_COUNT)){
                  frame_counter = 1;
                  vc.set(CV_CAP_PROP_POS_FRAMES,0);
                }

                vc >> workingImage;
                resize(workingImage,workingImage,cv::Size(320,240));
                cvtColor(workingImage,workingImage,CV_BGR2GRAY);
                if(VREP_SIM){
                    flip(workingImage,workingImage,1);
                }
                OCVimage = workingImage(Rect(ROI_x,ROI_y,ROI_width,ROI_height));

            }


            /*if( online && (current_imageTime - previous_imageTime) >= 1e-4){


                camera_rate = 1.0/(current_imageTime - previous_imageTime);
                std::cout << "camera_rate: " << camera_rate << std::endl;
                previous_imageTime = current_imageTime;
                cameraRate_f << camera_rate << ";" << endl;//*/


                // update working images
                prev_img = img.clone();
                img = OCVimage.clone();
                //img.copyTo(prev_img);
                //OCVimage.copyTo(img);


                //Update needed variables for the current step
                updateTcAndLowPass();
                updateCameraPose();

                //make the algorithm run
                try{
                    drive.run(img,prev_img,SAVE_VIDEO,record,move_robot,changeRefTheta);
                }
                catch(...){
                    cerr << "Problem in drive.run. " << endl;
                    std::exit(1);
                }

                applyControlInputs(); // here wz is updated

                //command NAO
                if((move_robot) || (manual)){
                    //motionProxy.move(v,0.0f,w);//FRAME_ROBOT
                    drive.set_linearVel(vmax);
                    drive.set_angVel(wz);
                    //motionProxy.move(v,vy,wz);//FRAME_ROBOT
                    motionProxy.move(vx,vy,wz);//FRAME_ROBOT

                    //motionProxy.move(0.0,0.0,0.5);
                }
                else{
                    //drive.set_linearVel(0.0);
                    drive.set_angVel(0.0);
                    motionProxy.stopMove();
                }
                double toc = getTickCount();
                double tictoc = (toc - tic)/getTickFrequency();


                tic = toc;

                if(tictoc < 0.033){
                    sleep(0.033 - tictoc);
                }


                //cout << "tictoc: " << tictoc << endl;//*/
            //}
        }//*/

    }

    //drive.set_headYawRegulation(true);
    regulateHeadYaw.join();
    cleanAllActivities();

}

short int potentialNavigationNAO::catchState(char key){
    if((int)key == 27){
      cout << "[potentialNavigationNAO] Caught ESC! Waiting for closing/deleting running activities ... "<< endl;
      return -1;
    }
    else if(key=='g'){
        cout << "Go NAO!" << endl;
        record = true;
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
    else if(key=='c'){
        changeRefTheta = true;
    }
    return 1;
}

void potentialNavigationNAO::applyControlInputs(){
    //if manual then read joystick values
    if(manual){
        int res = joy->readEv(); //<--- don't print this value, or the joystick won't answer
        if(res != -1){
            if(jse->type & JS_EVENT_AXIS){
                if((int)jse->number == LEFT_UD_AXIS){//tilt down
                    v = -(jse->value)*vmax/(double)JOY_MAXVAL;
                    cout << "v: " << v << endl;
                }
                else if((int)jse->number == RIGHT_LR_AXIS){//tilt up
                    w = (jse->value)*wmax/(double)JOY_MAXVAL;
                    cout << "w: " << w << endl;
                }
            }
        }
    }
    else{
        //v = drive.get_linearVel(); //FRAME_ROBOT

        vx = drive.get_Vx(); //FRME_ROBOT
        vy = drive.get_Vy(); //FRAME_ROBOT
        wz = drive.get_Wz(); //FRAME_ROBOT//*/
        //w = -drive.get_angularVel(); //FRAME_ROBOT

    }
}

void potentialNavigationNAO::updateCameraPose(){
    string cameraFrameName = (camera_flag) ? ("CameraBottom") : ("CameraTop");
    string HeadYawFrameName = PAN_JOINT;
    cameraFrame = motionProxy.getTransform(cameraFrameName,FRAME_ROBOT,true);
    /*headYawFrame = motionProxy.getTransform(HeadYawFrameName,FRAME_ROBOT,false);

    std::cout << "cameraFrame: " << std::endl;
    std::cout << cameraFrame[0] << ", " << cameraFrame[1] << ", " << cameraFrame[2] << ", " << cameraFrame[3] << ", " << std::endl
              << cameraFrame[4] << ", " << cameraFrame[5] << ", " << cameraFrame[6] << ", " << cameraFrame[7] << ", " << std::endl
              << cameraFrame[8] << ", " << cameraFrame[9] << ", " << cameraFrame[10] << ", " << cameraFrame[11] << ", " << std::endl
              << cameraFrame[12] << ", " << cameraFrame[13] << ", " << cameraFrame[14] << ", " << cameraFrame[15] << ", " << std::endl;
    std::cout << "\nheadYawFrame: " << std::endl;
    std::cout << headYawFrame[0] << ", " << headYawFrame[1] << ", " << headYawFrame[2] << ", " << headYawFrame[3] << ", " << std::endl
              << headYawFrame[4] << ", " << headYawFrame[5] << ", " << headYawFrame[6] << ", " << headYawFrame[7] << ", " << std::endl
              << headYawFrame[8] << ", " << headYawFrame[9] << ", " << headYawFrame[10] << ", " << headYawFrame[11] << ", " << std::endl
              << headYawFrame[12] << ", " << headYawFrame[13] << ", " << headYawFrame[14] << ", " << headYawFrame[15] << ", " << std::endl << std::endl;//*/


    camera_tilt = atan2(-cameraFrame.at(8),sqrt(cameraFrame.at(9)*cameraFrame.at(9) + cameraFrame.at(10)*cameraFrame.at(10)));
    //std::cout << "camera_tilt [rad]: " << camera_tilt  << std::endl;
    camera_tilt = M_PI/2.0 - camera_tilt; //should be always 1.2° (39.7°) for top (bottom camera)
    camera_height = cameraFrame.at(11); //should be always 0.45831m for bottom camera
    //std::cout << "camera height: " << camera_height << std::endl;
    drive.set_cameraPose(cameraFrame);
    drive.set_tilt(camera_tilt);
    drive.set_cameraHeight(camera_height);

}

void potentialNavigationNAO::updateTcAndLowPass(){
    gettimeofday(&end_tod,NULL);

    elapsed_tod = (end_tod.tv_sec + (double)end_tod.tv_usec /1000000.0)
          - (start_tod.tv_sec + (double)start_tod.tv_usec/1000000.0);

    start_tod = end_tod;

    double loop_time;
    double now = getTickCount();
    loop_time = (now - curtime)/getTickFrequency();
    curtime = now;

    //std::cout << "Loop time: " << loop_time << std::endl;
    //cout << "Loop rate:    " << 1.0/loop_time << "Hz" << endl;

    cycle_f << 1.0/loop_time << "; " << endl;


    /*drive.setTc(elapsed_tod);
    drive.setImgLowPassFrequency(cutoff_f);
    drive.setBarLowPassFrequency(cutoff_f*fc_ratio*0.5);//*/

    //If under Ethernet connection, the framerate is around 15Hz


    //drive.setTc(0.04);//25Hz
    drive.setTc(1.0/25.0);
    drive.setImgLowPassFrequency(1.0);
    drive.setBarLowPassFrequency(1.0);

    /*cout << "elapsed_time: " << elapsed_tod << endl;
    cout << "freq: " << 1.0/elapsed_tod << endl << endl;
    /*cout << "image cut-off frequency: " << drive.getImgLowPassFrequency() << endl;
    cout << "control cut-off frequency: " << drive.getBarLowPassFrequency() << endl << endl;//*/

}

void potentialNavigationNAO::printTiltInfo(){

    string text_str;
    ostringstream convert;
    Size t_size, value_size;
    double font_scale = 0.9;
    Mat infoImg;
    //OCVimage.copyTo(infoImg);
    infoImg = OCVimage.clone();


    text_str = "";
    text_str = "TILT [rad]= ";
    t_size = getTextSize(text_str,1,font_scale,1,0);
    putText(infoImg, text_str,Point(10,infoImg.rows - 10 - t_size.height),1,font_scale,Scalar(255,255,255),1,CV_AA);

    text_str = "";
    convert.str(""); convert.clear();
    convert << setprecision(4) << tilt_cmd;
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
