#include "math.h"
#include <sys/time.h>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv/cv.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Householder>
#include <eigen3/Eigen/LU>
#include <eigen3/Eigen/QR>//*/


// Aldebaran includes
#include <alcommon/almodule.h>
#include <alcommon/albroker.h>
#include <alproxies/almotionproxy.h>
#include <alproxies/alvideodeviceproxy.h>
#include <alproxies/alvideorecorderproxy.h>
#include <alerror/alerror.h>
#include <alvision/alvisiondefinitions.h>
#include <qi/os.hpp>


#include <boost/bind.hpp>
#include <thread>


#define VREP_SIM true

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
//#define MINYAW -2.0856685
//#define MAXYAW 2.0856685
#define MINYAW -2.07694//-1.59//-1.59/2.0
#define MAXYAW 2.07694//1.59//1.59/2.0

#define SAVE_VIDEO false
#define FRAME_ROBOT 2

using namespace cv;

void arrowedLine2(Mat&, cv::Point2f, cv::Point2f, const Scalar&, int thickness, int line_type, int shift, double tipLength);

double low_pass_filter(double in, double out_old, double Tc, double tau);
double high_pass_filter(double in, double in_prev, double out_old, double Tc, double tau);
void getFlowField(const Mat& u, const Mat& v, Mat& flowField);

string type2str(int);

void callbackButton(int,void*);


vector<Point> contoursConvexHull( vector<vector<Point> >  );
