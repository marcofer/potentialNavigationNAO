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

using namespace cv;

void arrowedLine2(Mat&, cv::Point2f, cv::Point2f, const Scalar&, int thickness, int line_type, int shift, double tipLength);

double low_pass_filter(double in, double out_old, double Tc, double tau);
double high_pass_filter(double in, double in_prev, double out_old, double Tc, double tau);
void getFlowField(const Mat& u, const Mat& v, Mat& flowField);

string type2str(int);

void callbackButton(int,void*);


