#ifndef DRIVING_H
#define DRIVING_H

//System includes
#include <iostream>
#include <string.h>
#include <fstream>

//Opencv includes
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
#include "opencv2/nonfree/nonfree.hpp"

//Eigen includes
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Householder>
#include <eigen3/Eigen/LU>
//#include <eigen3/Eigen/QR>//*/
#include <eigen3/Eigen/SVD>


#include <boost/math/tools/config.hpp>

#include "parallel_process.h"

#define ANGULAR_VEL_MAX 0.1//0.83
#define LINEAR_VEL_MAX 0.04//0.0952

using namespace std;
using namespace cv;
using namespace cv::gpu;


class of_driving{
	
public:
	//Constructor
	of_driving();

	//Destructor
	~of_driving();

	//Set the size of the image used by the image processing algorithm
	void set_imgSize(int w, int h);

	//initialization of the sampling time
	inline void setTc(double T_c) {Tc = T_c;}
    inline void setImgLowPassFrequency(double f){img_lowpass_freq = f;}
    inline void setBarLowPassFrequency(double f){bar_lowpass_freq = f;}
    inline double getImgLowPassFrequency(){return img_lowpass_freq;}
    inline double getBarLowPassFrequency(){return bar_lowpass_freq;}

    inline void set_tilt(double tilt) {camera_tilt = tilt;}
    inline void set_cameraHeight(double h) {camera_height = h;}


    void set_cameraPose(std::vector<float>);


    //run function - Real time image processing algorithm
    void run(Mat& img, Mat& prev_img, bool, bool);

	void setRectHeight(int rect_cmd);
	//Print on the image he information about the current pan and tilt angles of the camera
	void plotPanTiltInfo(Mat& img, float tilt_cmd, float pan_cmd);

	//get functions
	inline double get_steering() {return - steering;}
	inline double get_tilt() {return tilt_angle;}
	inline double get_pan() {return pan_angle;}
    inline double get_linearVel() {return linear_vel;}
    inline double get_angularVel() {return angular_vel;}
    inline double get_linVelMax() {return linear_vel;}
    inline double get_Vy() {return vy;}
    inline double get_Wz() {return wz;}
    inline double get_angVelMax() {return max_w;}
    inline double get_throttle() {return ankle_angle;}
    inline double get_theta() {return theta;}
    inline Matx21f get_NavVec() {return p_bar;}

    void createWindowAndTracks();
	//the control variable: steering velocity (or angular velocity?)
	double R;

	//initialize flows
    void initFlows(bool);

    void openFiles(const string);
    void closeFiles();


private:

    //Camera calibration matrix
    double focal_length;
    double camera_height;

    bool record;

    cv::Point2f principal_point;
    Eigen::Matrix3d K, Kinv, cameraR;
    Eigen::Matrix<double,3,1> cameraT;
    Eigen::Matrix4d cameraPose;
    Eigen::Matrix<double,1,6> Lx_l, Lx_r;//*/
    double vy, wz;

	int area_ths;

        bool open_close;

    //Centroids variables
    vector < vector < cv::Point > > contours, good_contours;
    vector < Vec4i > cannyHierarchy;
    vector < Point2f > centroids, l_centroids, r_centroids;
    Point2f x_r, x_l;
    Point2f prevx_r, prevx_l;
    Point2f old_xr, old_xl;

	Mat H;

	int dp_threshold;

	//Mat ROI_ransac;
	//Rect rect_ransac;
	double ROI_width;
	double ROI_height;
	double ROI_x ;
	double ROI_y;

	double RANSAC_imgPercent;


	//Optical Flow algorithm flag
	int of_alg;
	string of_alg_name;

	Mat image;
	//Image size
	int img_height, img_width;

	//Pair of images
	Mat GrayImg, GrayPrevImg;

	//Sampling time
	double Tc;

	//Maximum angular velocity
    double max_w;
    //Maximum linear velocity
    double max_v;
	//Old angular velocity value (for low-pass filtering)
	double Rold;

	double px_old, py_old;

	double publishing_rate;

	//linear velocity
    double linear_vel;
    double angular_vel;

	double img_lowpass_freq;
	double bar_lowpass_freq;

	//Car control variables
	double steering;
	double ankle_angle;
	double tilt_angle;
	double pan_angle;
	double camera_tilt;

	bool camera_set;

	//number of layer for multiresolution pyramid transform
	int maxLayer;

	//gradient scale factor
	double grad_scale;

    double field_weight;
    int weight_int;

	//erode/dilate scale factor
	double open_erode, open_dilate, close_erode, close_dilate;
	int open_erode_int, open_dilate_int, close_erode_int, close_dilate_int;

	//optical flow field
    Mat optical_flow, old_flow;
    Mat noFilt_of;
    Mat atanMat;

	//planar flow field
    Mat planar_flow;
    Mat noFilt_pf;


	//dominant flow field
	Mat dominant_plane, best_plane, old_plane;
    Mat noFilt_dp, noFilt_best;

    Mat smoothed_plane;
    Mat noFilt_sp;

	//Dominant plane convex hull
	Mat dominantHull;

	//Gradient vector field
    Mat gradient_field;
    Mat vortex_field;
    Mat result_field, noFilt_rf;
    Mat noFilt_gf, noFilt_vf;
    Mat inverted_dp;

	//Potential Field
	Mat potential_field;

	//Control Force 
	Matx21f p_bar;
    Matx21f noFilt_pbar;
	//Navigation angle
    double theta, theta_old;

	//Affine Coefficients
    Matx22f A, nf_A;
    Matx21f b, nf_b;
    Mat Ab, noFilt_Ab;

	bool control;

	//window size of the optical flow algorithm
	int windows_size;

	//Display flow resolution
	int flowResolution;

	//RANSAC inliers counter
	int point_counter, best_counter;
    int nf_point_counter, nf_best_counter;

	//RANSAC iterations
	int iteration_num;

	//Farneback algorithm iterations
	int of_iterations;
	double pyr_scale;
	int pyr_scale10;

	//RANSAC terminate condition
	double max_counter;

	//Threshold for flows comparison
	double epsilon;
	int eps_int;

	//Vehicle wheelbase
	double wheelbase;


	/** Variable used for video recording **/
    VideoWriter record_total;

	//flag to activate fake black corners
	bool fake_corners;

	//scale factor for optcal flow
	int of_scale;

    ofstream nofilt_barFile, filt_barFile, theta_f, angularVel_f, error_f, xl_f, xr_f, centr_w_f, R_f, vx_f, vy_f, wz_f, det_f, Ju_f,
             J_f;


    /* Methods */

	void computeOpticalFlowField(Mat&, Mat&);

	void estimateAffineCoefficients(bool,Mat&,Mat&,Rect&);
	void buildPlanarFlowAndDominantPlane(Mat&);

    void computeCentroids();
    void velocityScaling();

    void computeGradientVectorField();
	void computePotentialField();
	void computeControlForceOrientation();


    void computeRobotVelocities();
    void computeFlowDirection();

    Mat displayImages(Mat&);

    void displayImagesWithName(Mat&, string name);
	
    clock_t start, end;
    timeval time_tod, start_plot;


    //*** GPU BROX OPTICAL FLOW ***/
	double scale;
	int scale_int;
	double alpha;
	int alpha_int;
	int gamma;
	int inner;
	int outer;
	int solver;
	//*** ***/

	int cores_num;


};

void arrowedLine2(Mat&, cv::Point2f, cv::Point2f, const Scalar&, int thickness, int line_type, int shift, double tipLength);

double low_pass_filter(double in, double out_old, double Tc, double tau);
double high_pass_filter(double in, double in_prev, double out_old, double Tc, double tau);
void getFlowField(const Mat& u, const Mat& v, Mat& flowField);

string type2str(int);

void callbackButton(int,void*);

template <typename T> inline T clamp (T x, T a, T b);


#endif // DRIVING_H
