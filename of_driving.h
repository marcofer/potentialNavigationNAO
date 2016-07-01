#ifndef DRIVING_H
#define DRIVING_H


#include "utils.h"

#include "parallel_process.h"

#define ANGULAR_VEL_MAX 0.1//0.83
#define LINEAR_VEL_MAX 0.06//0.0952


using namespace std;
using namespace cv;
using namespace cv::gpu;


class of_driving{
	
public:
	//Constructor
	of_driving();

	//Destructor
	~of_driving();

    AL::ALMotionProxy* motionPtr;
    inline void set_ALMotionPtr(AL::ALMotionProxy* ptr){motionPtr = ptr;}
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
    inline void set_linearVel(double v) {linear_vel = v;}
    inline void set_angVel(double w) {wz = w;}
    inline void set_realPan(double pan){real_pan = pan;}
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
    inline double get_Vx() {return vx;}
    inline double get_Vy() {return vy;}
    inline double get_Wz() {return wz;}
    inline double get_u_pan() {return u_pan;}
    inline double get_angVelMax() {return max_w;}
    inline double get_throttle() {return ankle_angle;}
    inline double get_theta() {return theta;}
    inline Matx21f get_NavVec() {return p_bar;}


    void applyPanCmdonNAOqi();
    double getRealPanFromNAOqi();

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
    Eigen::Matrix<double,6,6> W;
    Eigen::Matrix4d cameraPose;
    double vx, vy, wz;
    double pan_dot;
    double u_pan, u_pan_old, real_pan;
    std::vector<float> headYawFrame;

	int area_ths;

        bool open_close;

    //Centroids variables
    vector < vector < cv::Point > > contours, good_contours, ground_contours;
    vector < Vec4i > hierarchy;
    vector < Point2f > centroids, l_centroids, r_centroids;
    Point2f x_r, x_l;
    Point2f prevx_r, prevx_l;
    Point2f old_xr, old_xl;
    double delta;
    int delta_int;
    Mat motImg;


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
             J_f, pan_f;


    /* Methods */


	void computeOpticalFlowField(Mat&, Mat&);

	void estimateAffineCoefficients(bool,Mat&,Mat&,Rect&);
	void buildPlanarFlowAndDominantPlane(Mat&);

    void extractPlaneBoundaries();
    void findGroundBoundaries();
    void buildMotionImage();
    void computeCentroids();
    void velocityScaling();

    void computeGradientVectorField();
	void computePotentialField();
	void computeControlForceOrientation();

    Eigen::MatrixXd buildInteractionMatrix(double, double);

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


#endif // DRIVING_H
