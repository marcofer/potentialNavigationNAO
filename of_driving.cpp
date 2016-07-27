#include "of_driving.h"

#include "math.h"
#include <sys/time.h>
#include <fstream>

using namespace cv::gpu;

enum OF_ALG{LK,FARNEBACK,GPU_LK_SPARSE,GPU_LK_DENSE,GPU_FARNEBACK,GPU_BROX};

of_driving::of_driving(){

    /*max_w = ANGULAR_VEL_MAX/8.0;
    max_v = LINEAR_VEL_MAX/2.0;
    linear_vel = LINEAR_VEL_MAX/2.0;//*/
    max_w = ANGULAR_VEL_MAX;
    max_v = LINEAR_VEL_MAX;
    linear_vel = LINEAR_VEL_MAX;//*/

    if(VREP_SIM){
        grad_scale = 1.0;
        windows_size = 13.0;//7.0;//13.0;//44.0;//13.0;
        maxLayer = 2;

        //epsilon = 1.0;//0.7;//0.5//0.4;//0.8;
        epsilon = 0.15;//.15;//0.7;

        flowResolution = 4;
        iteration_num = 10;
        of_iterations = 3;//3;

        open_erode = 4.0;//4.0;//4.0//1.0;//4.9;
        open_dilate = 1.0;//1.0;//1.0;
        close_erode = 1.0;//1.0;
        close_dilate = 1.0;//1.0;
        of_alg = 3;
        of_scale = 1;
        RANSAC_imgPercent = 0.1;

        dp_threshold = 80;//80;//80;
        area_ths = 20;//100;//400;

        xrmin = 100;
        xrmax = 120;
        low_min_narrow_ths = 90;//90;
        low_max_narrow_ths = 130;
        high_min_narrow_ths = 110;//110;
        high_max_narrow_ths = 140;



    }
    else{
        grad_scale = 1.0;
        windows_size = 13.0;//44.0;//13.0;
        maxLayer = 2;

        epsilon = 0.7;//0.7;//0.5//0.4;//0.8;

        flowResolution = 4;
        iteration_num = 10;
        of_iterations = 3;//3;

        open_erode = 1.0;//4.9;
        open_dilate = 7.0;
        close_erode = 1.0;
        close_dilate = 1.0;
        of_alg = 3;
        of_scale = 1;
        RANSAC_imgPercent = 0.1;

        dp_threshold = 60;//40
        area_ths = 200;

        xrmin = 110;
        xrmax = 130;

        low_min_narrow_ths = 100;
        low_max_narrow_ths = 110;
        high_min_narrow_ths = 110;
        high_max_narrow_ths = 120;


    }



    wheelbase = 2.06;

    open_erode_int = open_erode*10.0;
	close_erode_int = close_erode*10.0;
	open_dilate_int = open_dilate*10.0;
	close_dilate_int = close_dilate*10.0;
    
	Rold = 0.0;
	px_old = 0.0;
	py_old = 0.0;


	angular_vel = 0.0;
    vx = 0.0;
    vy = 0.0;
    wz = 0.0;
    pan_dot = 0.0;
    u_pan = 0.0;
    u_pan_old = 0.0;
    theta_des = 0.0;

	steering = 0.0;
	ankle_angle = 0.0;
	pan_angle = 0.0;
	tilt_angle = 0.0;

	camera_set = false;

	A = Matx22f(1,0,0,1);
	b = Matx21f(0,0);

	p_bar = Matx21f(0,0);

	theta = 0.0;
	theta_old = 0.0;

    field_weight = 0.0; //0.0 = gradient field ; 1.0 = vortex field
    weight_int = ((double)field_weight)*10;

	gettimeofday(&start_plot,NULL);

	cores_num = sysconf( _SC_NPROCESSORS_ONLN );

    record = false;

    headYawRegulation = false;

    gettimeofday(&start_t,0);

}

of_driving::~of_driving(){
	destroyAllWindows();

}

void of_driving::set_imgSize(int w, int h, int roi_x, int roi_y){
	img_width = w;
	img_height = h;
    real_roiy = roi_x;
    real_roiy = roi_y;
}

void of_driving::initFlows(bool save_video){
    optical_flow = Mat::zeros(img_height,img_width,CV_32FC2);
    old_flow = Mat::zeros(img_height,img_width,CV_32FC2);
	dominant_plane = Mat::zeros(img_height,img_width,CV_8UC1);
	old_plane = Mat::zeros(img_height,img_width,CV_8UC1);
	smoothed_plane = Mat::zeros(img_height,img_width,CV_8UC1);
	dominantHull = Mat::zeros( img_height, img_width, CV_8UC1 );
	best_plane = Mat::zeros(img_height,img_width,CV_8UC1);
	planar_flow = Mat::zeros(img_height,img_width,CV_32FC2);
    gradient_field = Mat::zeros(img_height,img_width,CV_32FC2);
    vortex_field = Mat::zeros(img_height,img_width,CV_32FC2);
    result_field = Mat::zeros(img_height,img_width,CV_32FC2);
    potential_field = Mat::zeros(img_height,img_width,CV_32FC2);
    inverted_dp = Mat::zeros(img_height,img_width,CV_8UC1);
    motImg = Mat::zeros(img_height,img_width,CV_32FC2);

    noFilt_of = Mat::zeros(img_height,img_width,CV_32FC2);
    noFilt_pf = Mat::zeros(img_height,img_width,CV_32FC2);
    noFilt_dp = Mat::zeros(img_height,img_width,CV_8UC1);
    noFilt_best = Mat::zeros(img_height,img_width,CV_8UC1);
    noFilt_sp = Mat::zeros(img_height,img_width,CV_8UC1);
    noFilt_gf = Mat::zeros(img_height,img_width,CV_32FC2);
    noFilt_vf = Mat::zeros(img_height,img_width,CV_32FC2);
    noFilt_rf = Mat::zeros(img_height,img_width,CV_32FC2);


    point_counter = 0;
    prev_counter = 0;
	best_counter = 0;
    nf_point_counter = 0;
    nf_best_counter = 0;

    maxLeftPoint = Point2f(0,0);
    minRightPoint = Point2f(0,0);
    prevLeftPoint = Point2f(0,0);
    prevRightPoint = Point2f(0,0);
    narrow_width = 0.0;
    old_narrow_width = 0.0;



    x_r = Point2f(img_width,img_height/2);
    x_l = Point2f(0,img_height/2);

    vr = Point2f(0,0);
    vl = Point2f(0,0);
    vr_old = Point2f(0,0);
    vl_old = Point2f(0,0);

    acc_r = Point2f(0,0);
    acc_l = Point2f(0,0);

    slow_xr = Point2f(img_width,img_height/2);
    slow_xl = Point2f(0,img_height/2);
    slow_vr = Point2f(0,0);
    slow_vl = Point2f(0,0);
    slow_oldxr = Point2f(img_width,img_height/2);
    slow_oldxl = Point2f(0,img_height/2);
    slow_oldvr = Point2f(0,0);
    slow_oldvl = Point2f(0,0);


    prevx_r = x_r; //maybe useless
    prevx_l = x_l; //maybe useless

    old_xr = Point2f(img_width,img_height/2);
    old_xl = Point2f(0,img_height/2);

    old2_xr = Point2f(img_width,img_height/2);
    old2_xl = Point2f(0,img_height/2);

    px_margin = 30;

    double fov = 0.8315; //47.64°, from the documentation
    focal_length = ((img_height + real_roiy)/2)/tan(fov/2);
    std::cout << "FOCAL LENGTH: " << focal_length << std::endl;
    principal_point = cv::Point2f((img_width + real_roix)/2,(img_height + real_roiy)/2);
    K << focal_length, 0, principal_point.x,
         0, focal_length, principal_point.y,
         0,   0,   1;

    //Kinv = K.inverse();//*/

    /*double ROI_width = img_width/2.0;
    double ROI_height = img_height/2.0;
    double ROI_x = img_width/4.0;
    double ROI_y = img_height/4.0;//*/

    ROI_width = img_width;
    //ROI_height = img_height/2;
    ROI_height = img_height;
	
    //ROI_x = 0;
    ROI_x = 0;//*/
	ROI_y = 0;//*/
    //ROI_y = 0;//*/

    /*double ROI_width = img_width;
    double ROI_height = img_height/4.0;
    double ROI_x = 0.0;
    double ROI_y = 3.0*img_height/4.0;//*/

    //max_counter = img_height*img_width/2;
    max_counter = ROI_height*ROI_width*RANSAC_imgPercent;
    //rect_ransac = Rect(ROI_x,ROI_y,ROI_width,ROI_height);
    //ROI_ransac = dominant_plane(rect_ransac);

    if(save_video){
        record_total.open("total.avi", CV_FOURCC('D','I','V','X'),15.0, cvSize(3*img_width,2*img_height), true);
        if( !record_total.isOpened() ) {
            cout << "VideoWriter failed to open!" << endl;
            }
    }//*/


    /*** GPU BROX OPTICAL FLOW ***/
    scale = 0.5;
    scale_int = 500;
    alpha_int = 250;
    alpha = 0.25;//0.197; //flow smoothness
    gamma = 38;//50.0 // gradient constancy importance
    inner = 7;//10; //number of lagged non-linearity iterations (inner loop)
    outer = 3; //number of warping iterations (number of pyramid levels)
    solver = 5;//10;

    pyr_scale = 0.5;
    pyr_scale10 = pyr_scale*10;

    eps_int = epsilon*100.0;

    delta = 0.5;
    delta_int = delta*100.0;

    switch(of_alg){
        case LK:
            cout << "LK" << endl;
            of_alg_name = "Lucas-Kanade Optical Flow";
            namedWindow(of_alg_name,WINDOW_AUTOSIZE);
            break;
        case FARNEBACK:
            cout << "Farneback" << endl;
            of_alg_name = "Farneback Optical Flow";
            break;
        case GPU_LK_SPARSE:
            cout << "GPU LK sparse" << endl;
            of_alg_name = "GPU Sparse Lucas-Kanade Optical Flow";
            break;//*/
        case GPU_LK_DENSE:
            cout << "GPU LK dense" << endl;
            of_alg_name = "GPU Dense Lucas-Kanade Optical Flow";
            break;
        case GPU_FARNEBACK:
            cout << "GPU Farneback" << endl;
            of_alg_name = "GPU Farneback Optical Flow";
            break;
        case GPU_BROX:
            cout << "GPU Brox" << endl;
            of_alg_name = "GPU Brox Optical Flow";
            break;
        default:
            break;
    }//*/



    open_close = 0;

}

void of_driving::createWindowAndTracks(){
    namedWindow(of_alg_name,WINDOW_AUTOSIZE);

    //createButton("OpenClose",callbackButton,&open_close,CV_CHECKBOX,0);


    if(DEBUG){
        createTrackbar("winSize",of_alg_name,&windows_size,51,NULL);//*/

        /*if(of_alg != LK && of_alg != GPU_BROX){
            createTrackbar("iters",of_alg_name,&of_iterations,50,NULL);
        }
        else if(of_alg == GPU_BROX){
            createTrackbar("gamma",of_alg_name,&gamma,50,NULL);
            createTrackbar("alpha",of_alg_name,&alpha_int,1000,NULL);
            createTrackbar("inner",of_alg_name,&inner,50,NULL);
            createTrackbar("outer",of_alg_name,&outer,5,NULL);
            createTrackbar("solver",of_alg_name,&solver,50,NULL);
        }//*/

        //createTrackbar("pyr levels",of_alg_name,&maxLayer,5,NULL);
        createTrackbar("epsilon*100",of_alg_name,&eps_int,500,NULL);
        //createTrackbar("of_scale",of_alg_name,&of_scale,10,NULL);
        //createTrackbar("maxLevel",of_alg_name,&maxLayer,6,NULL);
        //createTrackbar("pyr_scale*10",of_alg_name,&pyr_scale10,10,NULL);
        createTrackbar("dp_threshold",of_alg_name,&dp_threshold,255,NULL);
        createTrackbar("O_erode*10",of_alg_name,&open_erode_int,200,NULL);
        createTrackbar("O_dilat*10",of_alg_name,&open_dilate_int,200,NULL);
        createTrackbar("C_dilat*10",of_alg_name,&close_dilate_int,200,NULL);
        createTrackbar("C_erode*10",of_alg_name,&close_erode_int,200,NULL);
        //createTrackbar("field weights",of_alg_name,&weight_int,10,NULL);

        createTrackbar("area_ths",of_alg_name,&area_ths,500,NULL);
        //createTrackbar("delta*100",of_alg_name,&delta_int,500,NULL);//*/
        //createTrackbar("xrmin",of_alg_name,&xrmin,160,NULL);
        //createTrackbar("xrmax",of_alg_name,&xrmax,160,NULL);//*/

        createTrackbar("low_min_narrow",of_alg_name,&low_min_narrow_ths,160,NULL);
        createTrackbar("low_max_narrow",of_alg_name,&low_max_narrow_ths,160,NULL);
        createTrackbar("high_min_narrow",of_alg_name,&high_min_narrow_ths,200,NULL);
        createTrackbar("high_max_narrow",of_alg_name,&high_max_narrow_ths,200,NULL);//*/
    }
}


void callbackButton(int state, void* data){

	bool* var = reinterpret_cast<bool*>(data);
	*var = !(*var);
	cout << "Performing " << ( (*var) ? ("Opening -> Closing") : ("Closing -> Opening") )<< endl << endl; 
}

void of_driving::setRectHeight(int rect_cmd){
    float delta = 1.0;

    ROI_y += delta*rect_cmd;

    ROI_y = ((ROI_y > 0) ? ( (ROI_y <= img_height - ROI_height -1 ) ? (ROI_y) : (img_height - ROI_height -1)  ) : (0));

}




void of_driving::plotPanTiltInfo(Mat& img, float tilt_cmd, float pan_cmd){

    string text_str;
    ostringstream convert;
    Size text_size;

    text_str = "";
    text_str = "TILT (rad)";
    text_size = getTextSize(text_str,1,1,1,0);
    putText(img, text_str,Point(img_width-150,img_height - 40 + 0.5*text_size.height),1,1,Scalar(255,255,255),1,CV_AA);


    text_str = "";
    convert << setprecision(4) << tilt_angle;
    text_str = convert.str();
    putText(img,text_str,Point(img_width-150+text_size.width+10,img_height-40+0.5*text_size.height),1,1,Scalar(255,255,255),1,CV_AA);

    text_str = "";
    text_str = "PAN (rad)";
    text_size = getTextSize(text_str,1,1,1,0);
    putText(img,text_str,Point(img_width-150,img_height-20+0.5*text_size.height),1,1,Scalar(255,255,255),1,CV_AA);

    text_str = "";
    convert.str(""); convert.clear();
    convert << setprecision(4) << pan_angle;
    text_str = convert.str();
    putText(img,text_str,Point(img_width-150+text_size.width+10,img_height-20+0.5*text_size.height),1,1,Scalar(255,255,255),1,CV_AA);


    if(tilt_cmd==1){
        circle(img,Point(img_width-180,img_height-40),2,Scalar(0,0,255),2,CV_AA);
    }else if(tilt_cmd==-1){
        circle(img,Point(img_width-180,img_height-20),2,Scalar(0,0,255),2,CV_AA);
    }

    if(pan_cmd==1){
        circle(img,Point(img_width-190,img_height-30),2,Scalar(0,0,255),2,CV_AA);
    }else if(pan_cmd==-1){
        circle(img,Point(img_width-170,img_height-30),2,Scalar(0,0,255),2,CV_AA);
    }


}

void of_driving::run(Mat& img, Mat& prev_img, bool save_video, bool rec, bool move_robot, bool narrowCheck){

    record = rec;

	Rect rect_ransac(ROI_x,ROI_y,ROI_width,ROI_height);
	Mat ROI_ransac = dominant_plane(rect_ransac);

    epsilon = (double)eps_int/100.0;
    delta = (double)delta_int/100.0;
    alpha = (double)alpha_int/1000.0;
	pyr_scale = (double)pyr_scale10/10.0;
	open_erode = (double)open_erode_int/10.0;
	open_dilate = (double)open_dilate_int/10.0;
	close_erode = (double)close_erode_int/10.0;
	close_dilate = (double)close_dilate_int/10.0;
    field_weight = (double)weight_int/10.0;
	epsilon *= of_scale;


	if(windows_size%2 == 0){
        windows_size += 1;
	}

	int k = 0;
	best_counter = 0;

    //prev_img.copyTo(image);

    //cvtColor(img,GrayImg,CV_BGR2GRAY);
    //cvtColor(prev_img,GrayPrevImg,CV_BGR2GRAY);



    GrayImg = img.clone();
    GrayPrevImg = prev_img.clone();

    //img.copyTo(GrayImg);
    //prev_img.copyTo(GrayPrevImg);

    //blur(GrayImg,GrayImg,Size(windows_size*2,windows_size*2));
    //blur(GrayPrevImg,GrayPrevImg,Size(windows_size*2,windows_size*2));

    const int64 start = getTickCount();

	/// ---  1. Compute the optical flow field u(x,y,t) (output: optical_flow matrix)
    computeOpticalFlowField(GrayPrevImg,GrayImg);

    //buildMotionImage();

    /*parallel_for_(Range(0,cores_num),ParallelDominantPlaneFromMotion(cores_num,dominant_plane,old_plane,optical_flow,motImg,
                                                                     principal_point,epsilon,Tc,img_lowpass_freq,linear_vel,
                                                                     wz,camera_height,focal_length,camera_tilt,dp_threshold,W));

    cv::bitwise_not(dominant_plane,inverted_dp);

    //*/

    while(point_counter <= max_counter && k < iteration_num){


        /// --- 2. Compute affine coefficients by random selection of three points
		estimateAffineCoefficients(false,GrayPrevImg,ROI_ransac,rect_ransac);
		

        /// --- 3-4. Estimate planar flow from affine coefficients and Match the computed optical flow and esitmated planar flow, so detect the dominant plane. If the dominant plane occupies
        /// less than half of the image, then go to step (2)
		buildPlanarFlowAndDominantPlane(ROI_ransac);
		
        
		if(point_counter >= best_counter){
			best_counter = point_counter;
            //dominant_plane.copyTo(best_plane);
            best_plane = dominant_plane.clone();

		}

        if(nf_point_counter >= nf_best_counter){
            nf_best_counter = point_counter;
            noFilt_dp.copyTo(noFilt_best);
        }

		k++;
	}

	if(point_counter <= max_counter){
        //best_plane.copyTo(dominant_plane);
        noFilt_best.copyTo(noFilt_dp);

        dominant_plane = best_plane.clone();
	}

    float vel_counter = (point_counter - prev_counter)/Tc;
    if(record){
        plane_area_f << point_counter << "," << vel_counter << "; " << std::endl;
    }

    prev_counter = point_counter;

    /// --- 2. Robust computation of affine coefficients with all the points belonging to the detected dominant plane 	
    estimateAffineCoefficients(true,GrayPrevImg,ROI_ransac,rect_ransac);//*/

    /// --- 3-4. Estimate planar flow from affine coefficients and Match the computed optical flow and esitmated planar flow, so detect the dominant plane
    buildPlanarFlowAndDominantPlane(ROI_ransac);

    /// --- 5. Extract dominant plane boundary
    extractPlaneBoundaries();
    //findGroundBoundaries();

    /// --- 6a. Compute gradient vector field from dominant plane
    //computeGradientVectorField();

    /// --- 6b. Compute the obstacle centroids
    computeCentroids();

    //// --- 7. Compute the control force as average of potential field
    //computeControlForceOrientation();

    /// --- 8. Compute the translational and rotational robot velocities
    computeRobotVelocities(move_robot,narrowCheck);
	
    /// --- END. Show the intermediate steps
	Mat total = Mat::zeros(2*img_height,3*img_width,CV_8UC3);
	

	/*** MULTI-THREADED DISPLAY ***/
    parallel_for_(Range(0,6),ParallelDisplayImages(6,flowResolution,GrayPrevImg,optical_flow,planar_flow,dominant_plane,smoothed_plane,
                                                   result_field,p_bar,vx,angular_vel,total,rect_ransac, u_pan, real_pan, max_v, max_w, vy, wz,
                                                   l_centroids,r_centroids,x_l,x_r,good_contours,xrmin,xrmax, theta_des, px_margin, maxLeftPoint,
                                                   minRightPoint, narrowCheck));
    if(save_video){
        record_total.write(total);
    }//*/

	/*** SINGLE-THREADED DISPLAY ***/
	//total = displayImages(prev_img);//*/
	

    imshow(of_alg_name,total);
    cvWaitKey(1);//*/

}

void of_driving::computeOpticalFlowField(Mat& prevImg, Mat& img){

	/** LK VARIABLES **/
	vector<Mat> pyr, prev_pyr;
	vector<Point2f> prevPts, nextPts;//, nextPts;
	Size winSize(windows_size,windows_size); 
	Mat status, err;
	double pyr_scale = 0.5;
    int sampled_i = img_height/flowResolution;
    int sampled_j = img_width/flowResolution;
	bool withDerivatives = false;
	int pyr_idx, incr;
	(withDerivatives) ? (pyr_idx = maxLayer * 2) : (pyr_idx = maxLayer) ;
	(withDerivatives) ? (incr = 2) : (incr =  1) ;	


    if(of_alg == LK || of_alg == GPU_LK_SPARSE){
        //pick samples from the first image to compute sparse optical flow
        for (int i = 0 ; i < img.rows ; i += flowResolution){
            for (int j = 0 ; j < img.cols ; j += flowResolution){
                prevPts.push_back(Point2f(j,i));
            }
        }
    }


	Mat prevPtsMat, nextPtsMat;
	/*prevPtsMat.create(1,prevPts.size(),CV_32FC2);

	prevPtsMat = Mat(prevPts);

	resize(prevPtsMat,prevPtsMat,cv::Size(prevPts.size(),1));

	nextPtsMat.create(1,prevPts.size(),CV_32FC2);

	int idx_sample = sampled_i/2 * sampled_j + sampled_j/2 ;		    		
    /***/

    /** FARNEBACK VARIABLES **/
	int winsize = windows_size;
	int poly_n = 5;
	double poly_sigma = 1.2;
	int flags = OPTFLOW_FARNEBACK_GAUSSIAN;//OPTFLOW_USE_INITIAL_FLOW;//OPTFLOW_FARNEBACK_GAUSSIAN
	/***/

	/*** GPU METHODS VARIABLES ***/
	GpuMat u_flow, v_flow;
	Mat img32F, prevImg32F;
	GpuMat gpu_status;
	GpuMat* gpu_err;


	GpuMat gpuPrevPts(prevPtsMat);
	GpuMat gpuNextPts(nextPtsMat);

    //img.convertTo(img32F,CV_32F,1.0/255.0);
    //prevImg.convertTo(prevImg32F,CV_32F,1.0/255.0);
    //GpuMat gpuImg32F(img32F);
    //GpuMat gpuPrevImg32F(prevImg32F);//*/
	GpuMat gpuImg8U(img);
	GpuMat gpuPrevImg8U(prevImg);

    //gpu::BroxOpticalFlow brox_flow(alpha,gamma,scale,inner,outer,solver);
	gpu::PyrLKOpticalFlow sparseLK_flow, denseLK_flow;
	gpu::FarnebackOpticalFlow farneback_flow;//*/
    gpu::FastOpticalFlowBM fastBM;
	GpuMat buf;//*/



	/*gpu::FarnebackOpticalFlow farneback_flow;
	farneback_flow.numLevels = maxLayer;
	farneback_flow.pyrScale = pyr_scale;
	farneback_flow.winSize = windows_size;
	farneback_flow.numIters = of_iterations;
	//farneback_flow.flags = OPTFLOW_USE_INITIAL_FLOW;
	


	/*** CPU SINGLE-THREADED FARNEBACK ***/
	//calcOpticalFlowFarneback(prevImg, img, optical_flow, pyr_scale, maxLayer, winsize, of_iterations, poly_n, poly_sigma, flags);		

	/*** GPU SINGLE-THREADED FARNEBACK ***/
	/*farneback_flow(gpuPrevImg8U,gpuImg8U,u_flow,v_flow);
	getFlowField(Mat(u_flow),Mat(v_flow),optical_flow);//*/

	/*** CPU MULTI-THREADED FARNEBACK ***/
	//parallel_for_(Range(0,cores_num),ParallelOpticalFlow(cores_num,prevImg,img,optical_flow,pyr_scale,winsize,maxLayer,of_iterations,poly_n,poly_sigma,flags));

	/*** GPU MULTI-THREADED FARNEBACK ***/
	//parallel_for_(Range(0,2),ParallelOpticalFlow(2,farneback_flow,gpuPrevImg8U,gpuImg8U,u_flow,v_flow,optical_flow));

	switch(of_alg){
		case LK:
			buildOpticalFlowPyramid(prevImg, prev_pyr,winSize,maxLayer,withDerivatives,BORDER_REPLICATE,BORDER_REPLICATE,true);
			buildOpticalFlowPyramid(img, pyr,winSize,maxLayer,withDerivatives,BORDER_REPLICATE,BORDER_REPLICATE,true);
			calcOpticalFlowPyrLK(prev_pyr, pyr, prevPts, nextPts, status, err, winSize, maxLayer);
			//fill the optical flow matrix with velocity vectors ( = difference of points in the two images)
		    for (int i = 0 ; i < sampled_i ; i ++){
		    	for (int j = 0 ; j < sampled_j ; j ++){
		    		int idx = i * sampled_j + j ;		    		
	    			Point2f p(nextPts[idx] - prevPts[idx]);
	    			Mat temp(flowResolution,flowResolution,CV_32FC2,Scalar(p.x,p.y));
                    if((j*flowResolution + flowResolution <= img_width) && (i*flowResolution + flowResolution) <= img_height){
                        temp.copyTo(optical_flow(Rect(j*flowResolution,i*flowResolution,flowResolution,flowResolution)));
                        //optical_flow(Rect(j*flowResolution,i*flowResolution,flowResolution,flowResolution)) = temp.clone();
                    }
		    	}
		    }
			break;		
		case FARNEBACK:
	    	calcOpticalFlowFarneback(prevImg, img, optical_flow, pyr_scale, maxLayer, winsize, of_iterations, poly_n, poly_sigma, flags);	
			break;		
		case GPU_LK_SPARSE:
			sparseLK_flow.winSize = Size(windows_size,windows_size);
			sparseLK_flow.maxLevel = maxLayer;
			sparseLK_flow.iters = of_iterations;
			sparseLK_flow.useInitialFlow = false;
	    	sparseLK_flow.sparse(gpuPrevImg8U,gpuImg8U,gpuPrevPts,gpuNextPts,gpu_status,gpu_err);

	    	gpuNextPts.download(nextPtsMat);

			//fill the optical flow matrix with velocity vectors ( = difference of points in the two images)
		    for (int i = 0 ; i < sampled_i ; i ++){
		    	for (int j = 0 ; j < sampled_j ; j ++){
		    		int idx = i * sampled_j + j ;		    		
	    			Point2f p(nextPtsMat.at<Point2f>(1,idx) - prevPtsMat.at<Point2f>(1,idx));
	    			Mat temp(flowResolution,flowResolution,CV_32FC2,Scalar(p.x,p.y));
	    			if((j*flowResolution + flowResolution <= img_width) && (i*flowResolution + flowResolution) <= img_height)
						temp.copyTo(optical_flow(Rect(j*flowResolution,i*flowResolution,flowResolution,flowResolution)));	    		 
		    	}
		    }
			break;//*/
		case GPU_LK_DENSE:
			denseLK_flow.winSize = Size(windows_size,windows_size);
			denseLK_flow.maxLevel = maxLayer;
			denseLK_flow.iters = of_iterations;
			denseLK_flow.useInitialFlow = true;//*/
	    	denseLK_flow.dense(gpuPrevImg8U,gpuImg8U,u_flow,v_flow);
            getFlowField(Mat(u_flow),Mat(v_flow),optical_flow);
            break;
		case GPU_FARNEBACK:
			farneback_flow.numLevels = maxLayer;
			farneback_flow.pyrScale = pyr_scale;
			farneback_flow.winSize = windows_size;
			farneback_flow.numIters = of_iterations;
	    	farneback_flow(gpuPrevImg8U,gpuImg8U,u_flow,v_flow);
    		getFlowField(Mat(u_flow),Mat(v_flow),optical_flow);
			break;
        /*case GPU_BROX:
			brox_flow(gpuPrevImg32F,gpuImg32F,u_flow,v_flow);
    		getFlowField(Mat(u_flow),Mat(v_flow),optical_flow);
            break;//*/
		default:
			break;
	}//*/

	/// Scale the Optical Flow field
	optical_flow = optical_flow*of_scale;

	/*dilate(optical_flow, optical_flow, getStructuringElement(MORPH_ELLIPSE, Size(10.0,10.0)));
    erode(optical_flow, optical_flow, getStructuringElement(MORPH_ELLIPSE, Size(10.0,10.0)));//*/

    /// Copy the optical flow field before filtering
    //optical_flow.copyTo(noFilt_of);
    noFilt_of = optical_flow.clone();


    /*** LOW PASS FILTERING ***/
    /*for (int i = 0 ; i < optical_flow.rows ; i ++){
		Point2f* op_ptr = optical_flow.ptr<Point2f>(i);
		Point2f* of_ptr = old_flow.ptr<Point2f>(i);
		for (int j = 0 ; j < optical_flow.cols ; j ++){
			Point2f p(op_ptr[j]);
			Point2f oldp(of_ptr[j]);
			op_ptr[j].x = low_pass_filter(p.x,oldp.x,Tc,1.0/(img_lowpass_freq));
			op_ptr[j].y = low_pass_filter(p.y,oldp.y,Tc,1.0/(img_lowpass_freq));
			//op_ptr[j].x = low_pass_filter(p.x,oldp.x,Tc,1.0/(cutf));
			//op_ptr[j].y = low_pass_filter(p.y,oldp.y,Tc,1.0/(cutf));
		}
    }

    optical_flow.copyTo(old_flow);//*/

    // OPTIONALLY !!!!!!!!!
    //blur(optical_flow,optical_flow,Size(windows_size*2,windows_size*2));


}


void of_driving::estimateAffineCoefficients(bool robust, Mat& gImg,Mat& ROI_ransac, Rect& rect_ransac){

    vector<Point2f> prevSamples, nextSamples, noFilt_prev, noFilt_next;
    int i, j;

	////if affine coefficients are retrieved by only three random points ...
	if(!robust){ 
		for (int k = 0 ; k < 3 ; k ++){

			i = rand()%(ROI_ransac.rows) + rect_ransac.tl().y;
			j = rect_ransac.br().x - rand()%(ROI_ransac.cols); 

			Point2f p(j,i);
			Point2f* of_ptr = optical_flow.ptr<Point2f>(i);
            Point2f* noFilt_ofptr = noFilt_of.ptr<Point2f>(i);

            Point2f p2(of_ptr[j] + p);
            Point2f nfp(noFilt_ofptr[j] + p);

			prevSamples.push_back(p);
			nextSamples.push_back(p2);
            noFilt_next.push_back(nfp);
		}

        Ab = getAffineTransform(prevSamples,nextSamples);
        noFilt_Ab = getAffineTransform(prevSamples,noFilt_next);

	}
	//// ... or affine coeffiencts have to be robustly estimated with all the points of the dominant plane		
	else{
		for (int i = 0 ; i < img_height ; i ++){
            unsigned char* dp_ptr = dominant_plane.ptr<uchar>(i);
            Point2f* of_ptr = optical_flow.ptr<Point2f>(i);

            unsigned char* nf_dp_ptr = noFilt_dp.ptr<uchar>(i);
            Point2f* nf_of_ptr = noFilt_of.ptr<Point2f>(i);

            for (int j = 0 ; j < img_width ; j ++){
				int val = dp_ptr[j];
                int nf_val = nf_dp_ptr[j];

				if(val == 255){
					Point2f p(j,i);
					Point2f p2(of_ptr[j] + p);
					prevSamples.push_back(p);
					nextSamples.push_back(p2);
				}
                if(nf_val == 255){
                    Point2f p(j,i);
                    Point2f p2(nf_of_ptr[j] + p);
                    noFilt_prev.push_back(p);
                    noFilt_next.push_back(p2);
                }
			}
		}

        if(!prevSamples.empty() && ! nextSamples.empty())
            Ab = estimateRigidTransform(prevSamples,nextSamples,true);


        if(!noFilt_prev.empty() && ! noFilt_next.empty())
            noFilt_Ab = estimateRigidTransform(noFilt_prev,noFilt_next,true);

	}

    if(!Ab.empty()){
        A = Ab(Rect(0,0,2,2));
        b = Ab(Rect(2,0,1,2));
	}	
    if(!noFilt_Ab.empty()){
        nf_A = noFilt_Ab(Rect(0,0,2,2));
        nf_b = noFilt_Ab(Rect(2,0,1,2));
    }
}


/*** DEPRECATED ***/
/*void of_driving::estimatePlanarFlowField(Mat& gImg){

	/*Mat src = Mat::zeros(img_height,img_width,CV_32FC2);
	Mat dst = Mat::zeros(img_height,img_width,CV_32FC2);
	for (int i = 0 ; i< img_height ; ++ i){
		for (int j = 0 ; j < img_width ; ++ j){
			src.ptr<Point2f>(i)[j] = Point2f(j,i);
		}
	}
	
	if(!Ab.empty()){
		warpAffine(src,dst,Ab,Size(img_width,img_height),WARP_INVERSE_MAP,BORDER_TRANSPARENT);
		planar_flow = dst - src;
	}

}//*/


void of_driving::buildPlanarFlowAndDominantPlane(Mat& ROI_ransac){

	point_counter = 0;
    nf_point_counter = 0;

	/*** MULTI-THREADED ***/
    parallel_for_(Range(0,cores_num),ParallelDominantPlaneBuild(cores_num,dominant_plane,inverted_dp,old_plane,optical_flow,planar_flow,epsilon,
                                                                Tc,img_lowpass_freq,A,b,dp_threshold, noFilt_dp, noFilt_of, noFilt_pf,
                                                                nf_A, nf_b));
	

	if(open_close){
	//Morphological opening
		erode(dominant_plane, dominant_plane, getStructuringElement(MORPH_ELLIPSE, Size(open_erode,open_erode)));
		dilate(dominant_plane, dominant_plane, getStructuringElement(MORPH_ELLIPSE, Size(open_dilate,open_dilate)));//*/

		//Morphological closing
		dilate(dominant_plane, dominant_plane, getStructuringElement(MORPH_ELLIPSE, Size(close_dilate,close_dilate)));
		erode(dominant_plane, dominant_plane, getStructuringElement(MORPH_ELLIPSE, Size(close_erode,close_erode)));//*/
	}
	else{
		//Morphological closing
		dilate(dominant_plane, dominant_plane, getStructuringElement(MORPH_ELLIPSE, Size(close_dilate,close_dilate)));
		erode(dominant_plane, dominant_plane, getStructuringElement(MORPH_ELLIPSE, Size(close_erode,close_erode)));//*/

		//Morphological opening
		erode(dominant_plane, dominant_plane, getStructuringElement(MORPH_ELLIPSE, Size(open_erode,open_erode)));
		dilate(dominant_plane, dominant_plane, getStructuringElement(MORPH_ELLIPSE, Size(open_dilate,open_dilate)));//*/
	}

    cv::bitwise_not(dominant_plane,inverted_dp);

	/*** SINGLE-THREADED PLANAR FLOW CONSTRUCTION ***/
	
	/*for (int i = 0 ; i < img_height ; i ++){
		Point2f* i_ptr = planar_flow.ptr<Point2f>(i);
		for (int j = 0 ; j < img_width ; j ++){
			Matx21f p(j,i);
			Matx21f planar_vec((A*p + b - p));
			Point2f planar_p(planar_vec(0),planar_vec(1));
			i_ptr[j] = planar_p;
		}
	}//*/

	/*** SINGLE-THREADED DOMINANT PLANE CONSTRUCTION ***/
	/*for (int i = 0 ; i < img_height ; i ++){
		unsigned char * dp_ptr = dominant_plane.ptr<uchar>(i);
		Point2f* of_ptr = optical_flow.ptr<Point2f>(i);
		Point2f* pf_ptr = planar_flow.ptr<Point2f>(i);
		for (int j = 0 ; j < img_width ; j ++){
			Point2f p(j,i);
			Point2f xdot(of_ptr[j]);
			Point2f xhat(pf_ptr[j]);
			
			if(norm(xdot - xhat) < epsilon){
				dp_ptr[j] = 255;
			}
			else{
				dp_ptr[j] = 0.0;
			}
		}
	}

	//erode(dominant_plane, dominant_plane, getStructuringElement(MORPH_ELLIPSE, Size(erode_factor,erode_factor)));
	//dilate(dominant_plane, dominant_plane, getStructuringElement(MORPH_ELLIPSE, Size(dilate_factor,dilate_factor)));
	
	for (int i = 0 ; i < img_height ; i ++){
		unsigned char* dp_ptr = dominant_plane.ptr<uchar>(i);
		unsigned char* op_ptr = old_plane.ptr<uchar>(i);
		for (int j = 0 ; j < img_width ; j ++){
				dp_ptr[j] = low_pass_filter(dp_ptr[j],op_ptr[j],Tc,1.0/img_lowpass_freq);
		}
	}	

	dominant_plane.copyTo(old_plane);

	double thresh = 100; //100
	double maxVal = 255;
	threshold(dominant_plane,dominant_plane,thresh,maxVal,THRESH_BINARY);//*/

    //point_counter = countNonZero(dominant_plane);


	//computeMaxConvexHull();


    point_counter = countNonZero(ROI_ransac);//*/
    nf_point_counter = countNonZero(noFilt_dp);

    //imshow("ROI_ransac",ROI_ransac);
    

}

void of_driving::extractPlaneBoundaries(){
    contours.clear();
    good_contours.clear();
    /*l_contours.clear();
    r_contours.clear();
    l_good_contours.clear();
    r_good_contours.clear();//*/

    //Find contours
    Mat inv_dpROI = inverted_dp(Rect(0,0,img_width,img_height));
    //Mat leftROI = inverted_dp(Rect(0,0,img_width/2,img_height));
    //Mat rightROI = inverted_dp(Rect(img_width/2,0,img_width/2,img_height));

    cv::findContours(inv_dpROI,contours,hierarchy,CV_RETR_TREE,CV_CHAIN_APPROX_SIMPLE, Point(0,0));
    //cv::findContours(leftROI,l_contours,l_hierarchy,CV_RETR_TREE,CV_CHAIN_APPROX_SIMPLE, Point(0,0));
    //cv::findContours(rightROI,r_contours,r_hierarchy,CV_RETR_TREE,CV_CHAIN_APPROX_SIMPLE, Point(img_width/2,0));

    std::vector < std::vector < cv::Point > > temp_cont;

    for (int i = 0 ; i < contours.size() ; i ++){
        if(contourArea(contours[i]) > area_ths){
            good_contours.push_back(contours[i]);
        }
    }//*/
    /*for (int i = 0 ; i < l_contours.size() ; i ++){
        if(contourArea(l_contours[i]) > area_ths){
            l_good_contours.push_back(l_contours[i]);
        }
    }
    for (int i = 0 ; i < r_contours.size() ; i ++){
        if(contourArea(r_contours[i]) > area_ths){
            r_good_contours.push_back(r_contours[i]);
        }
    }//*/

    //findGroundBoundaries();

}

void of_driving::findGroundBoundaries(){
    ground_contours.clear();
    ground_contours.resize(good_contours.size());

    Eigen::MatrixXd vel(6,1);
    Eigen::Vector2d pdot, pof;
    Eigen::MatrixXd J(2,6);
    vel << linear_vel, 0, 0, 0, 0, wz;

    Mat b_of;
    //optical_flow.copyTo(b_of);
    b_of = optical_flow.clone();

    //blur(optical_flow,b_of,Size(windows_size*2,windows_size*2));

    //buildMotionImage();

    for (int i = 0 ; i < good_contours.size() ; i ++){
        for (int j = 0 ; j < good_contours[i].size() ; j ++){

            //Get current point
            Point cv_p(good_contours[i][j]);

            double x = cv_p.x;
            double y = cv_p.y;

            //Normalize the point wrt the principal point of the camera
            double xn = x - principal_point.x;
            double yn = y - principal_point.y;

            //Get the corresponding optical flow vector
            pof << b_of.at<Point2f>(Point2f(x,y)).x, b_of.at<Point2f>(Point2f(x,y)).y;

            //Compute the feature motion based on the feature dynamics
            J = buildInteractionMatrix(xn,yn);
            J = J*W;
            pdot = J*vel;

            //Check for similarity
            if((pdot - pof).norm() < delta){
                ground_contours[i].push_back(cv_p);//<--- qui crasha
            }
            //ground_contours.push_back(temp);
        }
    }//*/

    //Draw Contours
    Mat pair = Mat::zeros( Size(img_width*2,img_height), CV_8UC3 );
    Mat bound_img = pair(Rect(0,0,img_width,img_height));
    Mat ground_img = pair(Rect(img_width,0,img_width,img_height));
    Scalar red = Scalar(0,0,255);
    Scalar green = Scalar(0,255,0);


    //Print contours
    //cout << "good_contours.size(): " << good_contours.size() << endl;
    for(int i = 0 ; i < good_contours.size() ; i ++){
        //cout << "\tlayer " << i << ", size: " << good_contours[i].size() << endl;
        for (int j = 0 ; j < good_contours[i].size() ; j ++){
            circle(bound_img,good_contours[i][j],2,red,2,CV_AA);
        }
    }

    //Print contours
    //cout << "ground_contours.size(): " << ground_contours.size() << endl;
    for(int i = 0 ; i < ground_contours.size() ; i ++){
        //cout << "\tlayer " << i << ", size: " << ground_contours[i].size() << endl;
        for (int j = 0 ; j < ground_contours[i].size() ; j ++){
            circle(ground_img,ground_contours[i][j],2,green,2,CV_AA);
        }
    }//*/


    /*for (int i = 0 ; i < ground_contours.size() ; i ++){
        drawContours(ground_img, ground_contours, i, green, 2, 8, noArray(), 0, Point() );
        drawContours(bound_img, good_contours, i, red, 2, 8, noArray(), 0, Point() );
    }//*/

    /// Show in a window
    namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
    imshow( "Contours", pair );


}


void of_driving::buildMotionImage(){

    Eigen::MatrixXd J(2,6);
    Eigen::Vector2d pp(principal_point.x, principal_point.y);
    Eigen::Matrix<double,6,1> vel;
    vel << linear_vel, 0, 0, 0, 0, wz;

    for (int i = 0 ; i < img_height ; i ++){
        Point2f* mot_ptr = motImg.ptr<Point2f>(i);
        for (int j = 0 ; j < img_width ; j ++){
            Eigen::Vector2d p(j,i);
            p -= pp;
            J = buildInteractionMatrix(p(0),p(1));

            J = J*W;

            Eigen::Vector2d p2 = J*vel;
            mot_ptr[j] = Point2f(p2(0),p2(1));

        }
    }

    Mat motion;
    cvtColor(GrayImg,motion,CV_GRAY2BGR);

    for (int i = 0 ; i < motion.rows ; i+= flowResolution*2){
        const Point2f* mot_ptr = motImg.ptr<Point2f>(i);
        for (int j = 0 ; j < motion.cols ; j += flowResolution*2){
            cv::Point2f p(j,i);
            cv::Point2f p2(p + mot_ptr[j]);
            arrowedLine2(motion,p,p2,Scalar(0,0,255),0.1,8,0,0.1);
        }
    }

    imshow("motion",motion);


}//*/

Eigen::MatrixXd of_driving::buildInteractionMatrix(double x, double y){

    Eigen::MatrixXd J(2,6);

    double hc = camera_height;
    double gamma = camera_tilt;
    double f = focal_length;
    double Z = hc/cos(gamma + atan2(y,f));

    J <<    - f/Z,     0, x/Z,       x*y/f, -(f + x*x/f),  y,
                0,  -f/Z, y/Z, (f + y*y/f),       -x*y/f, -x;

    return J;
}

void of_driving::computeCentroids(){
    //Get Moments
    vector < Moments > mu;
    //vector < Moments > l_mu;
    //vector < Moments > r_mu;



    centroids.clear();
    l_centroids.clear();
    r_centroids.clear();
    lx_contours.clear();
    rx_contours.clear();
    ly_contours.clear();
    ry_contours.clear();

    for (int i = 0 ; i < good_contours.size() ; i ++){
        mu.push_back(moments(good_contours[i]));
    }//*/

    /*for (int i = 0 ; i < l_good_contours.size() ; i ++){
        l_mu.push_back(moments(l_good_contours[i]));
    }
    for (int i = 0 ; i < r_good_contours.size() ; i ++){
        r_mu.push_back(moments(r_good_contours[i]));
    }//*/


    //Get the mass centers
    //std::cout << "centroids found: " << mu.size() << std::endl;
    for (int i = 0 ; i < mu.size() ; i ++){
        Point2f c(mu[i].m10/mu[i].m00,mu[i].m01/mu[i].m00);
        centroids.push_back(c);
        centroids_vec_f << c.x << ", " << c.y << ",";
        //if(c.y < (img_height - img_height/2)){
            if(c.x > img_width/2){
                r_centroids.push_back(c);
                for (int j = 0 ; j < good_contours[i].size() ; j ++){
                    if(VREP_SIM || (!VREP_SIM && good_contours[i][j].y < img_height/2)){
                        rx_contours.push_back(good_contours[i][j].x);
                        ry_contours.push_back(good_contours[i][j].y);
                    }
                }
            }
            else if(c.x < img_width/2){
                l_centroids.push_back(c);
                for (int j = 0 ; j < good_contours[i].size() ; j ++){
                    if(VREP_SIM || (!VREP_SIM && good_contours[i][j].y < img_height/2)){
                        lx_contours.push_back(good_contours[i][j].x);
                        ly_contours.push_back(good_contours[i][j].y);
                    }
                }
            }
        //}
    }
    centroids_vec_f << ";" << std::endl;//*/

    Point l_maxLoc;
    Point r_minLoc;

    if(!lx_contours.empty()){
        minMaxLoc(lx_contours,NULL,&maxLeftPoint.x,NULL,&l_maxLoc);
    }
    else{
        maxLeftPoint = Point2f(0,prevLeftPoint.y);
    }

    if(!rx_contours.empty()){
        minMaxLoc(rx_contours,&minRightPoint.x,NULL,&r_minLoc);
    }
    else{
        minRightPoint = Point2f(img_width,prevRightPoint.y);
    }

    if(!ly_contours.empty() && !ry_contours.empty()){
        maxLeftPoint.y = ly_contours[l_maxLoc.x];
        minRightPoint.y = ry_contours[r_minLoc.x];
    }

    /*std::cout << "number of left contours: " << l_contours.size() << std::endl;
    for (int i = 0 ; i < l_contours.size() ; i ++){
        minMaxLoc(l_contours[i],&l_min,&l_maxXPxlContour[i]);
        std::cout << "\tmax Pixel of centroid " << i << ": " << l_maxXPxlContour[i] << std::endl;
    }

    std::cout << "number of right contours: " << r_contours.size() << std::endl;
    for (int i = 0 ; i < r_contours.size() ; i ++){
        minMaxLoc(r_contours[i],&l_min,&r_maxXPxlContour[i]);
        std::cout << "\tmax Pixel of centroid " << i << ": " << r_maxXPxlContour[i] << std::endl;
    }//*/


    /*for (int i = 0 ; i < l_mu.size() ; i ++){
        Point2f c(l_mu[i].m10/l_mu[i].m00,l_mu[i].m01/l_mu[i].m00);
        l_centroids.push_back(c);
    }
    for (int i = 0 ; i < r_mu.size() ; i ++){
        Point2f c(r_mu[i].m10/r_mu[i].m00,r_mu[i].m01/r_mu[i].m00);
        r_centroids.push_back(c);
    }//*/




    x_r = Point2f(0,0);
    x_l = Point2f(0,0);

    int r_size = r_centroids.size();
    int l_size = l_centroids.size();

    if(r_size == 0){
    x_r = Point2f(img_width,old_xr.y);
    }
    else{
        for (int i = 0 ; i < r_size ; i ++){
            x_r = Point2f(x_r + r_centroids[i]);
        }
        x_r = x_r*(1.0/((float)r_size));
    }
    if(l_size == 0){
    x_l = Point2f(0,old_xl.y);
    }
    else{
        for (int i = 0 ; i < l_size ; i ++){
            x_l = Point2f(x_l + l_centroids[i]);
        }
        x_l = x_l*(1.0/((float)l_size));
    }


    prevx_r = x_r;
    prevx_l = x_l;


}


void of_driving::computeGradientVectorField(){

	Size GaussSize(51,51);
	double sigmaX = img_width*img_height*0.5;
	int ddepth = CV_32F; //CV_16S


	/*** SINGLE-THREADED GRADIENT FIELD CONSTRUCTION ***/
	/*Mat grad_x, grad_y;
	GpuMat gpu_gx, gpu_gy;
	double scale = grad_scale;
	double delta = 0.0;

	GpuMat gpu_dp(dominant_plane);

	GaussianBlur(dominant_plane,smoothed_plane,GaussSize,sigmaX,0);
    //GaussianBlur(gpu_dp,gpu_dp,GaussSize,sigmaX,0);

    Scharr(smoothed_plane, grad_x, ddepth, 1, 0, scale);
    Scharr(smoothed_plane, grad_y, ddepth, 0, 1, scale);

	//Scharr(gpu_dp, gpu_gx, ddepth, 1, 0, scale);
    //Scharr(gpu_dp, gpu_gy, ddepth, 0, 1, scale);

    int rows = img_height;
    int cols = img_width;

    if(grad_x.isContinuous() && grad_y.isContinuous() && gradient_field.isContinuous()){
        cols *= rows;
        rows = 1;
    }


    for (int i = 0 ; i < img_height ; i ++){
    	float* x_ptr = grad_x.ptr<float>(i);
    	float* y_ptr = grad_y.ptr<float>(i);
    	Point2f* grad_ptr = gradient_field.ptr<Point2f>(i);
    	for (int j = 0 ; j < img_width ; j ++){
    		grad_ptr[j] = Point2f(x_ptr[j],y_ptr[j]);
    	}
    }//*/

	/*** MULTI-THREADED GRADIENT FIELD CONSTRUCTION ***/
    parallel_for_(Range(0,cores_num),ParallelGradientFieldBuild(cores_num, dominant_plane, smoothed_plane, gradient_field,
                                                                vortex_field, grad_scale,GaussSize, sigmaX,
                                                                ddepth, noFilt_dp, noFilt_sp, noFilt_gf, noFilt_vf));

}


void of_driving::computeControlForceOrientation(){
    Mat ROI;
    Mat nfROI;
    Mat resROI, nfResROI;
    //ROI = gradient_field(Rect(0,0,img_width,img_height/2)).clone();
    //nfROI = noFilt_gf(Rect(0,0,img_width,img_height/2)).clone();

    ROI = gradient_field(Rect(0,0,img_width,img_height/2));
    nfROI = noFilt_gf(Rect(0,0,img_width,img_height/2));

    /*result_field = gradient_field;
    noFilt_rf = noFilt_gf;//*/

    /*result_field = vortex_field;
    noFilt_rf = noFilt_vf;//*/


    result_field = (1.0 - field_weight)*gradient_field + field_weight*vortex_field;
    noFilt_rf = noFilt_gf + noFilt_vf;//*/

    resROI = result_field(Rect(0,0,img_width,img_height/2));
    nfResROI = noFilt_rf(Rect(0,0,img_width,img_height/2));

	p_bar = Matx21f(0,0);
    noFilt_pbar = Matx21f(0,0);

    int rows = ROI.rows;
    int cols = ROI.cols;

    if(ROI.isContinuous() && nfROI.isContinuous()){
        cols *= rows;
        rows = 1;
    }

	for (int i = 0 ; i < rows ; i ++){
        Point2f* roi_ptr = resROI.ptr<Point2f>(i);
        Point2f* nf_roi_ptr = nfResROI.ptr<Point2f>(i);
        for (int j = 0 ; j < cols ; j ++){
            Matx21f vec(roi_ptr[j].x,roi_ptr[j].y);
            Matx21f nf_vec(nf_roi_ptr[j].x,nf_roi_ptr[j].y);
            p_bar += vec;
            noFilt_pbar += nf_vec;
		}
	}

    p_bar *= (1.0/(ROI.rows*ROI.cols));
    noFilt_pbar *= (1.0/(ROI.rows*ROI.cols));


    Point2f pbar(p_bar(0),p_bar(1));
    Point2f noFilt_p(noFilt_pbar(0),noFilt_pbar(1));


    //cout << "norm(p_bar): " << norm(pbar) << endl;
    //cout << "norm(noFilt_p): " << norm(noFilt_p) << endl;

    //p_bar(0)  = low_pass_filter(pbar.x,px_old,Tc,1.0/bar_lowpass_freq);
    //p_bar(1)  = low_pass_filter(pbar.y,py_old,Tc,1.0/bar_lowpass_freq);//*/

	px_old = p_bar(0);
	py_old = p_bar(1);

	pbar.x = p_bar(0);
	pbar.y = p_bar(1);

    /*if(record){
        nofilt_barFile << noFilt_p.x << ", " << noFilt_p.y << "; " << endl;
        filt_barFile << pbar.x << ", " << pbar.y << "; " << endl;
    }//*/

    Point2f y(0,-1);

    if(norm(pbar) > 5/*5//*/){
		theta = pbar.dot(y);
		theta /= (norm(pbar)*norm(y));
        theta = acos(theta); //Principal arc cosine of x, in the interval [0,pi] radians. So, if we want to discriminate when the vector is on the left or right side, we need to check for the x component and
                             //remap accordingly from the range [0,pi] to [0,2pi]
        if (p_bar(0) < 0){
            theta = 2*M_PI - theta;
        }//*/
	}
	else{
		theta = 0.0;
	}

    /*if(record)
        theta_f << theta << "; " << endl;//*/

    //theta = low_pass_filter(theta,theta_old,Tc,1.0/ctrl_lowpass_freq);
    //cout << "theta: " << theta*180.0/M_PI << endl;

}


void of_driving::computeRobotVelocities(bool move_robot, bool narrowCheck){

    /*** Previous control law ***/
    double R = max_w*sin(theta);
	angular_vel = R ;

    /*if(record)
        angularVel_f << angular_vel << "; " << endl;//*/

    if(ankle_angle != -1){
        steering = wheelbase * R/linear_vel;
    }
    else{
        steering = 0.0;
    }

    ankle_angle = 0.0;
    /*****************************************/

    /*** Visual servoing control law ***/
    double Zl, Zr, hc, gamma, f, xl, yl, xr, yr;
    hc = camera_height;
    gamma = camera_tilt;
    f = focal_length;

    Eigen::Matrix<double,2,1> e;
    double err, theta_err;

    Eigen::Matrix<double,2,1> cmd_vel; //vy, wz
    Eigen::Matrix<double,2,1> Jvx;
    Eigen::Matrix<double,1,1> v;
    Eigen::Matrix<double,6,1> v6;
    Eigen::Matrix<double,2,2> Ju;
    Eigen::Matrix<double,1,6> J;
    Eigen::Matrix<double,1,3> Jv,Jw;
    Eigen::Matrix<double,1,6> Lx_l, Lx_r;
    Eigen::Matrix<double,2,6> Lxy_l, Lxy_r;
    Eigen::Matrix<double,1,6> Lxl, Lxr, Lyl, Lyr;
    Point2f c_rn, c_ln;

    Eigen::Matrix2d A;
    Eigen::Matrix<double,2,1> B;


    Eigen::Matrix2d lambda_d;
    lambda_d << 2.5, 0.0,
                0.0, 0.5;

    double detJu;
    Eigen::Matrix<double,2,2> Juinv;

    double m = 5.0;
    double c = 0.02;

    vr = (x_r - old_xr)*(1.0/Tc);
    vl = (x_l - old_xl)*(1.0/Tc);

    acc_r = (vr - vr_old)*(1.0/Tc);
    acc_l = (vl - vl_old)*(1.0/Tc);//*/

    /*if(x_r.x > img_width - px_margin && x_r.x != img_width && vr.x > 0){

        acc_r = 0.05*acc_r;
        vr = vr_old + Tc*acc_r;
        x_r = old_xr + vr*Tc;

        //x_r = (2*m*old_xr - m*old2_xr + Tc*c*old_xr)*(1.0/(m + Tc*c));
        x_r.x = ((x_r.x > img_width) ? (img_width) : (x_r.x));
        x_r.y = ((x_r.y < img_height) ? ((x_r.y > 0 ) ? (x_r.y) : (0)) : (img_height));
    }
    if(x_l.x < px_margin && x_l.x != 0.0 && vl.x < 0){

        acc_l = 0.05*acc_l;
        vl = vl_old + Tc*acc_l;
        x_l = old_xl + vl*Tc;

        //x_l = (2*m*old_xl - m*old2_xl + Tc*c*old_xl)*(1.0/(m + Tc*c));
        x_l.x = ((x_l.x < 0) ? (0) : (x_l.x));
        x_l.y = ((x_l.y < img_height) ? ((x_l.y > 0 ) ? (x_l.y) : (0)) : (img_height));
    }//*/


    float r_frac = 0.5;
    float l_frac = 0.5;

    x_r.x = low_pass_filter(x_r.x,old_xr.x,Tc,1.0/(img_lowpass_freq*r_frac));
    x_r.y = low_pass_filter(x_r.y,old_xr.y,Tc,1.0/(img_lowpass_freq*r_frac));
    x_l.x = low_pass_filter(x_l.x,old_xl.x,Tc,1.0/(img_lowpass_freq*l_frac));
    x_l.y = low_pass_filter(x_l.y,old_xl.y,Tc,1.0/(img_lowpass_freq*l_frac));

    vr_old = vr;
    vl_old = vl;

    old2_xr = old_xr;
    old2_xl = old_xl;
    old_xr = x_r;
    old_xl = x_l;//*/

    //Restore x_r and x_l coordinates with respect to the real image
    Point2f x_R = x_r + Point2f(real_roix,real_roiy);
    Point2f x_L = x_l + Point2f(real_roix,real_roiy);

    //Extract normalized coordinates
    c_rn = (x_R - principal_point);
    c_ln = (x_L - principal_point);

    xl = c_ln.x;
    yl = c_ln.y;
    xr = c_rn.x;
    yr = c_rn.y;

    //Build the interaction matrix
    Lxy_l = buildInteractionMatrix(xl,yl);
    Lxy_r = buildInteractionMatrix(xr,yr);

    Lx_l = Lxy_l.row(0);
    Lx_r = Lxy_r.row(0);


    double nw = abs(maxLeftPoint.x - minRightPoint.x);
    if(nw >= 30){
        narrow_width = nw;
    }

    narrow_width = low_pass_filter(narrow_width,old_narrow_width,Tc,1.0/(img_lowpass_freq*0.5));
    old_narrow_width = narrow_width;//*/

    if(record){
        narrow_distance_f << narrow_width << "; " << std::endl;
    }

    // Select reference head orientation q_p*
    double max_qp = MAXYAW;
    int low_thres = (theta_des == max_qp) ? (low_max_narrow_ths) : (low_min_narrow_ths);
    int high_thres = (theta_des == 0.0) ? (high_min_narrow_ths) : (high_max_narrow_ths);


    /*std::cout << "narrow_width: " << narrow_width << std::endl;
    std::cout << "theta_des: " << theta_des << std::endl;
    std::cout << "current low threshold: " << low_thres << std::endl;
    std::cout << "current high threshold: " << high_thres << std::endl << std::endl;//*/

    if(narrowCheck){

        if((l_centroids.size() != 0 && r_centroids.size() != 0) || theta_des != 0.0) {
            if(narrow_width > high_thres){
                theta_des = 0.0;
            }
            else if(narrow_width < low_thres){
                theta_des = max_qp;
                std::cout << "Orientation changed to 90 deg!" << std::endl;
            }
            else{
                theta_des = max_qp - ((narrow_width - low_thres)/(high_thres  - low_thres))*max_qp;
            }
        }//*/

    }

    bool single_control_var = true;


    //Define the error
    if(single_control_var){
        err = xl + xr;

        theta_err = real_pan - theta_des;
        //theta_err = theta_des - real_pan;

        //std::cout << "err: " << err << std::endl;
    }
    else{
       // e << yr - yl, xl + xr ;
        e << xr + xl , real_pan - theta_des;
        std::cout << "error: " << e << std::endl;
    }

    v << linear_vel;
    //v3 << vx, vy, 0, 0, 0, wz;


    if(single_control_var){

        // Multiply the interaction matrices by the twist matrix
        Lx_l = Lx_l*W;
        Lx_r = Lx_r*W;
        J = Lx_l + Lx_r;

        Jv = J.block<1,3>(0,0);
        Jw = J.block<1,3>(0,3);

        vx = linear_vel*cos(real_pan);
        vy = linear_vel*sin(real_pan);

        // CONTROLLER 2

        float lambda;
        float lambda_w;

        if(VREP_SIM){
            lambda = 0.5;//0.5
            lambda_w = 0.25;
        }
        else{
            lambda = 0.8;
            lambda_w = 0.15;
        }


        /*wz = lambda_w*theta_err; // !!! Non deve essere u_pan, ma l'angolo di pan misurato dagli encoder, devi correggere questa riga!!!
        wz = (wz > - ANGULAR_VEL_MAX) ? ( (wz < ANGULAR_VEL_MAX) ? (wz) : (ANGULAR_VEL_MAX) ) : (-ANGULAR_VEL_MAX) ;
        pan_dot = - (lambda*err + Jv(0)*vx + Jv(1)*vy)/Jw(2) - wz;//*/

        // CONTROLLER 1

        double Kw, Kp;

        if(VREP_SIM){
            Kw = 0.4; //0.4
            Kp = 0.05;//0.05
        }
        else{
            Kw = 1.5;//.8
            Kp = 0.05;
        }


        pan_dot = - Kp * theta_err;
        wz = - ( Kw*err + Jv(0)*vx + Jv(1)*vy)/Jw(2) - pan_dot;
        wz = (wz > - ANGULAR_VEL_MAX) ? ( (wz < ANGULAR_VEL_MAX) ? (wz) : (ANGULAR_VEL_MAX) ) : (-ANGULAR_VEL_MAX) ;//*/

        if((!VREP_SIM && move_robot) || (VREP_SIM && move_robot)){
            u_pan = u_pan_old + pan_dot*Tc;
            u_pan_old = u_pan;
            u_pan = (u_pan < MAXYAW) ? ( (u_pan > MINYAW) ? (u_pan) : (MINYAW) ) : (MAXYAW) ;
        }

        //std::cout << "pan_dot: " << pan_dot << std::endl;
        //std::cout << "u_pan: " << u_pan << std::endl;

        //applyPanCmdonNAOqi(move_robot);
        real_pan = getRealPanFromNAOqi();


        /*std::cout << "theta_des: " << theta_des << std::endl;
        std::cout << "real_pan: " << real_pan << std::endl;
        std::cout << "orientation error: " << theta_des - real_pan << std::endl;
        std::cout << "wz: " << wz << std::endl;
        std::cout << "wz (saturated): " << wz << std::endl << std::endl;//*/

        //velocityScaling();


        if(record){
            pan_f << pan_dot << ", " << u_pan << ", " << real_pan << "; " << std::endl;
        }

        /*std::cout << "u_pan: " << u_pan << std::endl;
        std::cout << "real_pan: " << real_pan << std::endl;
        std::cout << "vx: " << vx << std::endl;
        std::cout << "vy: " << vy << std::endl;
        std::cout << "wz: " << wz << std::endl;//*/

        //// THIS IS THE WORKING CONTROL LAW
        /*cmd_vel(1) = -(lambda*err + Jv(0)*v(0))/Jw(2);
        wz = cmd_vel(1);
        vy = 0.0;//*/
    }
    else{

        // Multiply the interaction matrices by the twist matrix
        Lx_l = Lx_l*W;
        Lx_r = Lx_r*W;//*/
        J = Lx_l + Lx_r;

        Jv = J.block<1,3>(0,0);
        Jw = J.block<1,3>(0,3);

        A << Jw(2), Jw(2),
                 1,     0;

        B << Jv(0)*cos(real_pan) + Jv(1)*sin(real_pan), 0;


        //theta = theta_des;
        // cmd_vel = [wz; qp_dot]
        cmd_vel = - A.inverse()*(lambda_d*e + B*linear_vel);


        //std::cout << "cmd_vel: " << cmd_vel << std::endl;

        if(!VREP_SIM || (VREP_SIM && move_robot)){ // <-- RICORDATI DI AGGIUSTARE QUI!!!
            wz = cmd_vel(0);
            pan_dot = cmd_vel(1);

            wz = (wz > - ANGULAR_VEL_MAX) ? ( (wz < ANGULAR_VEL_MAX) ? (wz) : (ANGULAR_VEL_MAX) ) : (-ANGULAR_VEL_MAX) ;
            u_pan = u_pan_old + pan_dot*Tc;
            u_pan_old = u_pan;
            u_pan = (u_pan < MAXYAW) ? ( (u_pan > MINYAW) ? (u_pan) : (MINYAW) ) : (MAXYAW) ;
        }


        //velocityScaling();

        //applyPanCmdonNAOqi(move_robot);
        real_pan = getRealPanFromNAOqi();
        vx = linear_vel*cos(real_pan);
        vy = linear_vel*sin(real_pan);

        if(record){
            pan_f << u_pan << ", " << real_pan << "; " << std::endl;
        }

    }


    //velocityScaling();

    if(record){
        if(single_control_var){
            error_f << err << ", " << (theta_des - real_pan) << "; " << endl;
        }
        else{
            error_f << e(0,0) << ", "
                    << e(1,0) << "; " << endl;//*/
        }
        xl_f << xl << ", " << yl << "; " << endl;
        xr_f << xr << ", " << yr << "; " << endl;
        det_f << detJu << "; " << endl;
        //vx_f << linear_vel << "; " << endl;
        //vy_f << cmd_vel(0) << ", " << vy << "; " << endl;//!!!! RICORDA DI INVERTIRE DI NUOVO GLI INDICI, QUESTA È UNA PROVA (DEVE ESSERE 0)
        //wz_f << cmd_vel(1) << ", " << wz << "; " << endl;//!!!! RICORDA DI INVERTIRE DI NUOVO GLI INDICI, QUESTA È UNA PROVA (DEVE ESSERE 1)

        vx_f << vx << "; " << endl;
        vy_f << vy << "; " << endl;
        wz_f << wz << "; " << endl;

        Ju_f << Juinv(0,0) << ", " << Juinv(0,1) << ", " << Juinv(1,0) << ", " << Juinv(1,1) << ";" << endl;
        J_f << Jv(0) << ", " << Jw(2) << "; " << endl;
    }

    //cout << "vx: " << linear_vel << "\t vy: " << vy << "\t wz: " << wz << endl;

}

void of_driving::applyPanCmdonNAOqi(bool* move_robotPtr, char *key){

    while((int)(*key) != 27){
            //headYawFrame.clear();

            //std::cout << " Inside thread ... " << std::endl ;
        if((!VREP_SIM && *move_robotPtr) || (VREP_SIM && *move_robotPtr)){

            AL::ALValue pan_name = AL::ALValue::array(PAN_JOINT);
            AL::ALValue pan_value = AL::ALValue::array(u_pan);
            //AL::ALValue pan_value = AL::ALValue::array(1.57);
            double fracSpeed = 1.0; //0.2

                motionPtr->setAngles(pan_name,pan_value,fracSpeed);

                if(!VREP_SIM){
                    qi::os::msleep(25);
                }
            }
            //std::cout << "Applying head yaw angle " << u_pan*180.0/M_PI << " deg" << std::endl;
            /*if(headYawRegulation)
                break;//*/

    }

}

void of_driving::callNaoqiMove(bool *move_robot, char* key){
    while((int)(*key) != 27){

        if(*move_robot){

            set_linearVel(LINEAR_VEL_MAX);
            set_angVel(wz);
            motionPtr->move(vx,vy,wz);//FRAME_ROBOT
        }
        else{
            set_angVel(0.0);
            motionPtr->stopMove();
        }

    }

}

double of_driving::getRealPanFromNAOqi(){

    std::vector<float> headYawFrame;

    headYawFrame = motionPtr->getTransform("HeadYaw",FRAME_ROBOT,true);

    /*std::cout << "\nheadYawFrame: " << std::endl;
    std::cout << headYawFrame[0] << ", " << headYawFrame[1] << ", " << headYawFrame[2] << ", " << headYawFrame[3] << ", " << std::endl
              << headYawFrame[4] << ", " << headYawFrame[5] << ", " << headYawFrame[6] << ", " << headYawFrame[7] << ", " << std::endl
              << headYawFrame[8] << ", " << headYawFrame[9] << ", " << headYawFrame[10] << ", " << headYawFrame[11] << ", " << std::endl
              << headYawFrame[12] << ", " << headYawFrame[13] << ", " << headYawFrame[14] << ", " << headYawFrame[15] << ", " << std::endl << std::endl;//*/

    float pitch = atan2( -headYawFrame[8], sqrt(headYawFrame[9]*headYawFrame[9] + headYawFrame[10]*headYawFrame[10]));
    //return real_pan = atan2(headYawFrame[4],headYawFrame[0]);
    return real_pan = atan2(headYawFrame[4]/cos(pitch),headYawFrame[0]/cos(pitch));
}

void of_driving::velocityScaling(){

    double k1,k2;
    double vx = max_v;
    double v_norm = sqrt(vx*vx + vy*vy);

    if( v_norm > max_v){
        //cout << "LINEAR VELOCITY SATURATED!!!" << endl;
        k1 = max_v/v_norm;
        vx *= k1;
        vy *= k1;
        wz *= k1;
    }
    double w_norm = abs(wz);
    if(w_norm > max_w){
        //cout << "ANGULAR VELOCITY SATURATED!!!" << endl;
        k2 = max_w/w_norm;
        //vx *= k2;
        //vy *= k2;
        wz *= k2;
    }

    linear_vel = vx;
}

void of_driving::computeFlowDirection(){

    int rows = optical_flow.rows;
    int cols = optical_flow.cols;

    for (int i = 0 ; i < rows ; i ++){
        Point2f* of_ptr = optical_flow.ptr<Point2f>(i);
        unsigned char* atan_ptr = atanMat.ptr<uchar>(i);
        for (int j = 0 ; j < cols ; j ++){
            Point2f p(of_ptr[j]);
            double dir = atan2(p.y,p.x);
            if(dir < 0)
                dir += 2*M_PI;
            int udir = dir/(2*M_PI)*255;
            atan_ptr[j] = udir;
        }
    }

    imshow("atanMat",atanMat);
}



void of_driving::set_cameraPose(std::vector<float> pose){

    Eigen::Matrix3d tskew;

    //This is the matrix Trc expressing the camera frame (not optical) wrt to the robot frame
    cameraPose(0,0) = pose.at(0);
    cameraPose(0,1) = pose.at(1);
    cameraPose(0,2) = pose.at(2);
    cameraPose(0,3) = pose.at(3);
    cameraPose(1,0) = pose.at(4);
    cameraPose(1,1) = pose.at(5);
    cameraPose(1,2) = pose.at(6);
    cameraPose(1,3) = pose.at(7);
    cameraPose(2,0) = pose.at(8);
    cameraPose(2,1) = pose.at(9);
    cameraPose(2,2) = pose.at(10);
    cameraPose(2,3) = pose.at(11);
    cameraPose(3,0) = 0;
    cameraPose(3,1) = 0;
    cameraPose(3,2) = 0;
    cameraPose(3,3) = 1;

    Eigen::Matrix4d Tco, Tro, Tor; //cameraPose is Trc
    Tco <<  0,  0, 1, 0,
           -1,  0, 0, 0,
            0, -1, 0, 0,
            0,  0, 0, 1;

    Tro = cameraPose*Tco;
    Tor = Tro.inverse();

    cameraR = Tor.topLeftCorner(3,3);
    cameraT = Tor.block<3,1>(0,3);

    // Write the translation vector as skew-symmetric matrix for the cross-product
    tskew <<          0, -cameraT(2),  cameraT(1),
             cameraT(2),           0, -cameraT(0),
            -cameraT(1),  cameraT(0),           0;

    // Build the twist matrix
    W.topLeftCorner(3,3) = cameraR;
    W.topRightCorner(3,3) = tskew*cameraR;
    W.bottomRightCorner(3,3) = cameraR;

}

/*** SINGLE-THREADED DISPLAY FUNCTION ***/
Mat of_driving::displayImages(Mat& img){

	string ofAlg;
	Mat gImg, u_img, p_img, gradient_img, potential_img, cf_img;
	Point2f center(img.cols/2,img.rows/2);
	Point2f pb(p_bar(0),p_bar(1));
	Point2f px(pb.x*20.0,0);
	Point2f y(0,-1);

	double pbnorm = norm(px);
	double pbnorm_max = 50;

	if (pbnorm > pbnorm_max){
		pbnorm = pbnorm_max;
        px.x = (px.x > 0) ? (50) : (-50);
	}

	double red = pbnorm/pbnorm_max*255.0 ;
	double green = 255.0 - red;

	img.copyTo(cf_img);
	potential_img = Mat::ones(img_height,img_width,CV_8UC1)*255;

    /*try{
	cvtColor(img,gImg,CV_BGR2GRAY);
	cvtColor(gImg,u_img,CV_GRAY2BGR);
	cvtColor(gImg,p_img,CV_GRAY2BGR);
    }
    catch(cv::Exception &e){
    	cerr << "cvtColor in displayImages" << endl;
    }//*/
    
    smoothed_plane.copyTo(gradient_img);
    cvtColor(gradient_img,gradient_img,CV_GRAY2BGR);


    /*** Optical Flow - Planar Flow - Gradient Field ***/
    for (int i = 0 ; i < img.rows ; i += flowResolution*2){
    	Point2f* of_ptr = optical_flow.ptr<Point2f>(i);
		Point2f* pf_ptr = planar_flow.ptr<Point2f>(i);
		Point2f* gf_ptr = gradient_field.ptr<Point2f>(i);
		Point2f* pot_ptr = potential_field.ptr<Point2f>(i);
 	   for (int j = 0 ; j < img.cols ; j += flowResolution*2){
    	    Point2f p(j,i);
        	Point2f po(p+of_ptr[j]);
			Point2f pp(p+pf_ptr[j]);
			Point2f pg(p+gf_ptr[j]*0.1);
			Point2f norm_pg((p + pg)*(1.0/norm(p + pg)));
        	arrowedLine2(u_img,p,po,Scalar(0,0,255),0.1,8,0,0.1);
			arrowedLine2(p_img,p,pp,Scalar(255,255,0),0.1,8,0,0.1);
			arrowedLine2(gradient_img,p,pg,Scalar(0,255,0),0.1,8,0,0.1);
	    } 
	}

	/*** Control Force***/

	arrowedLine2(cf_img,center,center + y*50,Scalar(255,0,0),3.0,8,0,0.1);
	arrowedLine2(cf_img,center,center + px,Scalar(0,green,red),3.0,8,0,0.1);

	Mat total = Mat::zeros(2*img_height,3*img_width,CV_8UC3);
	Mat dp_img, sp_img;
	cvtColor(dominant_plane,dp_img,CV_GRAY2BGR);
    cvtColor(smoothed_plane,sp_img,CV_GRAY2BGR);
	cvtColor(potential_img,potential_img,CV_GRAY2BGR);

	u_img.copyTo(total(Rect(0,0,img_width,img_height)));
	p_img.copyTo(total(Rect(img_width,0,img_width,img_height)));
	dp_img.copyTo(total(Rect(2*img_width,0,img_width,img_height)));
	sp_img.copyTo(total(Rect(0,img_height,img_width,img_height)));
	gradient_img.copyTo(total(Rect(img_width,img_height,img_width,img_height)));
	cf_img.copyTo(total(Rect(2*img_width,img_height,img_width,img_height)));//*/

	//imshow("Steps",total);
	string flow_type;
	switch(of_alg){
		case LK:
			flow_type = "Lucas-Kanade Optical Flow";
			break;
		case FARNEBACK:
			flow_type = "Farneback Optical Flow";
			break;
		case GPU_LK_DENSE:
			flow_type = "GPU Dense Lucas-Kanade Optical Flow";
			break;
		case GPU_FARNEBACK:
			flow_type = "GPU Farneback Optical Flow";
			break;
		case GPU_BROX:
			flow_type = "GPU Brox Optical Flow";
			break;
		default:
			break;
	}
	//imshow(flow_type,total);
	//imshow("dominant plane",dp_img);

    /*if(save_video){
		record_total.write(total);
    }//*/


	return total;
}

string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}



void of_driving::openFiles(const string full_path){
    //nofilt_barFile.open((full_path + "nofiltp.txt").c_str(),ios::app);
    //filt_barFile.open((full_path + "filtp.txt").c_str(),ios::app);
    theta_f.open((full_path + "theta.txt").c_str(),ios::app);
    pan_f.open((full_path + "pan.txt").c_str(),ios::app);
    //angularVel_f.open((full_path + "angularVel.txt").c_str(),ios::app);
    error_f.open((full_path + "centroids_error.txt").c_str(),ios::app);
    xl_f.open((full_path + "xl.txt").c_str(),ios::app);
    xr_f.open((full_path + "xr.txt").c_str(),ios::app);
    //R_f.open((full_path + "R.txt").c_str(),ios::app);
    vx_f.open((full_path + "vx.txt").c_str(),ios::app);
    vy_f.open((full_path + "vy.txt").c_str(),ios::app);
    wz_f.open((full_path + "wz.txt").c_str(),ios::app);
    det_f.open((full_path + "detJu.txt").c_str(),ios::app);
    Ju_f.open((full_path + "Ju.txt").c_str(),ios::app);
    J_f.open((full_path + "J.txt").c_str(),ios::app);
    centroids_vec_f.open((full_path + "centroids_vec_f.txt").c_str(),ios::app);
    plane_area_f.open((full_path + "plane_area.txt").c_str(),ios::app);
    narrow_distance_f.open((full_path + "narrow_distance.txt").c_str(),ios::app);

}

void of_driving::closeFiles(){
    //nofilt_barFile.close();
    //filt_barFile.close();
    theta_f.close();
    pan_f.close();
    //angularVel_f.close();
    error_f.close();
    xl_f.close();
    xr_f.close();
    //R_f.close();
    vx_f.close();
    vy_f.close();
    wz_f.close();
    det_f.close();
    Ju_f.close();
    J_f.close();
    centroids_vec_f.close();
    plane_area_f.close();
    narrow_distance_f.close();
}


//----DEPRECATED-----------------------------------//
void of_driving::computePotentialField(){

	for (int i = 0 ; i < img_height ; i ++){
		unsigned char* dp_ptr = dominant_plane.ptr<uchar>(i);
		Point2f* pot_ptr = potential_field.ptr<Point2f>(i);
		Point2f* gp_ptr = gradient_field.ptr<Point2f>(i);
		Point2f* plan_ptr = planar_flow.ptr<Point2f>(i);
		for (int j = 0 ; j < img_width ; j ++){
			//Scalar dp = dominant_plane.at<uchar>(i,j);
			if(dp_ptr[j] == 255){
				//potential_field.at<Point2f>(i,j) = gradient_field.at<Point2f>(i,j) - planar_flow.at<Point2f>(i,j);
				pot_ptr[j] = gp_ptr[j] - plan_ptr[j];
			}
			else{
				//potential_field.at<Point2f>(i,j) = gradient_field.at<Point2f>(i,j);
				pot_ptr[j] = gp_ptr[j];
			}
		}
	}//*/

}
/**************************************************************/
