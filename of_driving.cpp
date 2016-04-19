#include "of_driving.h"

#include "math.h"
#include <sys/time.h>
#include <fstream>

using namespace cv::gpu;

enum OF_ALG{LK,FARNEBACK,GPU_LK_SPARSE,GPU_LK_DENSE,GPU_FARNEBACK,GPU_BROX};

of_driving::of_driving(){

    Rm = ANGULAR_VEL_MAX/8.0;
    linear_vel = LINEAR_VEL_MAX/2.0;

    grad_scale = 1.0;
    windows_size = 13.0;//44.0;//13.0;
    maxLayer = 2;
    epsilon = 0.7;//0.5//0.4;//0.8;
    flowResolution = 4;
    iteration_num = 10;
    of_iterations = 3;//3;

    open_erode = 1.0;//4.9;
    open_dilate = 1.0;
    close_erode = 1.0;
    close_dilate = 1.0;
    of_alg = 3;
    of_scale = 1;
    RANSAC_imgPercent = 0.5;
    dp_threshold = 40;
    wheelbase = 2.06;

    open_erode_int = open_erode*10.0;
	close_erode_int = close_erode*10.0;
	open_dilate_int = open_dilate*10.0;
	close_dilate_int = close_dilate*10.0;
    
	Rold = 0.0;
	px_old = 0.0;
	py_old = 0.0;


	angular_vel = 0.0;

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

}

of_driving::~of_driving(){
	destroyAllWindows();

}

void of_driving::set_imgSize(int w, int h){
	img_width = w;
	img_height = h;
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

    noFilt_of = Mat::zeros(img_height,img_width,CV_32FC2);
    noFilt_pf = Mat::zeros(img_height,img_width,CV_32FC2);
    noFilt_dp = Mat::zeros(img_height,img_width,CV_8UC1);
    noFilt_best = Mat::zeros(img_height,img_width,CV_8UC1);
    noFilt_sp = Mat::zeros(img_height,img_width,CV_8UC1);
    noFilt_gf = Mat::zeros(img_height,img_width,CV_32FC2);
    noFilt_vf = Mat::zeros(img_height,img_width,CV_32FC2);
    noFilt_rf = Mat::zeros(img_height,img_width,CV_32FC2);

    point_counter = 0;
	best_counter = 0;
    nf_point_counter = 0;
    nf_best_counter = 0;

	area_ths = 50;

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

    createButton("OpenClose",callbackButton,&open_close,CV_CHECKBOX,0);


    createTrackbar("winSize",of_alg_name,&windows_size,51,NULL);

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
    createTrackbar("C_erode*10",of_alg_name,&close_erode_int,200,NULL);//*/
    createTrackbar("field weights",of_alg_name,&weight_int,10,NULL);//*/
    //createTrackbar("area_ths",of_alg_name,&area_ths,1000,NULL);


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

void of_driving::run(Mat& img, Mat& prev_img, bool save_video){


	Rect rect_ransac(ROI_x,ROI_y,ROI_width,ROI_height);
	Mat ROI_ransac = dominant_plane(rect_ransac);

	epsilon = (double)eps_int/100.0;
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

	prev_img.copyTo(image);

    //cvtColor(img,GrayImg,CV_BGR2GRAY);
    //cvtColor(prev_img,GrayPrevImg,CV_BGR2GRAY);
    img.copyTo(GrayImg);
    prev_img.copyTo(GrayPrevImg);


	GpuMat gpu_prevImg(GrayPrevImg);
	GpuMat gpu_Img(GrayImg);

    const int64 start = getTickCount();

	/// ---  1. Compute the optical flow field u(x,y,t) (output: optical_flow matrix)
    computeOpticalFlowField(GrayPrevImg,GrayImg);
    //computeOpticalFlowField(gpu_prevImg,gpu_Img);

    //computeFlowDirection();

	while(point_counter <= max_counter && k < iteration_num){


        // --- 2. Compute affine coefficients by random selection of three points 
		estimateAffineCoefficients(false,GrayPrevImg,ROI_ransac,rect_ransac);
		

		// --- 3-4. Estimate planar flow from affine coefficients and Match the computed optical flow and esitmated planar flow, so detect the dominant plane. If the dominant plane occupies
        // less than half of the image, then go to step (2)
		buildPlanarFlowAndDominantPlane(ROI_ransac);
		
        
		if(point_counter >= best_counter){
			best_counter = point_counter;
			dominant_plane.copyTo(best_plane);
		}

        if(nf_point_counter >= nf_best_counter){
            nf_best_counter = point_counter;
            noFilt_dp.copyTo(noFilt_best);
        }

		k++;
	}

	if(point_counter <= max_counter){
		best_plane.copyTo(dominant_plane);
        noFilt_best.copyTo(noFilt_dp);
	}


    /// --- 2. Robust computation of affine coefficients with all the points belonging to the detected dominant plane 	
    estimateAffineCoefficients(true,GrayPrevImg,ROI_ransac,rect_ransac);//*/

	/// --- 3. 

    /// --- 3-4. Estimate planar flow from affine coefficients and Match the computed optical flow and esitmated planar flow, so detect the dominant plane
    buildPlanarFlowAndDominantPlane(ROI_ransac);

    /*cout << "optical_flow.at<Point2f>(img_height - 10,img_width/2): " << optical_flow.at<Point2f>(img_height - 10,img_width/2) << endl;
    cout << "planar.at<Point2f>(img_height - 10,img_width/2): " << planar_flow.at<Point2f>(img_height - 10,img_width/2) << endl ;

    if(norm(optical_flow.at<Point2f>(img_height - 10,img_width/2) - planar_flow.at<Point2f>(img_height - 10,img_width/2)) > 0.5){
        cout << "!!!!!!!!!!!!!!!!!!norm(diff): " << norm(optical_flow.at<Point2f>(img_height - 10,img_width/2) - planar_flow.at<Point2f>(img_height - 10,img_width/2)) << endl << endl;
    }
    else{
        cout << "norm(diff): " << norm(optical_flow.at<Point2f>(img_height - 10,img_width/2) - planar_flow.at<Point2f>(img_height - 10,img_width/2)) << endl << endl;
    }//*/


    /// --- 5. Compute gradient vector field from dominant plane 
    computeGradientVectorField();

    /// --- 6. Compute the desired planar flow, from the theta_d <--- NOT REQUIRED HERE
    //computePotentialField();

    /// --- 7. Compute the control force as average of potential field
    computeControlForceOrientation(); 

    /// --- 8. Compute the translational and rotational robot velocities
    computeRobotVelocities();
	
    /// --- END. Show the intermediate steps
	Mat total = Mat::zeros(2*img_height,3*img_width,CV_8UC3);
	

	/*** MULTI-THREADED DISPLAY ***/
    parallel_for_(Range(0,6),ParallelDisplayImages(6,flowResolution,prev_img,optical_flow,planar_flow,dominant_plane,smoothed_plane,dominantHull,result_field,p_bar,angular_vel,total,rect_ransac, theta, linear_vel, Rm, noFilt_pbar));
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

	img.convertTo(img32F,CV_32F,1.0/255.0);
	prevImg.convertTo(prevImg32F,CV_32F,1.0/255.0);
	GpuMat gpuImg32F(img32F);
	GpuMat gpuPrevImg32F(prevImg32F);//*/
	GpuMat gpuImg8U(img);
	GpuMat gpuPrevImg8U(prevImg);

	gpu::BroxOpticalFlow brox_flow(alpha,gamma,scale,inner,outer,solver);
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
	    			if((j*flowResolution + flowResolution <= img_width) && (i*flowResolution + flowResolution) <= img_height)
						temp.copyTo(optical_flow(Rect(j*flowResolution,i*flowResolution,flowResolution,flowResolution)));	    		 
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
		case GPU_BROX:
			brox_flow(gpuPrevImg32F,gpuImg32F,u_flow,v_flow);
    		getFlowField(Mat(u_flow),Mat(v_flow),optical_flow);
			break;
		default:
			break;
	}//*/

	/// Scale the Optical Flow field
	optical_flow = optical_flow*of_scale;

	/*dilate(optical_flow, optical_flow, getStructuringElement(MORPH_ELLIPSE, Size(10.0,10.0)));
    erode(optical_flow, optical_flow, getStructuringElement(MORPH_ELLIPSE, Size(10.0,10.0)));//*/

    /// Copy the optical flow field before filtering
    optical_flow.copyTo(noFilt_of);

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
    parallel_for_(Range(0,cores_num),ParallelDominantPlaneBuild(cores_num,dominant_plane,old_plane,optical_flow,planar_flow,epsilon,
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


    nofilt_barFile << noFilt_p.x << ", " << noFilt_p.y << "; " << endl;
    filt_barFile << pbar.x << ", " << pbar.y << "; " << endl;


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

    theta_f << theta << "; " << endl;

    //theta = low_pass_filter(theta,theta_old,Tc,1.0/ctrl_lowpass_freq);
    //cout << "theta: " << theta*180.0/M_PI << endl;

}


void of_driving::computeRobotVelocities(){

	double R = Rm*sin(theta);
	angular_vel = R ;

    angularVel_f << angular_vel << "; " << endl;
    //cout << "w: " << angular_vel << endl;
	//R = low_pass_filter(R,Rold,Tc,1.0/ctrl_lowpass_freq);
	//Rold = R;

	if(ankle_angle != -1){
        steering = wheelbase * R/linear_vel;
    }
    else{
        steering = 0.0;
    }

	ankle_angle = 0.0;
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
    nofilt_barFile.open((full_path + "nofiltp.txt").c_str(),ios::app);
    filt_barFile.open((full_path + "filtp.txt").c_str(),ios::app);
    theta_f.open((full_path + "theta.txt").c_str(),ios::app);
    angularVel_f.open((full_path + "angularVel.txt").c_str(),ios::app);
}

void of_driving::closeFiles(){
    nofilt_barFile.close();
    filt_barFile.close();
    theta_f.close();
    angularVel_f.close();
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


/** Auxiliary functions **/

//double of_driving::low_pass_filter(double in, double out_old, double Tc, double tau){
double low_pass_filter(double in, double out_old, double Tc, double tau){
    //cout << "filtering" << endl;

    double out, alpha;

    //alpha = 1 / ( 1 + (tau/Tc));

    alpha = Tc/tau;

    out = alpha * in + (1-alpha) * out_old;

    return out;

}

//double of_driving::high_pass_filter(double in, double in_prev, double out_old, double Tc, double tau){
double high_pass_filter(double in, double in_prev, double out_old, double Tc, double tau){
    //cout << "filtering" << endl;

    double out, alpha;

    //alpha = 1 / ( 1 + (tau/Tc));

    out = alpha *  out_old + alpha * (in - in_prev);

    return out;

}


void arrowedLine2(Mat& img, cv::Point2f pt1, cv::Point2f pt2, const Scalar& color, int thickness, int line_type, int shift, 
    double tipLength)
{
    const double tipSize = norm(pt1-pt2)*tipLength; // Factor to normalize the size of the tip depending on the length of the arrow
    line(img, pt1, pt2, color, thickness, line_type, shift);
    const double angle = atan2( (double) pt1.y - pt2.y, (double) pt1.x - pt2.x );
    Point p(cvRound(pt2.x + tipSize * cos(angle + CV_PI / 4)),
    cvRound(pt2.y + tipSize * sin(angle + CV_PI / 4)));
    line(img, p, pt2, color, thickness, line_type, shift);
    p.x = cvRound(pt2.x + tipSize * cos(angle - CV_PI / 4));
    p.y = cvRound(pt2.y + tipSize * sin(angle - CV_PI / 4));
    line(img, p, pt2, color, thickness, line_type, shift);
}//*/


template <typename T> inline T clamp (T x, T a, T b)
{
    return ((x) > (a) ? ((x) < (b) ? (x) : (b)) : (a));
}

template <typename T> inline T mapValue(T x, T a, T b, T c, T d)
{
    x = clamp(x, a, b);
    return c + (d - c) * (x - a) / (b - a);
}

void getFlowField(const Mat& u, const Mat& v, Mat& flowField)
{
    for (int i = 0; i < flowField.rows; ++i)
    {
        const float* ptr_u = u.ptr<float>(i);
        const float* ptr_v = v.ptr<float>(i);

		Point2f* row = flowField.ptr<Point2f>(i);

        for (int j = 0; j < flowField.cols; ++j)
        {
            row[j].y = ptr_v[j];
            row[j].x = ptr_u[j];
        }
    }


}


