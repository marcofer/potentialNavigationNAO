#include "of_driving.h"


using namespace std;
using namespace cv;

void ParallelDominantPlaneFromMotion::operator()(const cv::Range& range) const{

    for(int k = range.start; k < range.end; k++)
    {
        cv::Mat dp_rect(dp,cv::Rect(0,dp.rows/coreNum*k,dp.cols,dp.rows/coreNum));
        cv::Mat op_rect(op,cv::Rect(0,op.rows/coreNum*k,op.cols,op.rows/coreNum));
        cv::Mat of_rect(of,cv::Rect(0,of.rows/coreNum*k,of.cols,of.rows/coreNum));
        cv::Mat fd_rect(fd,cv::Rect(0,fd.rows/coreNum*k,fd.cols,of.rows/coreNum));

        Eigen::Vector2d pp(ppoint.x,ppoint.y), p2;
        Eigen::Matrix<double,2,6> J;
        Eigen::Matrix<double,6,1> vel;
        vel << v, 0, 0, 0, 0, w;

        int rows = dp_rect.rows;
        int cols = dp_rect.cols;


        for (int i = 0 ; i < rows ; i ++){
            unsigned char* dp_ptr = dp_rect.ptr<uchar>(i);
            Point2f* of_ptr = of_rect.ptr<Point2f>(i);
            Point2f* fd_ptr = fd_rect.ptr<Point2f>(i);
            for (int j = 0 ; j < cols ; j ++){
                Eigen::Vector2d p((double)j,(double)i + k*dp_rect.rows);
                p -= pp;

                double x = p(0);
                double y = p(1);
                double Z = hc/cos(gamma + atan2(y,f));


                J <<    - f/Z,     0, x/Z,       x*y/f, -(f + x*x/f),  y,
                            0,  -f/Z, y/Z, (f + y*y/f),       -x*y/f, -x;

                J = J*W;

                p2 = J*vel*Tc;

                fd_ptr[j] = Point2f(p2(0),p2(1));

                /*Point2f xdot(of_ptr[j]);
                Point2f xhat(fd_ptr[j]);

                if( norm(xdot - xhat) < epsilon ) {
                    dp_ptr[j] = 255;
                }
                else{
                    dp_ptr[j] = 0.0;
                }//*/

            }
        }

        /*for (int i = 0 ; i < rows ; i ++){
            unsigned char* dp_ptr = dp_rect.ptr<uchar>(i);
            unsigned char* op_ptr = op_rect.ptr<uchar>(i);
            for (int j = 0 ; j < cols ; j ++){
                    dp_ptr[j] = low_pass_filter(dp_ptr[j],op_ptr[j],Tc,1.0/(cut_f));
            }
        }

        dp_rect.copyTo(op_rect);

        double thresh = dp_threshold; //100
        double maxVal = 255;
        threshold(dp_rect,dp_rect,thresh,maxVal,THRESH_BINARY);//*/


    }
}


void ParallelDominantPlaneBuild::operator()(const cv::Range& range) const{
     
    for(int k = range.start; k < range.end; k++)
    {

        cv::Mat dp_rect(dp,cv::Rect(0,dp.rows/coreNum*k,dp.cols,dp.rows/coreNum));
        cv::Mat invdp_rect(inv_dp,cv::Rect(0,inv_dp.rows/coreNum*k,inv_dp.cols,inv_dp.rows/coreNum));
        cv::Mat op_rect(op,cv::Rect(0,op.rows/coreNum*k,op.cols,op.rows/coreNum));
        cv::Mat of_rect(of,cv::Rect(0,of.rows/coreNum*k,of.cols,of.rows/coreNum));
        cv::Mat pf_rect(pf,cv::Rect(0,pf.rows/coreNum*k,pf.cols,pf.rows/coreNum));

        cv::Mat nf_dp_rect(nf_dp,cv::Rect(0,nf_dp.rows/coreNum*k,nf_dp.cols,nf_dp.rows/coreNum));
        cv::Mat nf_of_rect(nf_of,cv::Rect(0,nf_of.rows/coreNum*k,nf_of.cols,nf_of.rows/coreNum));
        cv::Mat nf_pf_rect(nf_pf,cv::Rect(0,nf_pf.rows/coreNum*k,nf_pf.cols,nf_pf.rows/coreNum));

        for (int i = 0 ; i < pf_rect.rows ; i ++){
            Point2f* i_ptr = pf_rect.ptr<Point2f>(i);
            Point2f* nf_i_ptr = nf_pf_rect.ptr<Point2f>(i);
            for (int j = 0 ; j < pf_rect.cols ; j ++){
                Matx21f p(j,i + k*dp.rows/coreNum);

                Matx21f planar_vec((A*p + b - p));
                Matx21f nf_planar_vec((nf_A*p + nf_b - p));

                Point2f planar_p(planar_vec(0),planar_vec(1));
                Point2f nf_planar_p(nf_planar_vec(0),nf_planar_vec(1));

                i_ptr[j] = planar_p;
                nf_i_ptr[j] = nf_planar_p;
            }
        }//*/

        int rows = dp_rect.rows;
        int cols = dp_rect.cols;
        
        if (dp_rect.isContinuous() && of_rect.isContinuous() && pf_rect.isContinuous() &&
            nf_dp_rect.isContinuous() && nf_of_rect.isContinuous() && nf_pf_rect.isContinuous()){
                cols *= rows;
                rows = 1;
        }//*/


        for (int i = 0 ; i < rows ; i ++){
            unsigned char * dp_ptr = dp_rect.ptr<uchar>(i);
            Point2f* of_ptr = of_rect.ptr<Point2f>(i);
            Point2f* pf_ptr = pf_rect.ptr<Point2f>(i);

            unsigned char * nf_dp_ptr = nf_dp_rect.ptr<uchar>(i);
            Point2f* nf_of_ptr = nf_of_rect.ptr<Point2f>(i);
            Point2f* nf_pf_ptr = nf_pf_rect.ptr<Point2f>(i);

            for (int j = 0 ; j < cols ; j ++){
                Point2f xdot(of_ptr[j]);
                Point2f xhat(pf_ptr[j]);

                Point2f nf_xdot(nf_of_ptr[j]);
                Point2f nf_xhat(nf_pf_ptr[j]);

                if( norm(xdot - xhat) < epsilon ) {
                    dp_ptr[j] = 255;
                }
                else{
                    dp_ptr[j] = 0.0;
                }

                if( norm(nf_xdot - nf_xhat) < epsilon ) {
                    nf_dp_ptr[j] = 255;
                }
                else{
                    nf_dp_ptr[j] = 0.0;
                }
            }
        }
        
        for (int i = 0 ; i < rows ; i ++){
            unsigned char* dp_ptr = dp_rect.ptr<uchar>(i);
            unsigned char* op_ptr = op_rect.ptr<uchar>(i);
            for (int j = 0 ; j < cols ; j ++){
                    dp_ptr[j] = low_pass_filter(dp_ptr[j],op_ptr[j],Tc,1.0/(cut_f));
            }
        }   

        dp_rect.copyTo(op_rect);

        double thresh = dp_threshold; //100
        double maxVal = 255;
        threshold(dp_rect,dp_rect,thresh,maxVal,THRESH_BINARY);//*/

        //cv::bitwise_not(dp_rect,invdp_rect);
    }//*/
}   

void ParallelGradientFieldBuild::operator()(const cv::Range& range) const {

    for (int k = range.start ; k < range.end ; k ++){

        cv::Mat dp_rect(dp,cv::Rect(0,dp.rows/coreNum*k,dp.cols,dp.rows/coreNum));
        cv::Mat sp_rect(sp,cv::Rect(0,sp.rows/coreNum*k,sp.cols,sp.rows/coreNum));
        cv::Mat gf_rect(gf,cv::Rect(0,gf.rows/coreNum*k,gf.cols,gf.rows/coreNum));//*/
        cv::Mat vf_rect(vf,cv::Rect(0,vf.rows/coreNum*k,vf.cols,vf.rows/coreNum));//*/

        cv::Mat nf_dp_rect(nf_dp,cv::Rect(0,nf_dp.rows/coreNum*k,nf_dp.cols,dp.rows/coreNum));
        cv::Mat nf_sp_rect(nf_sp,cv::Rect(0,nf_sp.rows/coreNum*k,nf_sp.cols,sp.rows/coreNum));
        cv::Mat nf_gf_rect(nf_gf,cv::Rect(0,nf_gf.rows/coreNum*k,nf_gf.cols,gf.rows/coreNum));
        cv::Mat nf_vf_rect(nf_vf,cv::Rect(0,nf_vf.rows/coreNum*k,nf_vf.cols,vf.rows/coreNum));

        Size GaussSize(51,51);
        Mat grad_x;
        Mat grad_y;
        Mat nf_grad_x;
        Mat nf_grad_y;

        double sigmaX = dp_rect.rows*dp_rect.cols*0.5;
        int ddepth = CV_32F; //CV_16S
        double delta = 0.0;

        ///TEST VORTEX!!!

        GaussianBlur(dp_rect,sp_rect,GaussSize,sigmaX,0);
        GaussianBlur(nf_dp_rect,nf_sp_rect,GaussSize,sigmaX,0);
        //dp_rect.copyTo(sp_rect);
        //nf_dp_rect.copyTo(nf_sp_rect);
        Scharr(sp_rect, grad_x, ddepth, 1, 0, scale, 0, BORDER_REPLICATE);
        Scharr(sp_rect, grad_y, ddepth, 0, 1, scale, 0, BORDER_REPLICATE);
        
        Scharr(nf_sp_rect, nf_grad_x, ddepth, 1, 0, scale, 0, BORDER_REPLICATE);
        Scharr(nf_sp_rect, nf_grad_y, ddepth, 0, 1, scale, 0, BORDER_REPLICATE);


        int rows = dp_rect.rows;
        int cols = dp_rect.cols;

        if(grad_x.isContinuous() && grad_y.isContinuous() && gf_rect.isContinuous() &&
           nf_grad_x.isContinuous() && nf_grad_y.isContinuous() && nf_gf_rect.isContinuous()){
            cols *= rows;
            rows = 1;
        }

        for (int i = 0 ; i < dp_rect.rows ; i ++){
            float* x_ptr = grad_x.ptr<float>(i);
            float* y_ptr = grad_y.ptr<float>(i);
            Point2f* grad_ptr = gf_rect.ptr<Point2f>(i);
            Point2f* vtx_ptr = vf_rect.ptr<Point2f>(i);

            float* nf_x_ptr = nf_grad_x.ptr<float>(i);
            float* nf_y_ptr = nf_grad_y.ptr<float>(i);
            Point2f* nf_grad_ptr = nf_gf_rect.ptr<Point2f>(i);
            Point2f* nf_vtx_ptr = nf_vf_rect.ptr<Point2f>(i);

            for (int j = 0 ; j < dp_rect.cols ; j ++){

                //GRADIENT FIELD
                grad_ptr[j] = Point2f(x_ptr[j],y_ptr[j]);
                nf_grad_ptr[j] = Point2f(nf_x_ptr[j],nf_y_ptr[j]);//*/

                //VORTEX FIELD //PAY ATTENTION TO += ! YOU'RE TRYING TO COMBINE THE FIELDS!!!
                if(x_ptr[j] > 0){
                    x_ptr[j] *= -1;
                    y_ptr[j] *= -1;
                }
                vtx_ptr[j] = Point2f(-y_ptr[j],x_ptr[j]);
                nf_vtx_ptr[j] = Point2f(-nf_y_ptr[j],nf_x_ptr[j]);//*/
            }
        }//*/

    }

}


void ParallelOpticalFlow::operator()(const cv::Range& range) const {

    for (int k = range.start ; k < range.end ; k ++){

        /*cv::Mat img_rect(img,cv::Rect(0,img.rows/coreNum*k,img.cols,img.rows/coreNum));
        cv::Mat img_rect2(img2,cv::Rect(0,img.rows/coreNum*k,img.cols,img.rows/coreNum));
        cv::Mat of_rect(optical_flow,cv::Rect(0,optical_flow.rows/coreNum*k,optical_flow.cols,optical_flow.rows/coreNum));//*/
        
        cv::Mat img_rect(img,cv::Rect(img.cols/coreNum*k,0,img.cols/coreNum,img.rows));
        cv::Mat img_rect2(img2,cv::Rect(img2.cols/coreNum*k,0,img2.cols/coreNum,img2.rows));
        cv::Mat of_rect(optical_flow,cv::Rect(optical_flow.cols/coreNum*k,0,optical_flow.cols/coreNum,optical_flow.rows));//*/
        
        calcOpticalFlowFarneback(img_rect, img_rect2, of_rect, pyr_scale, maxLayer, winSize, of_iterations, poly_n, poly_sigma, flags);//*/

        /*cv::gpu::GpuMat img_rect(img,cv::Rect(0,img.rows/coreNum*k,img.cols,img.rows/coreNum));
        cv::gpu::GpuMat img2_rect(img2,cv::Rect(0,img2.rows/coreNum*k,img2.cols,img2.rows/coreNum));
        cv::gpu::GpuMat u_rect(u_flow,cv::Rect(0,u_flow.rows/coreNum*k,u_flow.cols,u_flow.rows/coreNum));
        cv::gpu::GpuMat v_rect(v_flow,cv::Rect(0,v_flow.rows/coreNum*k,v_flow.cols,v_flow.rows/coreNum));
        cv::Mat of_rect(optical_flow,cv::Rect(0,optical_flow.rows/coreNum*k,optical_flow.cols,optical_flow.rows/coreNum));

        //cout << "img.rows/coreNum*k:" << img.rows/coreNum*k << endl;

        farneback_flow(img_rect,img2_rect,u_rect,v_rect);
        getFlowField(Mat(u_rect),Mat(v_rect),of_rect);//*/
    }

}


void ParallelDisplayImages::operator()(const cv::Range& range) const{

    for (int k = range.start ; k < 6 ; k ++){
        
        if(k == 1){
            Mat u_img, gImg;
            //cvtColor(img,gImg,CV_BGR2GRAY);

            //img.copyTo(gImg);
            gImg = img.clone();

            cvtColor(gImg,u_img,CV_GRAY2BGR);

            for (int i = 0 ; i < img.rows ; i+= flowResolution*2){
                const Point2f* of_ptr = of.ptr<Point2f>(i);
                for (int j = 0 ; j < img.cols ; j += flowResolution*2){
                    cv::Point2f p(j,i);
                    cv::Point2f p2(p + of_ptr[j]);
                    arrowedLine2(u_img,p,p2,Scalar(0,0,255),0.1,8,0,0.1); 
                }
            }

            u_img.copyTo(total(Rect(img.cols,0,img.cols,img.rows)));
        }

        if(k == 2){
            Mat p_img, gImg;
            //cvtColor(img,gImg,CV_BGR2GRAY);
            //img.copyTo(gImg);
            gImg = img.clone();


            cvtColor(gImg,p_img,CV_GRAY2BGR);

            for (int i = 0 ; i < img.rows ; i+= flowResolution*2){
                const Point2f* pf_ptr = pf.ptr<Point2f>(i);
                for (int j = 0 ; j < img.cols ; j += flowResolution*2){
                    cv::Point2f p(j,i);
                    cv::Point2f p2(p + pf_ptr[j]);
                    arrowedLine2(p_img,p,p2,Scalar(255,255,0),0.1,8,0,0.1);
                }
            }
            p_img.copyTo(total(Rect(img.cols*2,0,img.cols,img.rows)));
        }        


        if(k == 3){
            Mat dp_img;
            cvtColor(dp,dp_img,CV_GRAY2BGR);
            rectangle(dp_img,dpROI.tl(),dpROI.br(),Scalar(0,255,0),4);
            Point2f p(dp.cols/2,dp.rows - 10);
            //circle(dp_img,p,3,Scalar(0,0,255),2);
            dp_img.copyTo(total(Rect(0,img.rows,img.cols,img.rows)));
        }        

        if(k == 5){
            /*Mat sp_img;
            cvtColor(sp,sp_img,CV_GRAY2BGR);
            sp_img.copyTo(total(Rect(0,img.rows,img.cols,img.rows)));//*/
            Mat cf_img;
            //img.copyTo(cf_img);
            cf_img = Mat::zeros(img.size(),CV_8UC1);
            cvtColor(cf_img,cf_img,CV_GRAY2BGR);

            double dmax = 100.0;
            double pxmax = 20.0;
            Point2f center(img.cols/2,img.rows/2);
            Point2f pb(p_bar(0),p_bar(1));
            Point2f y(0,-1);
            Point2f Vy(vy,0);

            pb.x = pb.x*dmax/pxmax;
            Vy.x = Vy.x*dmax/pxmax;

            //circle 1
            int c_radius = 20;
            Point c_center(img.cols/4.0,img.rows/4.0);
            Point pan_line(-c_radius*sin(pan),-c_radius*cos(pan));
            Point real_pan_line(-c_radius*sin(real_pan),-c_radius*cos(real_pan));
            circle(cf_img,c_center,c_radius,Scalar(255,0,0));
            line(cf_img,c_center,c_center + pan_line,Scalar(255,255,0));
            //line(cf_img,c_center,c_center + Point(0,-c_radius),Scalar(0,255,255));
            line(cf_img,c_center,c_center + real_pan_line,Scalar(0,0,255));


            //circle 2
            Point c_center2(3.0*img.cols/4.0,img.rows/4.0);
            Point theta_des_line(-c_radius*sin(theta_des),-c_radius*cos(theta_des));
            circle(cf_img,c_center2,c_radius,Scalar(255,0,0));
            line(cf_img,c_center2,c_center2 + theta_des_line,Scalar(255,255,0));
            //line(cf_img,c_center2,c_center2 + Point(0,-c_radius),Scalar(0,255,255));


            //bar_length
            double bar_length = img.cols/2 - 10.0;
            double bar_height = 10.0;

            //v_bar
            Point2f v1(1.0/2.0*img.cols,1.0/2.0*img.rows);
            Point2f v2 = v1 + Point2f(bar_length,bar_height);
            rectangle(cf_img,v1,v2,Scalar(100,100,100),2);

            Point2f vy1(1.0/2.0*img.cols, v1.y + 2.0*bar_height);
            Point2f vy2 = vy1 + Point2f(bar_length,bar_height);
            rectangle(cf_img,vy1,vy2,Scalar(100,100,100),2);


            //w_bar
            const double angular_vel_max = wmax;
            const double linear_vel_max = vmax;

            double w_value = wz * (bar_length/2) / angular_vel_max ;
            double v_value = (linear_vel)/linear_vel_max*(bar_length/2);
            double vy_value = (vy)/linear_vel_max*(bar_length/2);

            double w_red = abs(w_value)/(bar_length/2)*255.0 ;
            double w_green = 255.0 - w_red;
            double v_red = abs(v_value)/(bar_length/2)*255.0 ;
            double v_green = 255.0 - v_red;
            double vy_red = abs(vy_value)/(bar_length/2)*255.0 ;
            double vy_green = 255.0 - vy_red;

            Point2f w1(1.0/2.0*img.cols,v1.y + 2.0*bar_height + 2.0*bar_height );
            Point2f w2 = w1 + Point2f(bar_length,bar_height);
            rectangle(cf_img,w1,w2,Scalar(100,100,100),2);

            string text_str;
            ostringstream convert;
            Size v_size, vvalue_size, ms_size;
            Size w_size, wvalue_size, rads_size;
            double font_scale = 0.9;
            double font_scale2 = 0.6;

            text_str = "";
            text_str = "vx = ";
            v_size = getTextSize(text_str,1,font_scale,1,0);
            putText(cf_img, text_str,Point(10,v2.y),1,font_scale,Scalar(255,255,255),1,CV_AA);

            text_str = "";
            convert.str(""); convert.clear();
            convert << setprecision(4) << linear_vel;
            text_str = convert.str();
            vvalue_size = getTextSize(text_str,1,font_scale,1,0);
            putText(cf_img, text_str,Point(v_size.width + 10,v2.y),1,font_scale,Scalar(255,255,255),1,CV_AA);

            text_str = "";
            text_str = "[m/s]";
            ms_size = getTextSize(text_str,1,font_scale2,1,0);
            putText(cf_img, text_str,Point(img.cols/2-ms_size.width -10,v2.y),1,font_scale2,Scalar(255,255,255),1,CV_AA);


            text_str = "";
            text_str = "vy = ";
            v_size = getTextSize(text_str,1,font_scale,1,0);
            putText(cf_img, text_str,Point(10,vy2.y),1,font_scale,Scalar(255,255,255),1,CV_AA);

            text_str = "";
            convert.str(""); convert.clear();
            convert << setprecision(4) << vy;
            text_str = convert.str();
            vvalue_size = getTextSize(text_str,1,font_scale,1,0);
            putText(cf_img, text_str,Point(v_size.width + 10,vy2.y),1,font_scale,Scalar(255,255,255),1,CV_AA);

            text_str = "";
            text_str = "[m/s]";
            ms_size = getTextSize(text_str,1,font_scale2,1,0);
            putText(cf_img, text_str,Point(img.cols/2-ms_size.width -10,vy2.y),1,font_scale2,Scalar(255,255,255),1,CV_AA);


            text_str = "";
            text_str = "w = ";
            w_size = getTextSize(text_str,1,font_scale,1,0);
            putText(cf_img, text_str,Point(10,w2.y),1,font_scale,Scalar(255,255,255),1,CV_AA);

            text_str = "";
            convert.str(""); convert.clear();
            convert << setprecision(4) << wz;
            text_str = convert.str();
            wvalue_size = getTextSize(text_str,1,font_scale,1,0);
            putText(cf_img, text_str,Point(w_size.width + 10,w2.y),1,font_scale,Scalar(255,255,255),1,CV_AA);

            text_str = "";
            text_str = "[rad/s]";
            rads_size = getTextSize(text_str,1,font_scale2,1,0);
            putText(cf_img, text_str,Point(img.cols/2-rads_size.width -10,w2.y),1,font_scale2,Scalar(255,255,255),1,CV_AA);


            line(cf_img,Point2f((w1+w2)*0.5),Point2f((w1+w2)*0.5) + Point2f(w_value,0), Scalar(0,w_green,w_red),5.0,2,0);
            line(cf_img,Point2f((v1+v2)*0.5),Point2f((v1+v2)*0.5) + Point2f(v_value,0), Scalar(0,v_green,v_red),5.0,2,0);
            line(cf_img,Point2f((vy1+vy2)*0.5),Point2f((vy1+vy2)*0.5) + Point2f(vy_value,0), Scalar(0,v_green,v_red),5.0,2,0);
            circle(cf_img,Point2f((w1+w2)*0.5),3,Scalar(0,0,255),2);
            circle(cf_img,Point2f((v1+v2)*0.5),3,Scalar(0,0,255),2);


            // Navigation Vector visual information
            //arrowedLine2(cf_img,center,center + y*50,Scalar(255,0,0),3.0,8,0,0.1);
            //arrowedLine2(cf_img,center,center + pb,Scalar(0,255,0),3.0,1,0,0.1);
            //arrowedLine2(cf_img,center,center + Vy,Scalar(255,255,0),3.0,1,0,0.1);

            cf_img.copyTo(total(Rect(img.cols*2,img.rows,img.cols,img.rows)));

        }       

        if(k == 4){
            /*Mat gradient_img;
            cvtColor(sp,gradient_img,CV_GRAY2BGR);

            for (int i = 0 ; i < img.rows ; i+= flowResolution*2){
                const Point2f* gf_ptr = gf.ptr<Point2f>(i);
                for (int j = 0 ; j < img.cols ; j += flowResolution*2){
                    cv::Point2f p(j,i);
                    cv::Point2f p2(p + gf_ptr[j]*0.1);
                    arrowedLine2(gradient_img,p,p2,Scalar(0,255,0),0.1,8,0,0.1);
                }
            }
            gradient_img.copyTo(total(Rect(img.cols,img.rows,img.cols,img.rows)));//*/

            //SHOW CENTROIDS
            //Draw contours' centers
            Mat centroid_img;
            //dp.copyTo(centroid_img);
            centroid_img = dp.clone();


            bitwise_not(centroid_img,centroid_img);
            cvtColor(centroid_img,centroid_img,CV_GRAY2BGR);

            Point2f xr_n = xr - Point2f(160,60);
            Point2f xl_n = xl - Point2f(160,60);

            Point2f err = xr_n + xl_n;
            err.y = err.y/2;
            err = err + Point2f(160,60);

            for (int i = 0 ; i < rc.size() ; i ++){
                circle(centroid_img,rc[i],4,Scalar(255,0,0),-1,8,0);
            }
            for (int i = 0 ; i < lc.size() ; i ++){
                circle(centroid_img,lc[i],4,Scalar(255,0,0),-1,8,0);
            }

            Scalar l_color, r_color;
            Scalar gray(50,50,50);
            Scalar red(0,0,255);

            /*   if(xr.x > centroid_img.cols - px_margin){
                r_color = gray;
            }
            else{
                r_color = red;
            }//*/
            l_color = red;
            r_color = red;


            //Draw centroids
            circle(centroid_img,xr,4,r_color,-1,8,0);
            circle(centroid_img,xl,4,l_color,-1,8,0);

            //Draw error centroid
            circle(centroid_img,err,4,Scalar(255,0,255),-1,8,0);


            if(narrowCheck){
                // Draw min-max points of contours
                circle(centroid_img,maxLP,4,Scalar(255,255,0),-1,8,0);
                circle(centroid_img,minRP,4,Scalar(255,255,0),-1,8,0);
                circle(centroid_img,middleP,4,Scalar(255,255,0),-1,8,0);
                line(centroid_img,minRP,maxLP,Scalar(255,255,0));
            }

            string text_str;
            ostringstream convert;
            text_str = "";
            text_str = "min x-distance: ";
            float text_scale = 0.6;
            Size text_size = getTextSize(text_str,1,text_scale,1,0);
            Point text_point(img.cols/2 - text_size.width,img.rows - text_size.height - 5);
            putText(centroid_img, text_str,text_point,1,text_scale,Scalar(0,0,255),1,CV_AA);

            text_str = "";
            convert.str(""); convert.clear();
            convert << setprecision(4) << narrow_width;
            text_str = convert.str();
            Size minMaxSize = getTextSize(text_str,1,text_scale,1,0);
            Point minMaxPoint = Point(text_point.x + text_size.width + 5, text_point.y);
            putText(centroid_img, text_str,minMaxPoint,1,text_scale,Scalar(0,0,255),1,CV_AA);

            line(centroid_img,Point(centroid_img.cols/2,0),Point(centroid_img.cols/2,centroid_img.rows),Scalar(30,30,30));
            line(centroid_img,Point(0,centroid_img.rows/2),Point(centroid_img.cols,centroid_img.rows/2),Scalar(30,30,30));

            // Draw contours
            for (int i = 0 ; i < good_contours.size() ; i ++){
                Scalar green(0,255,0);
                drawContours(centroid_img,good_contours,i,green,3);
            }


            Scalar green(0,255,0);
            Scalar purple(255,0,255);
            //line(centroid_img,Point(centroid_img.cols/2 + xmin,0),Point(centroid_img.cols/2 + xmin,centroid_img.rows),green);
            //line(centroid_img,Point(centroid_img.cols/2 + xmax,0),Point(centroid_img.cols/2 + xmax,centroid_img.rows),purple);
            //line(centroid_img,Point(px_margin,0),Point(px_margin,centroid_img.rows),Scalar(255,0,0));
            //line(centroid_img,Point(centroid_img.cols - px_margin,0),Point(centroid_img.cols - px_margin,centroid_img.rows),Scalar(255,0,0));

            centroid_img.copyTo(total(Rect(img.cols,img.rows,img.cols,img.rows)));//*/
        }        

        if(k == 0){
            Mat copy_img;
            cvtColor(img,copy_img,CV_GRAY2BGR);

            copy_img.copyTo(total(Rect(0,0,img.cols,img.rows)));//*/

        }//*/



    }

}
