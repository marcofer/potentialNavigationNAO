#include "of_driving.h"


using namespace std;
using namespace cv;


void ParallelDominantPlaneBuild::operator()(const cv::Range& range) const{
     
    for(int k = range.start; k < range.end; k++)
    {

        cv::Mat dp_rect(dp,cv::Rect(0,dp.rows/coreNum*k,dp.cols,dp.rows/coreNum));
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
        
        if(k == 0){
            Mat u_img, gImg;
            //cvtColor(img,gImg,CV_BGR2GRAY);
            img.copyTo(gImg);

            cvtColor(gImg,u_img,CV_GRAY2BGR);

            for (int i = 0 ; i < img.rows ; i+= flowResolution*2){
                const Point2f* of_ptr = of.ptr<Point2f>(i);
                for (int j = 0 ; j < img.cols ; j += flowResolution*2){
                    cv::Point2f p(j,i);
                    cv::Point2f p2(p + of_ptr[j]);
                    arrowedLine2(u_img,p,p2,Scalar(0,0,255),0.1,8,0,0.1); 
                }
            }

            u_img.copyTo(total(Rect(0,0,img.cols,img.rows)));
        }

        if(k == 1){
            Mat p_img, gImg;
            //cvtColor(img,gImg,CV_BGR2GRAY);
            img.copyTo(gImg);
            cvtColor(gImg,p_img,CV_GRAY2BGR);

            for (int i = 0 ; i < img.rows ; i+= flowResolution*2){
                const Point2f* pf_ptr = pf.ptr<Point2f>(i);
                for (int j = 0 ; j < img.cols ; j += flowResolution*2){
                    cv::Point2f p(j,i);
                    cv::Point2f p2(p + pf_ptr[j]);
                    arrowedLine2(p_img,p,p2,Scalar(255,255,0),0.1,8,0,0.1);
                }
            }
            p_img.copyTo(total(Rect(img.cols,0,img.cols,img.rows)));
        }        


        if(k == 2){
            Mat dp_img;
            cvtColor(dp,dp_img,CV_GRAY2BGR);
            rectangle(dp_img,dpROI.tl(),dpROI.br(),Scalar(0,255,0),4);
            Point2f p(dp.cols/2,dp.rows - 10);
            //circle(dp_img,p,3,Scalar(0,0,255),2);
            dp_img.copyTo(total(Rect(2*img.cols,0,img.cols,img.rows)));
        }        

        if(k == 3){
            Mat sp_img;
            cvtColor(sp,sp_img,CV_GRAY2BGR);
            sp_img.copyTo(total(Rect(0,img.rows,img.cols,img.rows)));
        }       

        if(k == 4){
            Mat gradient_img;
            cvtColor(sp,gradient_img,CV_GRAY2BGR);

            for (int i = 0 ; i < img.rows ; i+= flowResolution*2){
                const Point2f* gf_ptr = gf.ptr<Point2f>(i);
                for (int j = 0 ; j < img.cols ; j += flowResolution*2){
                    cv::Point2f p(j,i);
                    cv::Point2f p2(p + gf_ptr[j]*0.1);
                    arrowedLine2(gradient_img,p,p2,Scalar(0,255,0),0.1,8,0,0.1);
                }
            }
            gradient_img.copyTo(total(Rect(img.cols,img.rows,img.cols,img.rows)));
        }        

        if(k == 5){
            Mat cf_img;
            img.copyTo(cf_img);
            cvtColor(cf_img,cf_img,CV_GRAY2BGR);

            double dmax = 100.0;
            double pxmax = 20.0;
            Point2f center(img.cols/2,img.rows/2);
            Point2f pb(p_bar(0),p_bar(1));
            Point2f nfpb(nf_pbar(0),nf_pbar(1));
            Point2f y(0,-1);

            pb.x = pb.x*dmax/pxmax;
            nfpb.x = nfpb.x*dmax/pxmax;

            //bar_length
            double bar_length = img.cols/2 - 10.0;
            double bar_height = 10.0;

            //v_bar
            Point2f v1(1.0/2.0*img.cols,3.0/4.0*img.rows);
            Point2f v2 = v1 + Point2f(bar_length,bar_height);
            rectangle(cf_img,v1,v2,Scalar(100,100,100),2);

            //w_bar
            const double angular_vel_max = w;
            const double linear_vel_max = v;

            double w_value = angular_vel * (bar_length/2) / angular_vel_max ;
            double v_value = (linear_vel_max/2)/linear_vel_max*(bar_length/2);
            
            double w_red = abs(w_value)/(bar_length/2)*255.0 ;
            double w_green = 255.0 - w_red;            
            double v_red = abs(v_value)/(bar_length/2)*255.0 ;
            double v_green = 255.0 - v_red;            

            Point2f w1(1.0/2.0*img.cols,3.0/4.0*img.rows + 2.0*bar_height);
            Point2f w2 = w1 + Point2f(bar_length,bar_height);
            rectangle(cf_img,w1,w2,Scalar(100,100,100),2);

            string text_str;
            ostringstream convert;
            Size v_size, vvalue_size, ms_size;
            Size w_size, wvalue_size, rads_size;
            double font_scale = 0.9;
            double font_scale2 = 0.6;

            text_str = "";
            text_str = "v = ";
            v_size = getTextSize(text_str,1,font_scale,1,0);
            putText(cf_img, text_str,Point(10,v2.y),1,font_scale,Scalar(255,255,255),1,CV_AA);
           
            text_str = "";
            convert.str(""); convert.clear();
            convert << setprecision(4) << linear_vel_max;
            text_str = convert.str();
            vvalue_size = getTextSize(text_str,1,font_scale,1,0);
            putText(cf_img, text_str,Point(v_size.width + 10,v2.y),1,font_scale,Scalar(255,255,255),1,CV_AA);

            text_str = "";
            text_str = "[m/s]";
            ms_size = getTextSize(text_str,1,font_scale2,1,0);
            putText(cf_img, text_str,Point(img.cols/2-ms_size.width -10,v2.y),1,font_scale2,Scalar(255,255,255),1,CV_AA);

            text_str = "";
            text_str = "w = ";
            w_size = getTextSize(text_str,1,font_scale,1,0);
            putText(cf_img, text_str,Point(10,w2.y),1,font_scale,Scalar(255,255,255),1,CV_AA);
           
            text_str = "";
            convert.str(""); convert.clear();
            convert << setprecision(4) << angular_vel;
            text_str = convert.str();
            wvalue_size = getTextSize(text_str,1,font_scale,1,0);
            putText(cf_img, text_str,Point(w_size.width + 10,w2.y),1,font_scale,Scalar(255,255,255),1,CV_AA);

            text_str = "";
            text_str = "[rad/s]";
            rads_size = getTextSize(text_str,1,font_scale2,1,0);
            putText(cf_img, text_str,Point(img.cols/2-rads_size.width -10,w2.y),1,font_scale2,Scalar(255,255,255),1,CV_AA);


            line(cf_img,Point2f((w1+w2)*0.5),Point2f((w1+w2)*0.5) + Point2f(w_value,0), Scalar(0,w_green,w_red),5.0,2,0);
            line(cf_img,Point2f((v1+v2)*0.5),Point2f((v1+v2)*0.5) + Point2f(v_value,0), Scalar(0,v_green,v_red),5.0,2,0);
            circle(cf_img,Point2f((w1+w2)*0.5),3,Scalar(0,0,255),2);
            circle(cf_img,Point2f((v1+v2)*0.5),3,Scalar(0,0,255),2);

            arrowedLine2(cf_img,center,center + y*50,Scalar(255,0,0),3.0,8,0,0.1);
            arrowedLine2(cf_img,center,center + pb,Scalar(0,255,0),3.0,1,0,0.1);
            arrowedLine2(cf_img,center,center + nfpb,Scalar(0,255,255),3.0,1,0,0.1);

            cf_img.copyTo(total(Rect(2*img.cols,img.rows,img.cols,img.rows)));

        }//*/



    }

}

