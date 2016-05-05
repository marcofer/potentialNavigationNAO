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

#include <iostream>



class ParallelDominantPlaneBuild : public cv::ParallelLoopBody{

    private:
    	int coreNum;
        cv::Mat& dp;
        cv::Mat& inv_dp;
        cv::Mat& op;
        cv::Mat of;
        cv::Mat& pf;
        
        double epsilon;
        double Tc;
        double cut_f;
            
    	cv::Matx22f A;
        cv::Matx21f b;

        int dp_threshold;

        cv::Mat& nf_dp;
        cv::Mat nf_of;
        cv::Mat& nf_pf;

        cv::Matx22f nf_A;
        cv::Matx21f nf_b;

    public:
        ParallelDominantPlaneBuild(int ncores, cv::Mat& dpImage, cv::Mat& invDpImage, cv::Mat& opImage, cv::Mat ofImage, cv::Mat& pfImage, double eps,
                                   double sampling_time, double f, cv::Matx22f A_, cv::Matx21f b_, int ths, cv::Mat& nf_dp_,
                                   cv::Mat nf_of_, cv::Mat& nf_pf_, cv::Matx22f nf_A_, cv::Matx21f nf_b_)
                    : coreNum(ncores), dp(dpImage), inv_dp(invDpImage), op(opImage), of(ofImage), pf(pfImage), epsilon(eps), Tc(sampling_time), cut_f(f),
                      A(A_), b(b_), dp_threshold(ths), nf_dp(nf_dp_), nf_of(nf_of_), nf_pf(nf_pf_), nf_A(nf_A_), nf_b(nf_b_){}

        virtual void operator()(const cv::Range& range) const;
};



class ParallelGradientFieldBuild : public cv::ParallelLoopBody{

	private:
		int coreNum;
		cv::Mat& dp;
		cv::Mat& sp;
		cv::Mat& gf;
        cv::Mat& vf;
        double scale;
        cv::Size size;
        double sigmaX;
		int depth;

        cv::Mat& nf_dp;
        cv::Mat& nf_sp;
        cv::Mat& nf_gf;
        cv::Mat& nf_vf;

	public:
        ParallelGradientFieldBuild(int ncores, cv::Mat& dpImage, cv::Mat& spImage, cv::Mat& gfImage, cv::Mat& vfImage,
                                   double s, cv::Size size_, double sigma_x, int depth_, cv::Mat& nf_dp_, cv::Mat& nf_sp_,
                                   cv::Mat& nf_gf_,cv::Mat& nf_vf_)
                    : coreNum(ncores), dp(dpImage), sp(spImage), gf(gfImage), vf(vfImage), scale(s), size(size_),
                      sigmaX(sigma_x), depth(depth_), nf_dp(nf_dp_),nf_sp(nf_sp_),nf_gf(nf_gf_), nf_vf(nf_vf_){}

		virtual void operator()(const cv::Range& range) const;

};


class ParallelOpticalFlow : public cv::ParallelLoopBody {

	private:
		int coreNum;
        cv::Mat img, img2;
        cv::Mat& optical_flow;
        double pyr_scale;
		double winSize;
		double maxLayer;
		double of_iterations;
		double poly_n;
		double poly_sigma;
		double flags;
		/*cv::gpu::GpuMat img, img2;
		
		cv::gpu::FarnebackOpticalFlow& farneback_flow;
		const cv::gpu::GpuMat u_flow, v_flow;//*/

	public:
		/*ParallelOpticalFlow(int cores, cv::gpu::FarnebackOpticalFlow& flowHandler, cv::gpu::GpuMat img_, cv::gpu::GpuMat img2_, const cv::gpu::GpuMat u, const cv::gpu::GpuMat v, cv::Mat& of)
					: coreNum(cores), farneback_flow(flowHandler), img(img_), img2(img2_), u_flow(u), v_flow(v), optical_flow(of){}//*/
		ParallelOpticalFlow(int cores, cv::Mat img_, cv::Mat img2_, cv::Mat& of, double pyr, double winsize, double levels, double iters, double poly_deg, double sigma, double flags_)
					: coreNum(cores), img(img_), img2(img2_), optical_flow(of), pyr_scale(pyr), winSize(winsize), maxLayer(levels), of_iterations(iters), poly_n(poly_deg), poly_sigma(sigma), flags(flags_){}//*/

		virtual void operator()(const cv::Range& range) const;

};



class ParallelDisplayImages : public cv::ParallelLoopBody {

	private:
		int coreNum;
		int flowResolution;
		cv::Mat img;
		cv::Mat of;
		cv::Mat pf;
		cv::Mat dp;
		cv::Mat sp;
        cv::Mat gf;
		cv::Matx21f p_bar;
        cv::Matx21f nf_pbar;
		double angular_vel;
		cv::Mat& total;
		cv::Rect dpROI;
        double theta;
        double v;
        double w;

	public:
        ParallelDisplayImages(int cores, int flow_res, cv::Mat img_,cv::Mat of_,cv::Mat pf_,cv::Mat dp_,cv::Mat sp_, cv::Mat gf_,cv::Matx21f p_bar_,double w,cv::Mat& total_, cv::Rect roi, double th,
                              double lin, double ang, cv::Matx21f nf_pbar_)
            : coreNum(cores), flowResolution(flow_res), img(img_), of(of_), pf(pf_), dp(dp_), sp(sp_), gf(gf_), p_bar(p_bar_), angular_vel(w), total(total_), dpROI(roi), theta(th), v(lin), w(ang), nf_pbar(nf_pbar_){}
		virtual void operator()(const cv::Range& range) const;


};
