#include "utils.h"

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

vector<Point> contoursConvexHull( vector<vector<Point> > contours )
{
    vector<Point> result;
    vector<Point> pts;
    for ( size_t i = 0; i< contours.size(); i++)
        for ( size_t j = 0; j< contours[i].size(); j++)
            pts.push_back(contours[i][j]);
    convexHull( pts, result );
    return result;
}

