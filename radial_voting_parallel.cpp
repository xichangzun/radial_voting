#include <opencv2/opencv.hpp>
#include <cmath>
#include <vector>
#include <tbb/tbb.h>
using namespace cv;
using namespace std;
typedef float(*w_f)(int,int,int,int,float);
const float PI = 3.1415927;
float _delta = 30.0*PI/180.0;
float default_weight(int y,int x,int center_y,int center_x,float sigma){
    return 1;
}
float gauss_weight(int y,int x,int center_y,int center_x,float sigma){
    float s = 2*sigma*sigma;
    float index = -1*(pow(x-center_x,2.0)+pow(y-center_y,2.0))/s;
    float weight = exp(index)/(s*PI);
    return weight;
}
vector<Mat> get_area_templates(int rmin,int rmax,w_f weight=default_weight,float d=_delta){
    vector<Mat> area_templates;
    int num = 360;
    if (rmax < 15)
        num = 90;
    else if (rmax < 30)
        num = 180;

    float radio = 2*PI/num;
    int x = rmax;
    int y = rmax;
    int low_bound = rmin*rmin;
    int up_bound = rmax*rmax;
    for(int i = 0;i< num;i++){
        Mat init_areas = Mat::zeros(rmax*2+1,rmax*2+1,CV_64F);
        float orient = i*radio;
        int center_x = floor(x+((rmin+rmax)/2)*cos(orient));
        int center_y = floor(y+((rmin+rmax)/2)*sin(orient));
        for(int p = 0;p < 2*rmax+1;p++){
            for(int q = 0;q<2*rmax+1;q++){
                int dist = (p-x)*(p-x)+(q-y)*(q-y);
                if (dist > low_bound && dist < up_bound){
                    float vec_angle = atan2(q-y,p-x);
                    if(vec_angle<0) vec_angle += PI*2;
                    float angle_diff = abs(vec_angle - orient);
                    if(angle_diff <= d || angle_diff >= PI*2-d){
                        init_areas.at<double>(q,p) = 1*weight(q,p,center_y,center_x,(rmax-rmin)/2);
                    }
                }
            }
        }
        area_templates.push_back(init_areas);
    }
    return area_templates;
}

Mat v_a_from_templates(int y,int x,vector<Mat> area_templates,Mat angle){
    float orient = angle.at<double>(y,x)+ PI;
    if(orient > 2*PI) orient -= 2*PI;
    float radio = 2*PI/area_templates.size();
    int index = floor((orient/radio)+0.5);
    index %= area_templates.size();
    Mat voting_area = Mat::zeros(angle.rows,angle.cols,CV_64F);
    Mat sample = area_templates[index].clone();
    int rmax = sample.cols/2;
    int move_y = rmax - y;
    int move_x = rmax - x;

    int x_low = max(x-rmax,0);
    int x_high = min(x+rmax,angle.cols);
    int y_low = max(y-rmax,0);
    int y_high = min(y+rmax,angle.rows);

    int s_x_l = x_low + move_x;
    int s_x_h = x_high + move_x;
    int s_y_l = y_low + move_y;
    int s_y_h = y_high + move_y;

    sample(Range(s_y_l,s_y_h),Range(s_x_l,s_x_h)).copyTo(voting_area(Range(y_low,y_high),Range(x_low,x_high)));
    return voting_area;
}

Mat find_local_max(Mat area,int distance,double thresh_rel){
    Mat maxim,mask;
    Mat local_area = getStructuringElement(MORPH_RECT,Size( 2*distance + 1, 2*distance + 1 ));
    dilate(area, maxim, local_area,Point(-1,-1),1,BORDER_REFLECT);
    compare(area, maxim, mask, CMP_GE);
    double th_max,th_min;
    minMaxLoc(area,&th_min,&th_max);
    double thresh = thresh_rel*th_max;
    for(int x = 0;x < area.cols;x++){
        for(int y = 0;y< area.rows;y++){
            if( mask.at<uchar>(y,x) == 255 && area.at<double>(y,x) <= thresh ){
                mask.at<uchar>(y,x) = 0;
            }
        }
    }
    return mask;
}

class VA_GEN{
public:
    
    vector<Mat>& templates;
    Mat& my_angle;
    Mat* result;
    void operator() (tbb::blocked_range<int>& r) const {
        const Mat& angle = my_angle;
        int tsize = templates.size();
        float radio = 2*PI/tsize;
        int rmax = templates[0].cols/2;
        for(int i= r.begin();i!= r.end();i++){
            Mat& tmp = result[i];

            for(int j =0 ;j<angle.cols;j++){

                float orient = angle.at<double>(i,j)+ PI;
                if(orient > 2*PI) orient -= 2*PI;
                int index = floor((orient/radio)+0.5);
                index %= tsize;
                int move_y = rmax - i;
                int move_x = rmax - j;

                int x_low = max(j-rmax,0);
                int x_high = min(j+rmax,angle.cols);
                int y_low = max(i-rmax,0);
                int y_high = min(i+rmax,angle.rows);

                int s_x_l = x_low + move_x;
                int s_x_h = x_high + move_x;
                int s_y_l = y_low + move_y;
                int s_y_h = y_high + move_y;
                // cout <<y_low<<" "<< y_high <<" "<<x_low<<" "<<x_high<<endl;
                // cout <<s_y_l<<" "<< s_y_h << " "<< s_x_l << " " << s_x_h<<endl;
                tmp(Range(y_low,y_high),Range(x_low,x_high)) += templates[index](Range(s_y_l,s_y_h),Range(s_x_l,s_x_h));

            }
        }
    }

    VA_GEN(vector<Mat>& area_templates,Mat& angle,Mat * arr ):templates(area_templates),my_angle(angle),result(arr){
    }
};
void parallel_v_a(vector<Mat> area_templates,Mat angle,Mat * imgarr){
    VA_GEN a(area_templates,angle,imgarr);
    tbb::parallel_for(tbb::blocked_range<int>(0,angle.rows), a);
}

int main(int argc,char** argv){
    // read img and trans to gray
    Mat img = imread("notebooks/speci2.png"),gray;
    cvtColor(img,gray,COLOR_BGR2GRAY);

    // caculate grad
    Mat sobelx,sobely,magni,angle;
    Sobel(gray,sobelx,CV_64F,1,0,5);
    Sobel(gray,sobely,CV_64F,0,1,5);
    cartToPolar(sobelx,sobely,magni,angle);

    // get distance map
    // get foreground by threshold;
    Mat ret;
    threshold(gray,ret,0,255,THRESH_BINARY_INV| THRESH_OTSU);

    // fill hole
    Mat im_filled = Mat::zeros(ret.rows+2,ret.cols+2,ret.type());
    ret.copyTo(im_filled(Range(1,ret.rows+1),Range(1,ret.cols+1)));
    floodFill(im_filled,Point(0,0),Scalar(255));
    Mat im_new;
    im_filled(Range(1,ret.rows+1),Range(1,ret.cols+1)).copyTo(im_new);
    Mat new_ret = ret|(~im_new);

    //erode to detach 
    Mat opening;
    morphologyEx(new_ret,opening,MORPH_ERODE,getStructuringElement(MORPH_RECT,Size(3,3)));
    // dis trans
    Mat distance_map;
    distanceTransform(opening,distance_map,DIST_L2,DIST_MASK_5);

    // get voting area
    tbb::tick_count t0 = tbb::tick_count::now();
    double ma_mean = mean(magni)[0];
    vector<Mat> area_templates = get_area_templates(5,15,gauss_weight);

    tbb::tick_count t1 = tbb::tick_count::now();
    Mat results[angle.rows];
    for(int i = 0;i<angle.rows;i++){
        results[i] = Mat::zeros(angle.rows,angle.cols,CV_64F);
    }
    parallel_v_a(area_templates,angle,results);

    tbb::tick_count t2 = tbb::tick_count::now();
    Mat voting_area = Mat::zeros(img.rows,img.cols,CV_64F);
    for(Mat i: results){
        voting_area += i;
    }
    tbb::tick_count t3 = tbb::tick_count::now();

    cout << "templates consume: " << (t1-t0).seconds() <<endl;
    cout << "for consume: " << (t2-t1).seconds() <<endl;
    cout << "reduce consume: " << (t3-t2).seconds() <<endl;

    Mat dst;
    distance_map.convertTo(dst, CV_64F);
    Mat refined_voting_area = voting_area.mul(dst);
    
    // fing local maxima
    Mat mask = find_local_max(refined_voting_area,5,0.04);
    imwrite("mask.png",mask);
    for(int x = 0;x< img.cols;x++){
        for(int y = 0; y< img.rows;y++){
            if(mask.at<uchar>(y,x) == 255){
                img.at<Vec3b>(y,x)[0] = 0;
                img.at<Vec3b>(y,x)[1] = 0;
                img.at<Vec3b>(y,x)[2] = 0;
            }
                
        }
    }
    imwrite("result.png",img);
    return 0;
}