#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <map>

int
main (int argc, char *argv[])
{
    
    
    cv::Mat src_img = cv::imread("/Users/shogonakata/dev/RedLightDetection/images/lane3.jpg", 1);
    if(src_img.empty()) return -1;
    
    // 1.HSV変換
    cv::Mat hsv_img;
    cv::cvtColor(src_img,hsv_img,CV_BGR2HSV);
    
    // 2.グレー画像に変換
    cv::Mat gray_img;
    cv::cvtColor(src_img, gray_img, CV_BGR2GRAY);
    
    // 3.ラインの色が白か黄色かでフィルターした画像を生成
    cv::Mat white_hue,yellow_hue;
    threshold(gray_img,white_hue,180,255,cv::THRESH_BINARY);
    inRange(hsv_img, cv::Scalar(20, 100, 100), cv::Scalar(30, 255, 255), yellow_hue);
    
    // 4.白と黄色の論理和のフィルターを作り、元画像のグレーから論理積を取る
    cv::Mat yellow_white_hue,yellow_white_hue_img;
    cv::bitwise_or(white_hue, yellow_hue,yellow_white_hue);
    cv::bitwise_and(gray_img, yellow_white_hue, yellow_white_hue_img);
    
    // 5.ガウシアンブラー
    cv::Mat gaussian_img;
    cv::GaussianBlur(yellow_white_hue_img, gaussian_img, cv::Size(13,13), 2, 2);
    
    // 6.Cannyエッジ検出
    cv::Mat canny_img;
    double low_threshold = 50;
    double high_threshold = 150;
    cv::Canny(gaussian_img, canny_img, low_threshold, high_threshold);
    
    // 7.自車線のある台形空間のみを切り出す
    cv::Mat interest_img;
    cv::Mat mask(canny_img.size(), canny_img.type(), cv::Scalar::all(0));
    cv::Point pt[4];
    pt[0] = cv::Point(mask.cols/9,mask.rows);
    pt[1] = cv::Point(mask.cols/2-mask.cols/8, mask.rows/2);
    pt[2] = cv::Point(mask.cols/2+mask.cols/8, mask.rows/2);
    pt[3] = cv::Point(mask.cols/9*8, mask.rows);
    cv::fillConvexPoly(mask, pt, 4, cv::Scalar(255,255,255) );
    cv::bitwise_and(canny_img, mask ,interest_img);
    
    // 8.Hough変換をして直線を検出する
    cv::Mat hough_img;
    hough_img = src_img.clone();
    std::vector<cv::Vec2f> lines;
    cv::HoughLines(interest_img, lines, 1, CV_PI/180, 50, 0, 0);
    
    // 9.直線の中から最もセンターに近い直線をレーンと認識する.
    cv::Point most_near_lane_right_pt1;
    cv::Point most_near_lane_right_pt2;
    cv::Point most_near_lane_left_pt1;
    cv::Point most_near_lane_left_pt2;
    std::vector<cv::Vec2f>::iterator it = lines.begin();
    for(; it!=lines.end(); ++it) {
        
        // -公式-
        // x = cosTheta*rho + t*(sinTheta)
        // y = sinTheta*rho + t*(-cosTheta)
        
        // 直線に原点から下ろした垂線の長さ
        float rho = (*it)[0];
        // 垂線とx軸とのなす角度
        float theta = (*it)[1];
        // 原点からシータ角度で伸ばしていき直線と交わるx座標
        double x0 = cos(theta)*rho;
        // 原点からシータ角度で伸ばしていき直線と交わるy座標
        double y0 = sin(theta)*rho;
        cv::Point pt1;
        
        // pt1はy軸が画面下限になっている座標
        // y1=hough_img.rowsの時のtを探してpt1を求める.
        double t1 = (hough_img.rows -y0 )/(-cos(theta));
        pt1.x = cv::saturate_cast<int>(x0 + t1*(sin(theta)));
        pt1.y = cv::saturate_cast<int>(hough_img.rows);
        
        // 横線はレーンではないので削除する
        if(pt1.x > hough_img.cols || pt1.x <0){
            continue;
        }
        
        // 左側のレーン
        if(pt1.x <= hough_img.cols/2){
            // 初回
            if(most_near_lane_left_pt1.x == 0){
                most_near_lane_left_pt1 = pt1;
                continue;
            }
            // 最も中央に近いレーンを選ぶ
            if(hough_img.cols/2 - most_near_lane_left_pt1.x >= hough_img.cols/2 - pt1.x){
                most_near_lane_left_pt1 = pt1;
                // pt2はy軸が画面中央になっている座標
                // y1=hough_img.rows/2の時のtを探してpt2を求める.
                // -sinTheta*rho /(-cosTheta)=t
                double t2 = ((hough_img.rows/3*2) -y0 )/(-cos(theta));
                most_near_lane_left_pt2.x = cv::saturate_cast<int>(x0 + t2*(sin(theta)));
                most_near_lane_left_pt2.y = cv::saturate_cast<int>(hough_img.rows/3*2);
            }
        }
        
        // 右側のレーン
        if(pt1.x > hough_img.cols/2){
            // 初回
            if(most_near_lane_right_pt1.x == 0){
                most_near_lane_right_pt1 = pt1;
                continue;
            }
            // 最も中央に近いレーンを選ぶ
            if(most_near_lane_right_pt1.x - hough_img.cols/2 >= pt1.x - hough_img.cols/2){
                most_near_lane_right_pt1 = pt1;
                // pt2はy軸が画面中央になっている座標
                // y1=hough_img.rows/2の時のtを探してpt2を求める.
                // -sinTheta*rho /(-cosTheta)=t
                double t2 = ((hough_img.rows/3*2) -y0 )/(-cos(theta));
                most_near_lane_right_pt2.x = cv::saturate_cast<int>(x0 + t2*(sin(theta)));
                most_near_lane_right_pt2.y = cv::saturate_cast<int>(hough_img.rows/3*2);
            }
        }
    }
    
    cv::line(hough_img, most_near_lane_left_pt1, most_near_lane_left_pt2, cv::Scalar(0,0,255), 3, CV_AA);
    cv::line(hough_img, most_near_lane_right_pt1, most_near_lane_right_pt2, cv::Scalar(0,0,255), 3, CV_AA);
    
    cv::namedWindow("Detection", CV_WINDOW_AUTOSIZE|CV_WINDOW_FREERATIO);
    cv::imshow("Detection", hough_img);
    cv::waitKey(0);
}


