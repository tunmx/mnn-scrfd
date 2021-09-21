#include <iostream>
#include "SCRFD.h"
#include "model_load.h"

using namespace cv;
using namespace std;


int main() {
    // use scrfd_2.5g_bnkps_shape320x320
    string path = "../models/scrfd_2.5g_bnkps_shape320x320.mnn";
    SCRFD scrfd;
    scrfd.load_heads(scrfd_2_5g_bnkps_head_info);
    scrfd.reload(path, true, 320, 2, 4);
    Mat img = imread("../images/human.jpeg");
    vector<FaceInfo> results;
    double time;
    time = (double) cv::getTickCount();
    scrfd.detect(img, results);
    time = ((double) cv::getTickCount() - time) / cv::getTickFrequency();
    cout << "use time：" << time << "秒\n";
    SCRFD::draw(img, true, results);
    cv::imshow("result", img);
    cv::waitKey(0);

    return 0;
}
