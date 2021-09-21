//
// Created by tunm on 2021/9/19.
//

#ifndef MNN_SCRFD_SCRFD_H
#define MNN_SCRFD_SCRFD_H

#include "opencv2/opencv.hpp"
#include "MNN/Interpreter.hpp"
#include "MNN/Tensor.hpp"
#include "MNN/ImageProcess.hpp"
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include "model_load.h"

typedef struct FaceInfo_ {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    float lmk[10];
} FaceInfo;

class SCRFD {
public:
    SCRFD();

    void reload(std::string &path, bool use_kps, int input_seize_level = 320, int num_anchors = 2, int thread_num = 1);

    void detect(cv::Mat &bgr, std::vector<FaceInfo> &results);

    static void generate_anchors(int stride, int input_size, int num_anchors, std::vector<float> &anchors);

    void decode(MNN::Tensor *cls_pred, MNN::Tensor *box_pred, MNN::Tensor *lmk_pred, int stride,
                std::vector<FaceInfo> &results);

    void nms(std::vector<FaceInfo> &input_faces, float nms_threshold);

    ~SCRFD();

    void load_heads(const std::vector<DetHeadInfo>& heads_info);

    static void draw(cv::Mat& img, bool use_kps, const std::vector<FaceInfo>& results);

private:
    std::shared_ptr<MNN::Interpreter> interpreter_;
    MNN::Tensor *input_ = nullptr;
    MNN::Session *session_ = nullptr;
    int input_size_;
    int num_anchors_;
    float prob_threshold_ = 0.5f;
    float nms_threshold_ = 0.4f;
    bool use_kps_ = false;

    float mean_[3] = {127.5f, 127.5f, 127.5f};
    float normal_[3] = {0.0078125, 0.0078125, 0.0078125};

    std::string input_name_ = "input.1";
    std::vector<DetHeadInfo> heads_info_;
};


#endif //MNN_SCRFD_SCRFD_H
