//
// Created by Administrator on 2022/1/17.
//

#ifndef FACE_DETECTOR_CENTERFACE_H
#define FACE_DETECTOR_CENTERFACE_H

#include <vector>

#include <MNN/Interpreter.hpp>
#include <MNN/ImageProcess.hpp>
#include <opencv2/opencv.hpp>

using namespace MNN;
using namespace MNN::CV;
using namespace cv;

using namespace std;

struct FaceInfo {
    cv::Rect face_;
    float score_;
    float keypoints_[10];
};

class centerface {
public:
    centerface();
    ~centerface();
    int Init(const char* model_path);
    int Detect(const Mat& img_src, vector<FaceInfo>* faces);

private:
    uint8_t* GetImage(const Mat& img_src);
    float InterRectArea(const cv::Rect& a, const cv::Rect& b);
    int ComputeIOU(const cv::Rect& rect1, const cv::Rect& rect2, float* iou);
    int NMS(const vector<FaceInfo>& faces, vector<FaceInfo>* result, const float& threshold);

private:
    bool initialized_;
    shared_ptr<ImageProcess> pretreat_;
    shared_ptr<Interpreter> interpreter_;
    Session* session_ = nullptr;
    Tensor* input_tensor_ = nullptr;

    const float meanVals_[3] = { 0.0f, 0.0f, 0.0f };
    const float normVals_[3] = { 1.0f, 1.0f, 1.0f };
    const float scoreThreshold_ = 0.5f;
    const float nmsThreashold_ = 0.5f;

};


#endif //FACE_DETECTOR_CENTERFACE_H
