//
// Created by Administrator on 2022/1/17.
//

#include "centerface.h"

centerface::centerface() {

}

centerface::~centerface() {
    interpreter_->releaseModel();
    interpreter_->resizeSession(session_);
}

int centerface::Init(const char* model_path) {

    cout << "start init." << endl;
    string model_file = string(model_path) + "centerface.mnn";
    interpreter_ = unique_ptr<Interpreter>(MNN::Interpreter::createFromFile(model_file.c_str()));
    if (nullptr == interpreter_) {
        cout << "load centerface failed." << endl;
        return -1;
    }

    /* create interpreter */
    ScheduleConfig scheduleConfig;
    scheduleConfig.numThread = 4;
    scheduleConfig.type = MNN_FORWARD_CPU;
    BackendConfig backendConfig;
    backendConfig.power = BackendConfig::Power_High;
    backendConfig.precision = BackendConfig::Precision_High;
    scheduleConfig.backendConfig = &backendConfig;

    /* create session */
    session_ = interpreter_->createSession(scheduleConfig);
    input_tensor_ = interpreter_->getSessionInput(session_, nullptr);

    /* image process */

    Matrix trans;

    trans.setScale(1.0f, 1.0f);

    ImageProcess::Config config;
    config.filterType = CV::BICUBIC;
    ::memcpy(config.mean, meanVals_, sizeof(meanVals_));
    ::memcpy(config.normal, normVals_, sizeof(normVals_));
    config.sourceFormat = CV::RGBA;

    pretreat_ = shared_ptr<ImageProcess>(ImageProcess::create(config));
    pretreat_->setMatrix(trans);

    cout << "end init" << endl;
    return 0;
}

int centerface::Detect(const Mat& img_src, vector<FaceInfo>* faces) {

    faces->clear();

    cout << "start detect." << endl;
    if (img_src.empty()) {
        cout << "input empty." << endl;
    }

    int img_w = img_src.cols;
    int img_h = img_src.rows;
    int w_resized = img_w / 32 * 32;
    int h_resized = img_h / 32 * 32;
    float scale_x = static_cast<float>(img_w) / w_resized;
    float scale_y = static_cast<float>(img_h) / h_resized;

    interpreter_->resizeTensor(input_tensor_, { 1, 3, h_resized, w_resized });
    interpreter_->resizeSession(session_);

    Mat img_resized;
    cv::resize(img_src, img_resized, Size(w_resized, h_resized));
    /* why read in RGBA */
    uint8_t* data_ptr = GetImage(img_resized);
    pretreat_->convert(data_ptr, w_resized, h_resized, 0, input_tensor_);

    // run session
    interpreter_->runSession(session_);

    // get output
    MNN::Tensor* tensor_heatmap = interpreter_->getSessionOutput(session_, "537");
    MNN::Tensor* tensor_scale = interpreter_->getSessionOutput(session_, "538");
    MNN::Tensor* tensor_offset = interpreter_->getSessionOutput(session_, "539");
    MNN::Tensor* tensor_landmark = interpreter_->getSessionOutput(session_, "540");

    // copy to host
    auto heatmap_tmp = new Tensor(tensor_heatmap, tensor_heatmap->getDimensionType());
    auto scale_tmp = new Tensor(tensor_scale, tensor_scale->getDimensionType());
    auto offset_tmp = new Tensor(tensor_offset, tensor_offset->getDimensionType());
    auto landmark_tmp = new Tensor(tensor_landmark, tensor_landmark->getDimensionType());
    tensor_heatmap->copyToHostTensor(heatmap_tmp);
    tensor_scale->copyToHostTensor(scale_tmp);
    tensor_offset->copyToHostTensor(offset_tmp);
    tensor_landmark->copyToHostTensor(landmark_tmp);

    int output_w = heatmap_tmp->width();
    int output_h = heatmap_tmp->height();
    int channel_step = output_w * output_h;

    vector<FaceInfo> faces_tmp;
    for (int h = 0; h < output_h; ++h) {
        for (int w = 0; w < output_w; ++w) {
            int index = h * output_w + w;
            float score = heatmap_tmp->host<float>()[index];
            if (score < scoreThreshold_) {
                continue;
            }
            float s0 = 4 * exp(scale_tmp->host<float>()[index]);
            float s1 = 4 * exp(scale_tmp->host<float>()[index + channel_step]);
            float o0 = offset_tmp->host<float>()[index];
            float o1 = offset_tmp->host<float>()[index + channel_step];

            float ymin = MAX(0, 4 * (h + o0 + 0.5) - 0.5 * s0);
            float xmin = MAX(0, 4 * (w + o1 + 0.5) - 0.5 * s1);
            float ymax = MIN(ymin + s0, h_resized);
            float xmax = MIN(xmin + s1, w_resized);

            FaceInfo face_info;
            face_info.score_ = score;
            face_info.face_.x = scale_x * xmin;
            face_info.face_.y = scale_y * ymin;
            face_info.face_.width = scale_x * (xmax - xmin);
            face_info.face_.height = scale_y * (ymax - ymin);

            for (int num = 0; num < 5; ++num) {
                face_info.keypoints_[2 * num] = scale_x * (s1 * landmark_tmp->host<float>()[(2 * num + 1) * channel_step + index] + xmin);
                face_info.keypoints_[2 * num + 1] = scale_y * (s0 * landmark_tmp->host<float>()[(2 * num + 0) * channel_step + index] + ymin);
            }
            faces_tmp.push_back(face_info);
        }
    }
    sort(faces_tmp.begin(), faces_tmp.end(), [](const FaceInfo& a, const FaceInfo& b) { return a.score_ > b.score_; });

    NMS(faces_tmp, faces, nmsThreashold_);

    return 0;
}

uint8_t* centerface::GetImage(const Mat& img_src) {

    uchar* data_ptr = new uchar[img_src.total() * 4];
    Mat img_tmp(img_src.size(), CV_8UC4, data_ptr);
    cv::cvtColor(img_src, img_tmp, cv::COLOR_BGR2RGBA, 4);
    return (uint8_t*)img_tmp.data;
}

float centerface::InterRectArea(const cv::Rect& a, const cv::Rect& b) {

    cv::Point left_top = cv::Point(MAX(a.x, b.x), MAX(a.y, b.y));
    cv::Point right_bottom = cv::Point(MIN(a.br().x, b.br().x), MIN(a.br().y, b.br().y));
    cv::Point diff = right_bottom - left_top;

    return MAX((diff.x + 1), 0) * MAX((diff.y + 1), 0);
}

int centerface::ComputeIOU(const cv::Rect& rect1, const cv::Rect& rect2, float* iou) {
    *iou = InterRectArea(rect1, rect2) / (rect1.area() + rect2.area() - InterRectArea(rect1, rect2));
    return 0;
}

int centerface::NMS(const vector<FaceInfo>& faces, vector<FaceInfo>* result, const float& threshold) {

    result->clear();

    vector<int> idx(faces.size());

    for (int i = 0; i < idx.size(); ++i) {
        idx[i] = i;
    }

    while (!idx.empty()) {
        int good_index = idx[0];
        result->push_back(faces[good_index]);
        vector<int> tmp = idx;
        idx.clear();
        for (int i = 1; i < tmp.size(); ++i) {
            int tmp_i = tmp[i];
            float iou = 0.0f;
            ComputeIOU(faces[good_index].face_, faces[tmp_i].face_, &iou);
            if (iou < threshold) {
                idx.push_back(tmp_i);
            }
        }
    }
    return 0;
}
