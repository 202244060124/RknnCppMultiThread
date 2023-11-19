#ifndef RKNN_POOL_H
#define RKNN_POOL_H

#include <iostream>
#include <queue>
#include <vector>
#include "RgaUtils.h"
#include "im2d.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "post_process.h"
#include "rga.h"
#include "rknn_api.h"
#include "thread_pool.h"
using cv::Mat;
using std::queue;
using std::vector;

class RknnProcess {
private:
    rknn_context m_rknnCtx;
    unsigned char* m_modelData;
    rknn_sdk_version m_rknnSdkVersion;
    rknn_input_output_num m_inputOutputNum;
    rknn_tensor_attr* m_inputAttrs;
    rknn_tensor_attr* m_outputAttrs;
    rknn_input m_inputs[1];
    void* m_resizeBuffer;
    int m_ret;
    int m_modelInputChannel = 3;
    int m_modelInputWidth = 0;
    int m_modelInputHeight = 0;

public:
    Mat m_srcImage;
    int Inference(void);
    int SetInputAttrs(void);
    void SetOutputAttrs(void);
    void SetInputPara(void);
    int ImageRgaResize(Mat& rgb_img, int img_width, int img_height);
    RknnProcess(char* model_path, int npu_id);
    ~RknnProcess();
};

#endif