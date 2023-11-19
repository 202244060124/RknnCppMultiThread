#include "rknn_process.h"

static unsigned char* LoadFileData(FILE* fp, size_t offset, size_t size)
{
    unsigned char* data;
    int ret;

    data = NULL;

    if (NULL == fp) {
        return NULL;
    }

    ret = fseek(fp, offset, SEEK_SET);
    if (ret != 0) {
        printf("blob seek failure.\n");
        return NULL;
    }

    data = (unsigned char*)malloc(size);
    if (data == NULL) {
        printf("buffer malloc failure.\n");
        return NULL;
    }
    ret = fread(data, 1, size, fp);
    return data;
}

static unsigned char* LoadModel(const char* model_path, int* model_data_size)
{
    FILE* fp;
    unsigned char* data;

    fp = fopen(model_path, "rb");
    if (NULL == fp) {
        printf("Open file %s failed.\n", model_path);
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);

    data = LoadFileData(fp, 0, size);

    fclose(fp);

    *model_data_size = size;
    return data;
}


int RknnProcess::SetInputAttrs(void)
{
    m_inputAttrs = new rknn_tensor_attr[m_inputOutputNum.n_input];
    memset(m_inputAttrs, 0, sizeof(m_inputAttrs));
    for (int i = 0; i < m_inputOutputNum.n_input; i++) {
        m_inputAttrs[i].index = i;
        m_ret = rknn_query(m_rknnCtx, RKNN_QUERY_INPUT_ATTR, &(m_inputAttrs[i]), sizeof(rknn_tensor_attr));
        if (m_ret < 0) {
            printf("rknn_init error m_ret=%d\n", m_ret);
            return -1;
        }
    }
    return 0;
}
void RknnProcess::SetOutputAttrs(void)
{
    // 设置输出数组
    m_outputAttrs = new rknn_tensor_attr[m_inputOutputNum.n_output];
    memset(m_outputAttrs, 0, sizeof(m_outputAttrs));
    for (int i = 0; i < m_inputOutputNum.n_output; i++) {
        m_outputAttrs[i].index = i;
        m_ret = rknn_query(m_rknnCtx, RKNN_QUERY_OUTPUT_ATTR, &(m_outputAttrs[i]), sizeof(rknn_tensor_attr));
    }
}
void RknnProcess::SetInputPara(void)
{
    // 设置输入参数
    if (m_inputAttrs[0].fmt == RKNN_TENSOR_NCHW) {
        printf("model is NCHW input fmt\n");
        m_modelInputChannel = m_inputAttrs[0].dims[1];
        m_modelInputHeight = m_inputAttrs[0].dims[2];
        m_modelInputWidth = m_inputAttrs[0].dims[3];
    } else {
        printf("model is NHWC input fmt\n");
        m_modelInputHeight = m_inputAttrs[0].dims[1];
        m_modelInputWidth = m_inputAttrs[0].dims[2];
        m_modelInputChannel = m_inputAttrs[0].dims[3];
    }

    memset(m_inputs, 0, sizeof(m_inputs));
    m_inputs[0].index = 0;
    m_inputs[0].type = RKNN_TENSOR_UINT8;
    m_inputs[0].size = m_modelInputWidth * m_modelInputHeight * m_modelInputChannel;
    m_inputs[0].fmt = RKNN_TENSOR_NHWC;
    m_inputs[0].pass_through = 0;
}

RknnProcess::RknnProcess(char* model_path, int npu_id)
{
    /* Create the neural network */
    printf("Loading mode...\n");
    int model_data_size = 0;
    // 读取模型文件数据
    m_modelData = LoadModel(model_path, &model_data_size);
    // 通过模型文件初始化rknn类
    m_ret = rknn_init(&m_rknnCtx, m_modelData, model_data_size, 0, NULL);
    if (m_ret < 0) {
        printf("rknn_init error m_ret=%d\n", m_ret);
        exit(-1);
    }
    //
    rknn_core_mask core_mask;
    if (npu_id == 0) {
        core_mask = RKNN_NPU_CORE_0;
    } else if (npu_id == 1) {
        core_mask = RKNN_NPU_CORE_1;
    } else {
        core_mask = RKNN_NPU_CORE_2;
    }
    int m_ret = rknn_set_core_mask(m_rknnCtx, core_mask);
    if (m_ret < 0) {
        printf("rknn_init core error m_ret=%d\n", m_ret);
        exit(-1);
    }

    // 初始化rknn类的版本
    m_ret = rknn_query(m_rknnCtx, RKNN_QUERY_SDK_VERSION, &m_rknnSdkVersion, sizeof(rknn_sdk_version));
    if (m_ret < 0) {
        printf("rknn_init error m_ret=%d\n", m_ret);
        exit(-1);
    }

    // 获取模型的输入参数
    m_ret = rknn_query(m_rknnCtx, RKNN_QUERY_IN_OUT_NUM, &m_inputOutputNum, sizeof(m_inputOutputNum));
    if (m_ret < 0) {
        printf("rknn_init error m_ret=%d\n", m_ret);
        exit(-1);
    }

    // 设置输入数组
    m_ret = SetInputAttrs();
    if (m_ret < 0) {
        printf("rknn_init set input attrs fail m_ret=%d\n", m_ret);
        exit(-1);
    }
    SetOutputAttrs();

    // 设置输入参数
    SetInputPara();

    // 申请缩放图片空间
    m_resizeBuffer = (void*)malloc(m_modelInputHeight * m_modelInputWidth * m_modelInputChannel);
    if (m_resizeBuffer == nullptr) {
        printf("rknn_init malloc resize buffer fail \n");
        exit(-1);
    }
}

RknnProcess::~RknnProcess()
{
    m_ret = rknn_destroy(m_rknnCtx);
    delete[] m_inputAttrs;
    delete[] m_outputAttrs;
    if (m_modelData) {
        free(m_modelData);
    }
    if (m_resizeBuffer) {
        free(m_resizeBuffer);
    }
}

int RknnProcess::ImageRgaResize(Mat& rgb_img, int img_width, int img_height)
{
    // init rga context
    // rga是rk自家的绘图库,绘图效率高于OpenCV
    rga_buffer_t srcRgaBuffer;
    rga_buffer_t dstRgaBuffer;
    memset(&srcRgaBuffer, 0, sizeof(srcRgaBuffer));
    memset(&dstRgaBuffer, 0, sizeof(dstRgaBuffer));
    im_rect srcRect;
    im_rect dstRect;
    memset(&srcRect, 0, sizeof(srcRect));
    memset(&dstRect, 0, sizeof(dstRect));
    memset(m_resizeBuffer, 0x00, m_modelInputHeight * m_modelInputWidth * m_modelInputChannel);

    srcRgaBuffer = wrapbuffer_virtualaddr((void*)rgb_img.data, img_width, img_height, RK_FORMAT_RGB_888);
    dstRgaBuffer = wrapbuffer_virtualaddr((void*)m_resizeBuffer, m_modelInputWidth, m_modelInputHeight, RK_FORMAT_RGB_888);
    int ret = imcheck(srcRgaBuffer, dstRgaBuffer, srcRect, dstRect);
    if (ret != IM_STATUS_NOERROR) {
        printf("%d, check error! %s", __LINE__, imStrError((IM_STATUS)ret));
        return -1;
    }
    IM_STATUS STATUS = imresize(srcRgaBuffer, dstRgaBuffer);

    cv::Mat resize_img(cv::Size(m_modelInputWidth, m_modelInputHeight), CV_8UC3, m_resizeBuffer);
    return 0;
}

int RknnProcess::Inference()
{
    cv::Mat rgbImg;
    // 获取图像宽高
    int img_width = m_srcImage.cols;
    int img_height = m_srcImage.rows;
    cv::cvtColor(m_srcImage, rgbImg, cv::COLOR_BGR2RGB);

    // You may not need resize when srcRgaBuffer resulotion equals to dstRgaBuffer resulotion
    // 如果输入图像不是指定格式
    if (img_width != m_modelInputWidth || img_height != m_modelInputHeight) {
        m_ret = ImageRgaResize(rgbImg, img_width, img_height);
        if (m_ret < 0) {
            printf("rga resize image buffer fail !\n");
            return -1;
        }
        m_inputs[0].buf = m_resizeBuffer;
    } else {
        m_inputs[0].buf = (void*)rgbImg.data;
    }

    // 设置rknn的输入数据
    rknn_inputs_set(m_rknnCtx, m_inputOutputNum.n_input, m_inputs);

    // 设置输出
    rknn_output outputs[m_inputOutputNum.n_output];
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < m_inputOutputNum.n_output; i++) {
        outputs[i].want_float = 0;
    }

    // 调用npu进行推演
    m_ret = rknn_run(m_rknnCtx, NULL);
    // 获取npu的推演输出结果
    m_ret = rknn_outputs_get(m_rknnCtx, m_inputOutputNum.n_output, outputs, NULL);

    // 总之就是绘图部分
    // post process
    // width是模型需要的输入宽度, img_width是图片的实际宽度
    float scale_w = (float)m_modelInputWidth / img_width;
    float scale_h = (float)m_modelInputHeight / img_height;

    detect_result_group_t detect_result_group;
    std::vector<float> out_scales;
    std::vector<int32_t> out_zps;
    for (int i = 0; i < m_inputOutputNum.n_output; ++i) {
        out_scales.push_back(m_outputAttrs[i].scale);
        out_zps.push_back(m_outputAttrs[i].zp);
    }
    post_process((int8_t*)outputs[0].buf, (int8_t*)outputs[1].buf, (int8_t*)outputs[2].buf, m_modelInputHeight, m_modelInputWidth,
                 BOX_THRESH, NMS_THRESH, scale_w, scale_h, out_zps, out_scales, &detect_result_group);

    // Draw Objects
    char text[256];
    for (int i = 0; i < detect_result_group.count; i++) {
        detect_result_t* det_result = &(detect_result_group.results[i]);
        sprintf(text, "%s %.1f%%", det_result->name, det_result->prop * 100);
        int x1 = det_result->box.left;
        int y1 = det_result->box.top;
        rectangle(m_srcImage, cv::Point(x1, y1), cv::Point(det_result->box.right, det_result->box.bottom), cv::Scalar(0, 0, 255, 0), 3);
        putText(m_srcImage, text, cv::Point(x1, y1 + 12), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));
    }
    m_ret = rknn_outputs_release(m_rknnCtx, m_inputOutputNum.n_output, outputs);

    return 0;
}

