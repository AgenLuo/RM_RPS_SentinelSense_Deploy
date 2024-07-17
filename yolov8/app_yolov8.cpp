#include"../utils/yolo.h"
#include"yolov8.h"
#include <thread>
#include "../include/serialport.h"

SerialPort serialPort1;

void setParameters(utils::InitParameter &initParameters) {
    initParameters.class_names = utils::dataSets::RM;
    initParameters.num_class = 12;
    initParameters.batch_size = 1;
    initParameters.dst_h = 640;
    initParameters.dst_w = 640;
    initParameters.input_output_names = {"images", "output0"};
    initParameters.conf_thresh = 0.5f;
    initParameters.iou_thresh = 0.7f;
    initParameters.save_path = "";
}

void task(YOLOV8& yolo, const utils::InitParameter& param, std::vector<cv::Mat>& imgsBatch, const int& delayTime, const int& batchi,
          const bool& isShow, const bool& isSave)
{
    //前处理、推理、后处理
    utils::DeviceTimer d_t0; yolo.copy(imgsBatch);	      float t0 = d_t0.getUsedTime();
    utils::DeviceTimer d_t1; yolo.preprocess(imgsBatch);  float t1 = d_t1.getUsedTime();
    utils::DeviceTimer d_t2; yolo.infer();				  float t2 = d_t2.getUsedTime();
    utils::DeviceTimer d_t3; yolo.postprocess(imgsBatch); float t3 = d_t3.getUsedTime();
    sample::gLogInfo <<
                     //"copy time = " << t0 / param.batch_size << "; "
                     "preprocess time = " << t1 / param.batch_size << "; "
                     "infer time = " << t2 / param.batch_size << "; "
                     "postprocess time = "<< t3 / param.batch_size << std::endl;
    // 调用 getitom 函数（内含测试参数）
    std::vector<utils::Result> results = utils::getitom(yolo.getObjectss(), param.class_names, delayTime, imgsBatch);
    // 调用 sendmessage 函数(内含串口参数)
    utils::sendmessage(serialPort1.fd,results);
    if (isShow)
        utils::show(yolo.getObjectss(), param.class_names, delayTime, imgsBatch);
    if (isSave)
        utils::save(yolo.getObjectss(), param.class_names, param.save_path, imgsBatch, param.batch_size, batchi);
    yolo.reset();
}

int main(int argc, char **argv) {
    // 程序初始化
    utils::InitParameter param;
    // yolo 相关初始参数设置（自行进入修改）
    setParameters(param);

    // path 相关初始参数路径
    std::string model_path = "../best.trt";
    std::string video_path = "../../data/people.mp4";
    std::string image_path = "../../data/bus.jpg";
    std::string yaml_path = "/home/plusseven/桌面/RM_RPS_SentinelSense_Deploy/config.yaml";

    // camera' id 相机id
    int camera_id = 0;

    char *port_path1 = "/dev/ttyUSB0";
    serialPort1.initSerialPort(port_path1);
    tcflush(serialPort1.fd, TCIFLUSH);

    // get input 默认参数
    utils::InputStream source;
//    source = utils::InputStream::IMAGE;
//    source = utils::InputStream::VIDEO;
//    source = utils::InputStream::CAMERA;

    int size = -1; // w or h
    int batch_size = 1;
    bool is_show = false;
    bool is_save = false;

//    YAML::Node config = YAML::LoadFile(yaml_path);
//    try {
        YAML::Node config = YAML::LoadFile(yaml_path);
        if (config["mode"]) {
            if (config["mode"]["debug"]) {
                if (config["config_de"]["model"])
                    model_path = config["config_de"]["model"].as<std::string>();
                if (config["config_de"]["size"])
                    size = config["config_de"]["size"].as<int>();
                if (config["config_de"]["batch_size"])
                    batch_size = config["config_de"]["batch_size"].as<int>();
                if (config["config_de"]["video"]){
                    source = utils::InputStream::VIDEO;
                    video_path = config["config_de"]["video"].as<std::string>();}
                if (config["config_de"]["img"]){
                    source = utils::InputStream::IMAGE;
                    image_path = config["config_de"]["img"].as<std::string>();}
                if (config["config_de"]["cam_id"]){
                    source = utils::InputStream::CAMERA;
                    camera_id = config["config_de"]["cam_id"].as<int>();}
                if (config["config_de"]["show"])
                    is_show = config["config_de"]["show"].as<bool>();
                if (config["config_de"]["save"])
                    is_save = config["config_de"]["save"].as<bool>();
                if (is_save) {
                    param.save_path = config["config_de"]["savePath"].as<std::string>();
                }
            } else if (config["mode"]["standard"]) {
                if (config["config_st"]["model"])
                    model_path = config["config_st"]["model"].as<std::string>();
                if (config["config_st"]["size"])
                    size = config["config_st"]["size"].as<int>();
                if (config["config_st"]["batch_size"])
                    batch_size = config["config_st"]["batch_size"].as<int>();
                if (config["config_st"]["video"]){
                    source = utils::InputStream::VIDEO;
                    video_path = config["config_st"]["video"].as<std::string>();}
                if (config["config_st"]["img"]){
                    source = utils::InputStream::IMAGE;
                    image_path = config["config_st"]["img"].as<std::string>();}
                if (config["config_st"]["cam_id"]){
                    source = utils::InputStream::CAMERA;
                    camera_id = config["config_st"]["cam_id"].as<int>();}
                if (config["config_st"]["show"])
                    is_show = config["config_st"]["show"].as<bool>();
                if (config["config_st"]["save"])
                    is_save = config["config_st"]["save"].as<bool>();
                if (is_save) {
                    param.save_path = config["config_st"]["savePath"].as<std::string>();
                }
            }
            param.batch_size = batch_size;
            param.dst_h = param.dst_w = size;
        }
//    }
//    catch (...) {
//        std::cerr << "Init with yaml failed" << std::endl;
//    }

    int total_batches = 0;
    int delay_time = 1;
    cv::VideoCapture capture;

    sample::gLogInfo << "Config below:image_path::" << image_path << ";video_path::" << video_path << ";camera_id::"
                     << camera_id << ";is_show::" << is_show << ";is_save::" << is_save << std::endl;
    if (!setInputStream(source, image_path, video_path, camera_id,
                        capture, total_batches, delay_time, param))
        //判断初始化是否成功
    {
        sample::gLogError << "read the input data errors!" << std::endl;
        return -1;
    }
    YOLOV8 yolo(param);
    // read model
    std::vector<unsigned char> trt_file = utils::loadModel(model_path);
    if (trt_file.empty()) {
        sample::gLogError << "trt_file is empty!" << std::endl;
        return -1;
    }
    // init model
    if (!yolo.init(trt_file)) {
        sample::gLogError << "initEngine() ocur errors!" << std::endl;
        return -1;
    }
    //初始化引擎
    yolo.check();
    cv::Mat frame;
    std::vector<cv::Mat> imgs_batch;
    imgs_batch.reserve(param.batch_size);
    sample::gLogInfo << imgs_batch.capacity() << std::endl;
    int batchi = 0;
    while (capture.isOpened())
    {
        if (batchi >= total_batches && source != utils::InputStream::CAMERA)
        {
            break;
        }
        if (imgs_batch.size() < param.batch_size) // get input
        {
            if (source != utils::InputStream::IMAGE)
            {
                capture.read(frame);
            }
            else
            {
                frame = cv::imread(image_path);
            }

            if (frame.empty())
            {
                sample::gLogWarning << "no more video or camera frame" << std::endl;
                task(yolo, param, imgs_batch, delay_time, batchi, is_show, is_save);
                imgs_batch.clear();
                batchi++;
                break;
            }
            else
            {
                imgs_batch.emplace_back(frame.clone());
            }
        }
        else
        {
            task(yolo, param, imgs_batch, delay_time, batchi, is_show, is_save);
            imgs_batch.clear();
            batchi++;
        }
    }
    return  -1;
}
