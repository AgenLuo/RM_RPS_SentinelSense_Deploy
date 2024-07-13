#include"../utils/yolo.h"
#include"yolov8.h"
#include <thread>
#include "../include/serialport.h"

void setParameters(utils::InitParameter& initParameters)
{
//	initParameters.class_names = utils::dataSets::coco80;
    initParameters.class_names = utils::dataSets::RM;
//	initParameters.class_names = utils::dataSets::voc20;
//	initParameters.num_class = 80; // for coco
    initParameters.num_class = 12;
//	initParameters.num_class = 20; // for voc2012
	initParameters.batch_size = 1;
	initParameters.dst_h = 640;
	initParameters.dst_w = 640;
	initParameters.input_output_names = { "images",  "output0" };
	initParameters.conf_thresh = 0.25f;
	initParameters.iou_thresh = 0.45f;
	initParameters.save_path = "";
}

void task(YOLOV8& yolo, const utils::InitParameter& param, std::vector<cv::Mat>& imgsBatch, const int& delayTime, const int& batchi,
	const bool& isShow, const bool& isSave)
{
    utils::DeviceTimer d_t0; yolo.copy(imgsBatch);	      float t0 = d_t0.getUsedTime();
    utils::DeviceTimer d_t1; yolo.preprocess(imgsBatch);  float t1 = d_t1.getUsedTime();
    utils::DeviceTimer d_t2; yolo.infer();				  float t2 = d_t2.getUsedTime();
    utils::DeviceTimer d_t3; yolo.postprocess(imgsBatch); float t3 = d_t3.getUsedTime();
	sample::gLogInfo << 
		//"copy time = " << t0 / param.batch_size << "; "
		"preprocess time = " << t1 / param.batch_size << "; "
		"infer time = " << t2 / param.batch_size << "; "
		"postprocess time = " << t3 / param.batch_size << std::endl;
    // 调用 getitom 函数
    std::vector<utils::Result> results = utils::getitom(yolo.getObjectss(), param.class_names, delayTime, imgsBatch);
    // 遍历结果向量并处理每个元素
    char *port_path1 = "/dev/ttyUSB0";
    SerialPort serialPort1;
    serialPort1.initSerialPort(port_path1);
    tcflush(serialPort1.fd, TCIFLUSH);
    char sig[3] = "ED";
    for (const auto& result : results)
    {
        char str[9] = "ST000000";

//        std::cout<<result.className<<"::"<<result.conf<<"%"<<"::"<<result.x<<"::"<<result.y<<std::endl;
        if (result.conf >= 0.6)
        {
            switch(result.x)
            {
                case 1 ... 216 :
                    std::cout << "A" << std::endl;
                    str[2] = 'A';
                    break;
                case 217 ... 432 :
                    std::cout << "B" << std::endl;
                    str[2] = 'B';
                    break;
                case 433 ... 648 :
                    std::cout << "C" << std::endl;
                    str[2] = 'C';
                    break;
                case 649 ... 864 :
                    std::cout << "D" << std::endl;
                    str[2] = 'D';
                    break;
                case 865 ... 1080:
                    std::cout << "E" << std::endl;
                    str[2] = 'E';
                    break;
            }
            switch(result.y)
            {
                case 1 ... 200 :
                    std::cout << "1" << std::endl;
                    str[3] = '1';
                    break;
                case 201 ... 400 :
                    std::cout << "2" << std::endl;
                    str[3] = '2';
                    break;
                case 401 ... 600 :
                    std::cout << "3" << std::endl;
                    str[3] = '3';
                    break;
            }
            strncpy(str + 4, result.className.c_str(), result.className.size()); // 拷贝string到char数组中，从第3个位置开始拷贝
            strncpy(str + 6, sig, 2); // 拷贝string到char数组中，从第3个位置开始拷贝
            std::cout <<"OUT::"<< str << std::endl; // 打印拼接后的结果
            write(serialPort1.fd, str, 6); // 写入6个字符到串口
        }
    }
	if(isShow)
		utils::show(yolo.getObjectss(), param.class_names, delayTime, imgsBatch);
	if(isSave)
		utils::save(yolo.getObjectss(), param.class_names, param.save_path, imgsBatch, param.batch_size, batchi);
	yolo.reset();
}

int main(int argc, char** argv)
{
    cv::CommandLineParser parser(argc, argv,
                                 {
                                         "{model         || tensorrt model file     }"
                                         "{size      || image (h, w), eg: 640   }"
                                         "{batch_size|| batch size              }"
                                         "{video     || video's path                        }"
                                         "{img       || image's path                        }"
                                         "{cam_id    || camera's device id          }"
                                         "{show      || if show the result          }"
                                         "{savePath  || save path, can be ignore}"
                                 });

    // parameters 这里需要进去修改类别等
	utils::InitParameter param;
	setParameters(param);

	// path 相关路径
	std::string model_path = "./best.trt";
	std::string video_path = "../../data/people.mp4";
	std::string image_path = "../../data/bus.jpg";
	// camera' id 相机id
	int camera_id = 0;

	// get input 默认参数
	utils::InputStream source;
//	source = utils::InputStream::IMAGE;
//	source = utils::InputStream::VIDEO;
//	source = utils::InputStream::CAMERA;

	// update params from command line parser 从命令行里更新参数
	int size = -1; // w or h
	int batch_size = 1;
	bool is_show = false;
	bool is_save = false;
	if(parser.has("model"))
	{
		model_path = parser.get<std::string>("model");
		sample::gLogInfo << "model_path = " << model_path << std::endl;
	}
	if(parser.has("size"))
	{
		size = parser.get<int>("size");
		sample::gLogInfo << "size = " << size << std::endl;
		param.dst_h = param.dst_w = size;
	}
	if(parser.has("batch_size"))
	{
		batch_size = parser.get<int>("batch_size");
		sample::gLogInfo << "batch_size = " << batch_size << std::endl;
		param.batch_size = batch_size;
	}
	if(parser.has("video"))
	{
		source = utils::InputStream::VIDEO;
		video_path = parser.get<std::string>("video");
		sample::gLogInfo << "video_path = " << video_path << std::endl;
	}
	if(parser.has("img"))
	{
		source = utils::InputStream::IMAGE;
		image_path = parser.get<std::string>("img");
		sample::gLogInfo << "image_path = " << image_path << std::endl;
	}
	if(parser.has("cam_id"))
	{
		source = utils::InputStream::CAMERA;
		camera_id = parser.get<int>("cam_id");
		sample::gLogInfo << "camera_id = " << camera_id << std::endl;
	}
	if(parser.has("show"))
	{
		is_show = true;
		sample::gLogInfo << "is_show = " << is_show << std::endl;
	}
	if(parser.has("savePath"))
	{
		is_save = true;
		param.save_path = parser.get<std::string>("savePath");
		sample::gLogInfo << "save_path = " << param.save_path << std::endl;
	}

	int total_batches = 0;
	int delay_time = 1;
	cv::VideoCapture capture;

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
	if (trt_file.empty())
	{
		sample::gLogError << "trt_file is empty!" << std::endl;
		return -1;
	}
	// init model
	if (!yolo.init(trt_file))
	{
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

