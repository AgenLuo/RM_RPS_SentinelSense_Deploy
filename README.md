# RM_RPS_SentinelSense_Deploy

## 项目介绍
    本项目是基于FeiYull发布的TensorRT-Alpha进行改进从而实现哨兵全向感知功能的神经网络训练代码。
     
    本项目旨在通过Ultralytics YOLOv8神经网络实现哨兵对周围战场环境的事态感知，从而为火控、导航和决策提供更多的战场信息。
    
    本项目相较原版：替换了参数设置模式、增加了串口通信模块、修改了部分其他代码。
    
    注意：本项目只包括模型加速部署部分，模型训练请去Train仓库。

## 环境依赖
    Ubuntu20.04
    CUDA(JetPack自带)
    cudnn(JetPack自带)
    tensorrt(JetPack自带)
    opencv(JetPack自带)


## 目录结构描述
    ├── cmake                   // common cmake所在文件夹
    
    │   └── common.cmake        // 全局cmake配置

    ├── include                 // 串口头文件所在文件夹
    
    │   ├── Content.h           // 目标跟踪相关文件夹
    
    │   ├── CRC_Check.h         // CRC校验
    
    │   └── serialport.h        // 串口

    ├── src                     // 串口模块所在文件夹
    
    │   ├── Content.cpp         // 目标跟踪相关文件夹
    
    │   ├── CRC_Check.cpp       // CRC校验
    
    │   └── serialport.cpp      // 串口
    
    ├── tool                    // 工具脚本所在文件夹

    │   └── onnx2trt            // onnx转trt的工具（弃用）
    
    ├── utils                   // 工具库
    
    │   ├── tracking            // 跟踪器所在文件夹
    
    │   ├── common_include.h    // 数据处理相关函数和脚本文件夹
    
    │   ├── kernel_function.cu  // kernel运算
    
    │   ├── kernel_function.h   // kernel头文件
    
    │   ├── utils.cpp           // 工具函数
    
    │   ├── utils.h             // 工具函数头文件
    
    │   ├── yolo.cpp            // 模型相关函数
    
    │   └── yolo.h              // 模型相关函数头文件
    
    ├── yolov8                  // 主函数文件夹
    
    │   ├── app_yolov8.cpp      // 主函数
    
    │   ├── CMakeLists.txt      // cmake文件
    
    │   ├── decode_yolov8.cu    // yolov8解耦
    
    │   ├── decode_yolov8.h     // yolov8解耦头文件
     
    │   ├── ...                 // 杂七杂八的东西
    
    │   ├── yolov8.cpp          // yolo相关函数
    
    │   └── yolov8.h            // yolo相关函数头文件

    ├── config.yaml             // 主函数参数配置文件
    
    ├── include                 // 串口参数配置文件
    
    └── LICENSE                 // 开源声明证书

## 使用说明
1.修改通用cmake文件

```bash
cd tensorrt-alpha/cmake
vim common.cmake
# set var TensorRT_ROOT to your path in line 20, eg:
# set(TensorRT_ROOT /usr/src/tensorrt)
```
2.将onnx在本地转换成trt文件

```bash
# put your onnx file in this path:tensorrt-alpha/data/yolov8
cd tensorrt-alpha/data/yolov8
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/feiyull/TensorRT-8.4.2.4/lib
# official
/usr/src/tensorrt/bin/trtexec   --onnx=yolov8n.onnx  --saveEngine=yolov8n.trt  --buildOnly --minShapes=images:1x3x640x640 --optShapes=images:2x3x640x640 --maxShapes=images:4x3x640x640
/usr/src/tensorrt/bin/trtexec   --onnx=yolov8s.onnx  --saveEngine=yolov8s.trt  --buildOnly --minShapes=images:1x3x640x640 --optShapes=images:2x3x640x640 --maxShapes=images:4x3x640x640
/usr/src/tensorrt/bin/trtexec   --onnx=yolov8x.onnx  --saveEngine=yolov8x.trt  --buildOnly --minShapes=images:1x3x640x640 --optShapes=images:2x3x640x640 --maxShapes=images:4x3x640x640
# recommend(thats actually what i do, im lazy)
/usr/src/tensorrt/bin/trtexec   --onnx=yolov8n.onnx  --saveEngine=yolov8n.trt  --fp16
```
3.配置参数

```bash
vim config.yaml
根据需求自己改
vim Port_config.yaml
根据需求自己改
vim src/serialport.cpp
修改yaml文件路径
vim yolov8/app_yolov8.cpp
修改yaml文件路径
```
4.运行

```bash
cd tensorrt-alpha/yolov8
mkdir build
cd build
cmake ..
make -j4
./app_yolov8 
```

## 注意事项
    使用开机自启的服务后需要将所有用到的路径改为绝对路径，不然程序会异常终止

## 版本内容更新
###### v1.0.0:
    1、提交了基础代码并实现所有基础功能

 