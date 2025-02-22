cmake_minimum_required(VERSION 3.10)

#set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_BUILD_TYPE "Release")

PROJECT(yolov8 VERSION 1.0.0 LANGUAGES C CXX CUDA)
message(STATUS CMAKE_CURRENT_SOURCE_DIR = ${CMAKE_CURRENT_SOURCE_DIR})
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/common.cmake)
message(STATUS TensorRT_ROOT = ${TensorRT_ROOT})
message(STATUS ${ALL_LIBS})
file(GLOB CPPS 
  ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/*.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/../utils/*.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/../utils/*.cpp
  ${TensorRT_ROOT}/samples/common/logger.cpp 
  ${TensorRT_ROOT}/samples/common/sampleOptions.cpp
  ${TensorRT_ROOT}/samples/common/sampleUtils.cpp
  )
list(REMOVE_ITEM CPPS app_yolov8.cpp)
message(STATUS CPPS = ${CPPS})
list (LENGTH CPPS length)
message(STATUS ***length*** = ${length}) 
find_package(OpenCV REQUIRED)
include_directories(${INCLUDE_DRIS} ${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR} /usr/src/tensorrt/samples/common)
message(STATUS ***INCLUDE_DRIS*** = ${INCLUDE_DRIS})
message(STATUS ***OpenCV_INCLUDE_DIRS*** = ${OpenCV_INCLUDE_DIRS})
add_library(${PROJECT_NAME} SHARED ${CPPS} /usr/src/tensorrt/samples/common/logger.cpp)
target_link_libraries(${PROJECT_NAME} ${ALL_LIBS} ${OpenCV_LIBRARIES})


set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 50 61 72 75)
target_compile_options(${PROJECT_NAME} PUBLIC 
   $<$<COMPILE_LANGUAGE:CUDA>:--default-stream per-thread -lineinfo; --use_fast_math --disable-warnings>)
  
add_executable(app_yolov8 app_yolov8.cpp
                          ../src/CRC_Check.cpp
                          ../src/Content.cpp
                          ../src/serialport.cpp
                )

#yaml-cpp
find_package(yaml-cpp REQUIRED)
include_directories(/usr/local/include/yaml-cpp)
target_link_libraries(${PROJECT_NAME} /usr/local/lib/libyaml-cpp.so)

# 在Linux系统下，多线程模块并非自动链接的，需要额外链接。
if(CMAKE_SYSTEM_NAME MATCHES "Linux")
  find_package(Threads)
  target_link_libraries(${PROJECT_NAME}  ${CMAKE_THREAD_LIBS_INIT})
endif()

# NVCC
# target_link_libraries(detect ${PROJECT_NAME} ${ALL_LIBS} ${OpenCV_LIBRARIES} libgflags_nothreads.a)
target_link_libraries(app_yolov8 ${PROJECT_NAME} ${ALL_LIBS} ${OpenCV_LIBRARIES} )
