cmake_minimum_required(VERSION 3.10)
project( cudaDetections )
find_package( OpenCV REQUIRED )
find_package(CUDAToolkit 12.3 REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES 75)
include_directories( ${CUDAToolkit_INCLUDE_DIRS} )

include_directories( ${OpenCV_INCLUDE_DIRS} )

include_directories(./include)

add_executable(cudaDetections src/videoProcessing.cpp src/main.cpp )
target_link_libraries( cudaDetections ${OpenCV_LIBS} )