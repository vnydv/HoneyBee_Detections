
#ifndef __CONFIG_H__
#define __CONFIG_H__

#include <string>
#include <opencv2/opencv.hpp>

namespace VideoProcessing {
    
    const std::string VIDEO_PATH = "/home/Videos/data/vidTest1.avi";
    const std::string WINDOW_NAME = "Output";
    
    const int DILATE_ITERATIONS = 2;
    const int MEDIAN_BLUR_KERNEL_SIZE = 3;

    const int THRESHOLD = 50;

    const cv::Size KERNEL_SIZE = cv::Size(3, 3);
    const cv::Size MORPH_KERNEL_SIZE = cv::Size(3, 3);

    const cv::Scalar RED_COLOR = cv::Scalar(0, 0, 255);
    const cv::Scalar GREEN_COLOR = cv::Scalar(0, 255, 0);
    const cv::Scalar BLUE_COLOR = cv::Scalar(255, 0, 0);

    const int MIN_THRESHOLD = 128;

    const int MOG_LEARNING_RATE = 0.01;

    const int MIN_AREA = 1000;
    const int MAX_AREA = 10000;

} // namespace VideoProcessing

#endif // __CONFIG_H__