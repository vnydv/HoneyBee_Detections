#include "../include/videoProcessing.h"
#include "../include/config.h"
#include <iostream>
#include <stdexcept>

namespace VideoProcessing {

    VideoProcessor::VideoProcessor(const std::string& videoPath) {
        videoCapture.open(videoPath);

        if (!videoCapture.isOpened()) {
            throw std::runtime_error("Error: Failed to open the video file.");
        }
        else{
            std::cout << "Video file opened successfully." << std::endl;
        }

        cv::cuda::Stream stream_r, stream_g, stream_b, stream_merge;
        initializeFilters();
    }

    VideoProcessor::~VideoProcessor() {
        videoCapture.release();
    }

    void VideoProcessor::displayFrame(const cv::cuda::GpuMat& frame, const std::string& windowName) {
        cv::Mat outputFrame;
        frame.download(outputFrame);
        cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
        cv::imshow(windowName, outputFrame);
    }

    void VideoProcessor::processVideo() {
        cv::Mat frame;
        cv::cuda::GpuMat d_input_frame, d_output_frame;

        while (videoCapture.read(frame)) {
            d_input_frame.upload(frame, stream_merge);
            processFrame(d_input_frame, d_output_frame);

            cv::Mat outputFrame;
            if (d_output_frame.empty()) {
                throw std::runtime_error("Error: Failed to process frame.");
            }            
            
            d_output.download(outputFrame, stream_merge);
            
            std::vector<cv::Rect> rect_array;
            getBoundBox(outputFrame, &rect_array);

            // draw bounding boxes
            for (size_t i = 0; i < rect_array.size(); i++) {
                cv::rectangle(frame, rect_array[i].tl(), rect_array[i].br(), GREEN_COLOR, 1);
            }

            cv::imshow("Output", frame);

            // Press 'Esc' to exits
            if (cv::waitKey(1) == 27) {
                break;
            }
        }
    }

    void VideoProcessor::initializeFilters() {
        cv::Mat dilateKernel = cv::getStructuringElement(cv::MORPH_RECT, KERNEL_SIZE);
        dilateFilter = cv::cuda::createMorphologyFilter(cv::MORPH_DILATE, CV_8UC1, dilateKernel, cv::Point(-1, -1), 1);
        if (dilateFilter.empty())
            throw std::runtime_error("Error: Failed to create dilation filter.");

        medianFilter = cv::cuda::createMedianFilter(CV_8UC1, MEDIAN_BLUR_KERNEL_SIZE);
        if (medianFilter.empty())
            throw std::runtime_error("Error: Failed to create median filter.");

        bgSubtractor = cv::cuda::createBackgroundSubtractorMOG2();
        if (bgSubtractor.empty())
            throw std::runtime_error("Error: Failed to create background subtractor.");

        cv::Mat morphExCrossKernel = cv::getStructuringElement(cv::MORPH_CROSS, MORPH_KERNEL_SIZE);
        morphOpenFilter = cv::cuda::createMorphologyFilter(cv::MORPH_OPEN, CV_8UC1, morphExCrossKernel, cv::Point(-1, -1), 1);
        if (morphOpenFilter.empty())
            throw std::runtime_error("Error: Failed to create morphological opening filter.");

        morphCloseFilter = cv::cuda::createMorphologyFilter(cv::MORPH_CLOSE, CV_8UC1, morphExCrossKernel, cv::Point(-1, -1), 1);
        if (morphCloseFilter.empty())
            throw std::runtime_error("Error: Failed to create morphological closing filter.");
    }


    void VideoProcessor::processFrame(const cv::cuda::GpuMat& inputFrame, cv::cuda::GpuMat& outputFrame) {
        // individual channel processing
        // rgb normalization
        splitChannels(inputFrame, d_bgr);
        dilateChannels(d_bgr, d_dilated);

        medianBlurChannels(d_dilated, d_dilated);
        computeDifferenceImages(d_bgr, d_dilated, d_diff_image);
        normalizeChannels(d_diff_image, d_diff_image);

        // sync the streams
        stream_r.waitForCompletion();
        stream_g.waitForCompletion();
        stream_b.waitForCompletion();

        mergeChannels(d_diff_image, d_output);

        // apply background subtraction
        applyThreshold(d_output, d_output);
        backgroundSubtraction(d_output, d_output);

        // apply morphological operations for noise reduction
        morphologicalOperations(d_output, d_output);

        // sync this stream
        stream_merge.waitForCompletion();

        outputFrame = d_output;
    }

    void VideoProcessor::splitChannels(const cv::cuda::GpuMat& inputFrame, cv::cuda::GpuMat outputChannels[]) {
        cv::cuda::split(inputFrame, outputChannels, stream_merge);
    }

    void VideoProcessor::dilateChannels(const cv::cuda::GpuMat inputChannels[], cv::cuda::GpuMat outputChannels[]) {
        dilateFilter->apply(inputChannels[0], outputChannels[0], stream_r);
        dilateFilter->apply(inputChannels[1], outputChannels[1], stream_g);
        dilateFilter->apply(inputChannels[2], outputChannels[2], stream_b);
    }

    void VideoProcessor::medianBlurChannels(cv::cuda::GpuMat inputChannels[], cv::cuda::GpuMat outputChannels[]) {
        medianFilter->apply(inputChannels[0], outputChannels[0], stream_r);
        medianFilter->apply(inputChannels[1], outputChannels[1], stream_g);
        medianFilter->apply(inputChannels[2], outputChannels[2], stream_b);
    }

    void VideoProcessor::computeDifferenceImages(const cv::cuda::GpuMat inputChannels[], const cv::cuda::GpuMat dilatedChannels[], cv::cuda::GpuMat diffImages[]) {
        cv::cuda::subtract(inputChannels[0], dilatedChannels[0], diffImages[0], cv::noArray(), -1, stream_r);
        cv::cuda::subtract(inputChannels[1], dilatedChannels[1], diffImages[1], cv::noArray(), -1, stream_g);
        cv::cuda::subtract(inputChannels[2], dilatedChannels[2], diffImages[2], cv::noArray(), -1, stream_b);
    }

    void VideoProcessor::normalizeChannels(cv::cuda::GpuMat inputChannels[], cv::cuda::GpuMat outputChannels[]) {
        cv::cuda::normalize(inputChannels[0], outputChannels[0], 0, 255, cv::NORM_MINMAX, CV_8UC1, cv::noArray(), stream_r);
        cv::cuda::normalize(inputChannels[1], outputChannels[1], 0, 255, cv::NORM_MINMAX, CV_8UC1, cv::noArray(), stream_g);
        cv::cuda::normalize(inputChannels[2], outputChannels[2], 0, 255, cv::NORM_MINMAX, CV_8UC1, cv::noArray(), stream_b);
    }

    void VideoProcessor::mergeChannels(const cv::cuda::GpuMat inputChannels[], cv::cuda::GpuMat& outputFrame) {
        cv::cuda::merge(inputChannels, 3, outputFrame, stream_merge);
    }

    void VideoProcessor::applyThreshold(const cv::cuda::GpuMat& inputFrame, cv::cuda::GpuMat& outputFrame) {
        cv::cuda::threshold(inputFrame, outputFrame, MIN_THRESHOLD, 255, cv::THRESH_BINARY, stream_merge);
    }

    void VideoProcessor::backgroundSubtraction(const cv::cuda::GpuMat& inputFrame, cv::cuda::GpuMat& outputFrame) {
        bgSubtractor->apply(inputFrame, outputFrame, MOG_LEARNING_RATE, stream_merge);
    }

    void VideoProcessor::morphologicalOperations(const cv::cuda::GpuMat& inputFrame, cv::cuda::GpuMat& outputFrame) {
        cv::cuda::GpuMat temp;
        medianFilter->apply(inputFrame, temp, stream_merge);
        morphOpenFilter->apply(temp, outputFrame, stream_merge);
        cv::cuda::threshold(outputFrame, outputFrame, MIN_THRESHOLD , 255, cv::THRESH_BINARY, stream_merge);
        morphCloseFilter->apply(outputFrame, temp, stream_merge);
        cv::cuda::threshold(temp, outputFrame, MIN_THRESHOLD, 255, cv::THRESH_BINARY, stream_merge);
    }

    void VideoProcessor::getBoundBox(cv::Mat image, std::vector<cv::Rect> *rect_array){
        std::vector<std::vector<cv::Point>> contours_;
        std::vector<cv::Vec4i> hierarchy;
                
        cv::findContours(image, contours_, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
        
        // chain contours if area > min_contours_area
        for (size_t i = 0; i < contours_.size(); i++) {
            if (cv::contourArea(contours_[i]) > MIN_AREA) {
                cv::Rect boundRect = cv::boundingRect(contours_[i]);
                rect_array->push_back(boundRect);
            }
        }
    }
}