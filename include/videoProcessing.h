
#ifndef __VIDEO_PROCESSING_H__
#define __VIDEO_PROCESSING_H__

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudabgsegm.hpp>
#include <string>
#include <vector>

// Video Processing Module
namespace VideoProcessing {

    class VideoProcessor {
        public:
            VideoProcessor(const std::string& videoPath);
            ~VideoProcessor();

            void processVideo();

        private:
            void initializeStream(cv::cuda::Stream& stream, uint32_t flags);
            void initializeFilters();

            void processFrame(const cv::cuda::GpuMat& inputFrame, cv::cuda::GpuMat& outputFrame);
            void splitChannels(const cv::cuda::GpuMat& inputFrame, cv::cuda::GpuMat outputChannels[]);
            void dilateChannels(const cv::cuda::GpuMat inputChannels[], cv::cuda::GpuMat outputChannels[]);
            void medianBlurChannels(cv::cuda::GpuMat inputChannels[], cv::cuda::GpuMat outputChannels[]);
            void computeDifferenceImages(const cv::cuda::GpuMat inputChannels[], const cv::cuda::GpuMat dilatedChannels[], cv::cuda::GpuMat diffImages[]);
            void normalizeChannels(cv::cuda::GpuMat inputChannels[], cv::cuda::GpuMat outputChannels[]);
            void mergeChannels(const cv::cuda::GpuMat inputChannels[], cv::cuda::GpuMat& outputFrame);
            void applyThreshold(const cv::cuda::GpuMat& inputFrame, cv::cuda::GpuMat& outputFrame);
            void backgroundSubtraction(const cv::cuda::GpuMat& inputFrame, cv::cuda::GpuMat& outputFrame);
            void morphologicalOperations(const cv::cuda::GpuMat& inputFrame, cv::cuda::GpuMat& outputFrame);            
            void getBoundBox(cv::Mat image, std::vector<cv::Rect> *rect_array);
            void displayFrame(const cv::cuda::GpuMat& frame, const std::string& windowName);


            cv::VideoCapture videoCapture;
            cv::cuda::GpuMat d_frame, d_bgr[3], d_dilated[3], d_diff_image[3], d_output;
            cv::cuda::Stream stream_r, stream_g, stream_b, stream_merge;
            cv::Ptr<cv::cuda::Filter> dilateFilter, medianFilter;
            cv::Ptr<cv::cuda::BackgroundSubtractorMOG2> bgSubtractor;
            cv::Ptr<cv::cuda::Filter> morphOpenFilter, morphCloseFilter;
        };

} // namespace VideoProcessing


#endif // __VIDEO_PROCESSING_H__