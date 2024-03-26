#include "../include/videoProcessing.h"
#include "../include/config.h"
#include <iostream>
#include <stdexcept>

namespace ErrorHandling {

    void handleError(const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    // TODO: add more checks...

}

int main(int argc, char** argv) {

    VideoProcessing::VideoProcessor processor = VideoProcessing::VideoProcessor(VideoProcessing::VIDEO_PATH);

    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <input_video_path>" << std::endl;        
        std::cerr << "... Using default video path: " << VideoProcessing::VIDEO_PATH << std::endl;        
    } else{
        // delete processor
        
        processor.~VideoProcessor();
        processor = VideoProcessing::VideoProcessor(argv[1]);
    }

    try {
        processor.processVideo();
    } catch (const std::exception& e) {
        ErrorHandling::handleError(e);
        return -1;
    }

    return 0;
}
