#include "ImageBlurScore.h"
#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return -1;
    }

    // Read the input image
    cv::Mat image = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cout << "Could not open or find the image!" << std::endl;
        return -1;
    }

    ImageBlurScore blurscorer;

    // Calculate and display the blur score
    double blurScore = blurscorer(image);
    std::cout << "Blur Score: " << blurScore << std::endl;

    return 0;
}
