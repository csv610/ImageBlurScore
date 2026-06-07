#include "ImageBlurScore.h"
#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return -1;
    }

    cv::Mat image = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cout << "Could not open or find the image!" << std::endl;
        return -1;
    }

    blur::ImageBlurScore blurscorer;

    double score = blurscorer(image);
    std::cout << "Blur Score (Laplacian): " << score << std::endl;

    score = blurscorer(image, blur::BlurMethod::FOURIER);
    std::cout << "Blur Score (Fourier):   " << score << std::endl;

    score = blurscorer(image, blur::BlurMethod::GRADIENT_MAGNITUDE);
    std::cout << "Blur Score (Gradient):  " << score << std::endl;

    score = blurscorer(image, blur::BlurMethod::CONTRAST);
    std::cout << "Blur Score (Contrast):  " << score << std::endl;

    score = blurscorer(image, blur::BlurMethod::WAVELET);
    std::cout << "Blur Score (Wavelet):   " << score << std::endl;

    score = blurscorer(image, blur::BlurMethod::BRENNER);
    std::cout << "Blur Score (Brenner):   " << score << std::endl;

    score = blurscorer(image, blur::BlurMethod::TENENGRAD);
    std::cout << "Blur Score (Tenengrad): " << score << std::endl;

    cv::Mat grid = blurscorer(image, 4, 4);
    std::cout << "Grid (4x4, Laplacian):" << std::endl;
    for (int r = 0; r < grid.rows; r++) {
        for (int c = 0; c < grid.cols; c++) {
            std::cout << grid.at<double>(r, c) << "\t";
        }
        std::cout << std::endl;
    }

    return 0;
}
