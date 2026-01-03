/**
 * @file ImageBlurScore.h
 * @brief Image blur detection and scoring library
 *
 * This header-only library provides multiple algorithms for computing
 * blur scores and sharpness metrics for images using OpenCV.
 */

#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>

/**
 * @enum BlurMethod
 * @brief Available blur detection algorithms
 */
enum class BlurMethod {
    LAPLACIAN,              ///< Laplacian variance method
    FOURIER,                ///< Fourier transform-based method
    GRADIENT_MAGNITUDE,     ///< Sobel gradient magnitude method
    CONTRAST,               ///< Contrast/standard deviation method
    PHASE_CORRELATION,      ///< Phase correlation method
    WAVELET                 ///< Wavelet decomposition method
};

/**
 * @class ImageBlurScore
 * @brief Computes blur score for images using various algorithms
 *
 * This class provides multiple methods to assess image blur and sharpness,
 * including Laplacian, Fourier, gradient, and wavelet-based approaches.
 */
class ImageBlurScore {
public:
    /**
     * @brief Computes blur score using the specified method
     * @param image Input image (BGR or grayscale)
     * @param method Blur detection algorithm to use (default: LAPLACIAN)
     * @return Blur score (higher = sharper, less blurred)
     */
    double operator()(const cv::Mat& image, BlurMethod method = BlurMethod::LAPLACIAN) const {
        return compute(image, method);
    }

    /**
     * @brief Computes blur score using the specified method
     * @param image Input image (BGR or grayscale)
     * @param method Blur detection algorithm to use
     * @return Blur score (higher = sharper, less blurred)
     */
    double compute(const cv::Mat& image, BlurMethod method = BlurMethod::LAPLACIAN) const {
        switch (method) {
            case BlurMethod::LAPLACIAN:
                return computeLaplacianScore(image);
            case BlurMethod::FOURIER:
                return computeFourierBlurScore(image);
            case BlurMethod::GRADIENT_MAGNITUDE:
                return computeGradientMagnitudeScore(image);
            case BlurMethod::CONTRAST:
                return computeContrastScore(image);
            case BlurMethod::PHASE_CORRELATION:
                return computePhaseCorrelationScore(image);
            case BlurMethod::WAVELET:
                return computeWaveletBlurScore(image);
            default:
                return computeLaplacianScore(image);
        }
    }

private:
    double computeLaplacianScore(const cv::Mat& image) const {
        cv::Mat gray, laplacian;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        cv::Laplacian(gray, laplacian, CV_64F);
        cv::Scalar mu, sigma;
        cv::meanStdDev(laplacian, mu, sigma);
        return sigma.val[0] * sigma.val[0];
    }

    double computeFourierBlurScore(const cv::Mat& image) {
        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

        // Compute DFT (Discrete Fourier Transform)
        cv::Mat planes[] = {cv::Mat_<float>(gray), cv::Mat::zeros(gray.size(), CV_32F)};
        cv::Mat complexI;
        cv::merge(planes, 2, complexI);
        cv::dft(complexI, complexI);

        // Compute magnitude of DFT
        cv::split(complexI, planes);
        cv::magnitude(planes[0], planes[1], planes[0]);
        cv::Mat magI = planes[0];

        // Focus measure may be variance of magnitude in frequency domain
        cv::Scalar mean, stddev;
        cv::meanStdDev(magI, mean, stddev);

        return stddev[0] * stddev[0];
    }

    double computeGradientMagnitudeScore(const cv::Mat& image) {
        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

        cv::Mat grad_x, grad_y;
        cv::Sobel(gray, grad_x, CV_32F, 1, 0, 3);
        cv::Sobel(gray, grad_y, CV_32F, 0, 1, 3);

        cv::Mat magnitude;
        cv::magnitude(grad_x, grad_y, magnitude);

        cv::Scalar stddev;
        cv::meanStdDev(magnitude, cv::noArray(), stddev);

        return stddev[0];
    }

    double computeContrastScore(const cv::Mat& image) {
        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

        cv::Scalar stddev;
        cv::meanStdDev(gray, cv::noArray(), stddev);

        return stddev[0];
    }

    double computePhaseCorrelationScore(const cv::Mat& image) {
        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

        // Create a shifted version of the image
        cv::Mat shifted = cv::Mat::zeros(gray.size(), gray.type());
        gray(cv::Rect(1, 1, gray.cols - 1, gray.rows - 1)).copyTo(shifted(cv::Rect(0, 0, gray.cols - 1, gray.rows - 1)));

        cv::Mat result;
        cv::phaseCorrelateRes(gray, shifted, result);

        double peak_height = result.at<double>(0, 3);

        return peak_height;
    }
    double computeWaveletBlurScore(const cv::Mat& image) {
        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

        // Iteratively down-sample the image and subtract the smoothed version to
        // approximate the wavelet detail coefficients
        cv::Mat down = gray.clone();
        cv::Mat up;
        int num_levels = 4;
        cv::Mat wavelet_approx = cv::Mat::zeros(gray.size(), CV_32F);

        for(int i = 0; i < num_levels; i++) {
            cv::pyrDown(down, down);
            cv::pyrUp(down, up, wavelet_approx.size());
            wavelet_approx += gray - up;
        }

        // Measure the amount of high-frequency content by measuring
        // the standard deviation of the wavelet coefficients
        cv::Scalar mean, stddev;
        cv::meanStdDev(wavelet_approx, mean, stddev);

        return stddev[0];
    }

    cv::Mat computeDarkChannel(const cv::Mat& src, int patchSize = 15) {
        cv::Mat darkChannel = cv::Mat::zeros(src.size(), CV_8UC1);
        int halfPatch = patchSize / 2;
        for (int i = 0; i < src.rows; i++) {
            for (int j = 0; j < src.cols; j++) {
                int darkValue = 255;
                for (int k = -halfPatch; k <= halfPatch; k++) {
                    for (int l = -halfPatch; l <= halfPatch; l++) {
                        int newI = std::clamp(i + k, 0, src.rows - 1);
                        int newJ = std::clamp(j + l, 0, src.cols - 1);
                        const cv::Vec3b& pixel = src.at<cv::Vec3b>(newI, newJ);
                        for (int c = 0; c < src.channels(); c++) {
                            darkValue = std::min(darkValue, (int)pixel[c]);
                        }
                    }
                }
                darkChannel.at<uchar>(i, j) = darkValue;
            }
        }
        return darkChannel;
    }

    double computeBlurScore(const cv::Mat& darkChannel) {
        cv::Scalar stddev;
        cv::meanStdDev(darkChannel, cv::noArray(), stddev);
        return stddev[0];
    }

};
