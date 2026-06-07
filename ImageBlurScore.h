/**
 * @file ImageBlurScore.h
 * @brief Image blur detection and scoring library
 *
 * This header-only library provides multiple algorithms for computing
 * blur scores and sharpness metrics for images using OpenCV.
 */

#pragma once
#include <stdexcept>
#include <opencv2/opencv.hpp>

namespace blur {

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
     * @brief Computes blur score for each block in a grid
     * @param image Input image (BGR or grayscale)
     * @param gridRows Number of rows in the grid
     * @param gridCols Number of columns in the grid
     * @param method Blur detection algorithm to use (default: LAPLACIAN)
     * @return CV_64F matrix of size gridRows x gridCols with per-block scores
     */
    cv::Mat operator()(const cv::Mat& image, int gridRows, int gridCols, BlurMethod method = BlurMethod::LAPLACIAN) const {
        return computeGrid(image, gridRows, gridCols, method);
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

    /**
     * @brief Computes blur score for each block in a grid
     * @param image Input image (BGR or grayscale)
     * @param gridRows Number of rows in the grid
     * @param gridCols Number of columns in the grid
     * @param method Blur detection algorithm to use (default: LAPLACIAN)
     * @return CV_64F matrix of size gridRows x gridCols with per-block scores
     */
    cv::Mat computeGrid(const cv::Mat& image, int gridRows, int gridCols, BlurMethod method = BlurMethod::LAPLACIAN) const {
        if (gridRows <= 0 || gridCols <= 0) {
            throw std::invalid_argument("gridRows and gridCols must be positive");
        }

        cv::Mat scores(gridRows, gridCols, CV_64F);

        int blockH = image.rows / gridRows;
        int blockW = image.cols / gridCols;

        for (int r = 0; r < gridRows; r++) {
            for (int c = 0; c < gridCols; c++) {
                int y1 = r * blockH;
                int x1 = c * blockW;
                int y2 = (r == gridRows - 1) ? image.rows : y1 + blockH;
                int x2 = (c == gridCols - 1) ? image.cols : x1 + blockW;

                cv::Rect roi(x1, y1, x2 - x1, y2 - y1);
                scores.at<double>(r, c) = compute(image(roi), method);
            }
        }

        return scores;
    }

private:
    double computeLaplacianScore(const cv::Mat& image) const {
        cv::Mat gray, laplacian;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        cv::Laplacian(gray, laplacian, CV_64F);
        cv::Scalar sigma;
        cv::meanStdDev(laplacian, cv::noArray(), sigma);
        return sigma.val[0] * sigma.val[0];
    }

    double computeFourierBlurScore(const cv::Mat& image) const {
        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

        // Compute DFT (Discrete Fourier Transform)
        cv::Mat planes[] = {cv::Mat_<float>(gray), cv::Mat::zeros(gray.size(), CV_32F)};
        cv::Mat complexI;
        cv::merge(planes, 2, complexI);
        cv::dft(complexI, complexI);

        // Compute log magnitude of DFT
        cv::split(complexI, planes);
        cv::magnitude(planes[0], planes[1], planes[0]);
        cv::Mat magI = planes[0];
        magI += cv::Scalar::all(1);
        cv::log(magI, magI);

        cv::Scalar mean, stddev;
        cv::meanStdDev(magI, mean, stddev);

        return stddev[0] * stddev[0];
    }

    double computeGradientMagnitudeScore(const cv::Mat& image) const {
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

    double computeContrastScore(const cv::Mat& image) const {
        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

        cv::Scalar stddev;
        cv::meanStdDev(gray, cv::noArray(), stddev);

        return stddev[0];
    }

    double computePhaseCorrelationScore(const cv::Mat& image) const {
        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        gray.convertTo(gray, CV_64F);

        cv::Mat shifted = cv::Mat::zeros(gray.size(), gray.type());
        gray(cv::Rect(1, 1, gray.cols - 1, gray.rows - 1)).copyTo(shifted(cv::Rect(0, 0, gray.cols - 1, gray.rows - 1)));

        double response = 0.0;
        cv::phaseCorrelate(gray, shifted, cv::noArray(), &response);

        return response;
    }
    double computeWaveletBlurScore(const cv::Mat& image) const {
        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        gray.convertTo(gray, CV_32F);

        // Multi-level wavelet-like decomposition using a Laplacian pyramid
        cv::Mat current = gray.clone();
        cv::Mat wavelet_detail = cv::Mat::zeros(gray.size(), CV_32F);
        int num_levels = 4;

        for(int i = 0; i < num_levels; i++) {
            cv::Mat down, up;
            cv::pyrDown(current, down);
            cv::pyrUp(down, up, current.size());

            cv::Mat detail = current - up;
            cv::resize(detail, detail, gray.size());
            wavelet_detail += detail;

            current = down;
        }

        // Measure the amount of high-frequency content by measuring
        // the standard deviation of the wavelet coefficients
        cv::Scalar mean, stddev;
        cv::meanStdDev(wavelet_detail, mean, stddev);

        return stddev[0];
    }

};

} // namespace blur
