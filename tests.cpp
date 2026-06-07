#include "ImageBlurScore.h"
#include <cassert>
#include <cmath>
#include <iostream>

static cv::Mat createTestImage(int rows, int cols) {
    cv::Mat img(rows, cols, CV_8UC3);
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            img.at<cv::Vec3b>(r, c) = cv::Vec3b(
                (uchar)(r * 255 / rows),
                (uchar)(c * 255 / cols),
                (uchar)((r + c) * 255 / (rows + cols))
            );
        }
    }
    return img;
}

static cv::Mat createUniformImage(int rows, int cols) {
    return cv::Mat(rows, cols, CV_8UC3, cv::Scalar(128, 128, 128));
}

#define TEST(name) \
    do { \
        std::cout << "  " << name << "... "; \
        std::cout.flush(); \
    } while (0)

#define PASS() std::cout << "PASS" << std::endl

int main() {
    blur::ImageBlurScore scorer;

    // --- Single image tests ---
    std::cout << "[Single-score tests]" << std::endl;

    {
        TEST("Laplacian on gradient image");
        cv::Mat img = createTestImage(100, 100);
        double s = scorer(img, blur::BlurMethod::LAPLACIAN);
        assert(std::isfinite(s));
        assert(s >= 0);
        PASS();
    }

    {
        TEST("Laplacian on uniform image (score should be ~0)");
        cv::Mat img = createUniformImage(100, 100);
        double s = scorer(img, blur::BlurMethod::LAPLACIAN);
        assert(s < 1.0);
        PASS();
    }

    {
        TEST("Fourier on gradient image");
        cv::Mat img = createTestImage(100, 100);
        double s = scorer(img, blur::BlurMethod::FOURIER);
        assert(std::isfinite(s));
        assert(s >= 0);
        PASS();
    }

    {
        TEST("Fourier on uniform image");
        cv::Mat img = createUniformImage(100, 100);
        double s = scorer(img, blur::BlurMethod::FOURIER);
        assert(std::isfinite(s));
        PASS();
    }

    {
        TEST("Gradient magnitude on gradient image");
        cv::Mat img = createTestImage(100, 100);
        double s = scorer(img, blur::BlurMethod::GRADIENT_MAGNITUDE);
        assert(std::isfinite(s));
        assert(s >= 0);
        PASS();
    }

    {
        TEST("Gradient magnitude on uniform image");
        cv::Mat img = createUniformImage(100, 100);
        double s = scorer(img, blur::BlurMethod::GRADIENT_MAGNITUDE);
        assert(s < 0.5);
        PASS();
    }

    {
        TEST("Contrast on gradient image");
        cv::Mat img = createTestImage(100, 100);
        double s = scorer(img, blur::BlurMethod::CONTRAST);
        assert(std::isfinite(s));
        assert(s > 0);
        PASS();
    }

    {
        TEST("Contrast on uniform image");
        cv::Mat img = createUniformImage(100, 100);
        double s = scorer(img, blur::BlurMethod::CONTRAST);
        assert(s < 0.5);
        PASS();
    }

    {
        TEST("Phase correlation on gradient image");
        cv::Mat img = createTestImage(100, 100);
        double s = scorer(img, blur::BlurMethod::PHASE_CORRELATION);
        assert(std::isfinite(s));
        PASS();
    }

    {
        TEST("Wavelet on gradient image");
        cv::Mat img = createTestImage(100, 100);
        double s = scorer(img, blur::BlurMethod::WAVELET);
        assert(std::isfinite(s));
        assert(s >= 0);
        PASS();
    }

    {
        TEST("Brenner on gradient image");
        cv::Mat img = createTestImage(100, 100);
        double s = scorer(img, blur::BlurMethod::BRENNER);
        assert(std::isfinite(s));
        assert(s >= 0);
        PASS();
    }

    {
        TEST("Brenner on uniform image (score should be ~0)");
        cv::Mat img = createUniformImage(100, 100);
        double s = scorer(img, blur::BlurMethod::BRENNER);
        assert(s < 1.0);
        PASS();
    }

    {
        TEST("Tenengrad on gradient image");
        cv::Mat img = createTestImage(100, 100);
        double s = scorer(img, blur::BlurMethod::TENENGRAD);
        assert(std::isfinite(s));
        assert(s >= 0);
        PASS();
    }

    {
        TEST("Tenengrad on uniform image (score should be ~0)");
        cv::Mat img = createUniformImage(100, 100);
        double s = scorer(img, blur::BlurMethod::TENENGRAD);
        assert(s < 1.0);
        PASS();
    }

    // --- Grid tests ---
    std::cout << "[Grid tests]" << std::endl;

    {
        TEST("4x4 grid returns correct shape");
        cv::Mat img = createTestImage(200, 200);
        cv::Mat g = scorer(img, 4, 4);
        assert(g.rows == 4);
        assert(g.cols == 4);
        assert(g.type() == CV_64F);
        PASS();
    }

    {
        TEST("1x1 grid equals single score");
        cv::Mat img = createTestImage(100, 100);
        double single = scorer(img);
        cv::Mat g = scorer(img, 1, 1);
        assert(std::abs(g.at<double>(0, 0) - single) < 0.001);
        PASS();
    }

    {
        TEST("All grid scores are finite and non-negative");
        cv::Mat img = createTestImage(150, 200);
        cv::Mat g = scorer(img, 3, 5);
        for (int r = 0; r < g.rows; r++) {
            for (int c = 0; c < g.cols; c++) {
                double v = g.at<double>(r, c);
                assert(std::isfinite(v));
                assert(v >= 0);
            }
        }
        PASS();
    }

    {
        TEST("computeGrid throws on invalid grid dimensions");
        cv::Mat img = createTestImage(100, 100);
        bool caught = false;
        try {
            scorer.computeGrid(img, 0, 4);
        } catch (const std::invalid_argument&) {
            caught = true;
        }
        assert(caught);
        caught = false;
        try {
            scorer.computeGrid(img, 4, -1);
        } catch (const std::invalid_argument&) {
            caught = true;
        }
        assert(caught);
        PASS();
    }

    {
        TEST("Uneven division (100x100 into 3x3)");
        cv::Mat img = createTestImage(100, 100);
        cv::Mat g = scorer(img, 3, 3);
        assert(g.rows == 3);
        assert(g.cols == 3);
        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < 3; c++) {
                assert(std::isfinite(g.at<double>(r, c)));
            }
        }
        PASS();
    }

    std::cout << "All tests passed!" << std::endl;
    return 0;
}
