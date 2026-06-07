# ImageBlurScore

A C++ header-only library for computing image blur and sharpness scores using OpenCV. Supports multiple algorithms and grid-based scoring.

## Requirements

- C++17 or later
- OpenCV 4.0+

## Integration

Copy `ImageBlurScore.h` into your project and include it:

```cpp
#include "ImageBlurScore.h"

ImageBlurScore blurscorer;
cv::Mat image = cv::imread("image.jpg");

// Single score (default: Laplacian method)
double score = blurscorer(image);

// Specific algorithm
double score = blurscorer(image, BlurMethod::WAVELET);

// Grid of scores (e.g. 4x4 blocks)
cv::Mat grid = blurscorer(image, 4, 4);
cv::Mat grid = blurscorer.computeGrid(image, 4, 4, BlurMethod::FOURIER);
```

Higher scores indicate sharper (less blurred) images.

## Build Test Program

### CMake

```bash
mkdir build && cd build
cmake ..
make
./blur_test path/to/image.jpg
```

### g++ (direct)

```bash
g++ -std=c++17 main.cpp -o blur_test `pkg-config --cflags --libs opencv4`
./blur_test path/to/image.jpg
```

## Algorithms

| Method | Enum | Description |
|--------|------|-------------|
| Laplacian | `BlurMethod::LAPLACIAN` | Variance of Laplacian operator |
| Fourier | `BlurMethod::FOURIER` | Log-magnitude variance in frequency domain |
| Gradient | `BlurMethod::GRADIENT_MAGNITUDE` | Sobel gradient magnitude standard deviation |
| Contrast | `BlurMethod::CONTRAST` | Grayscale intensity standard deviation |
| Phase Correlation | `BlurMethod::PHASE_CORRELATION` | Peak height of phase correlation response |
| Wavelet | `BlurMethod::WAVELET` | Multi-level Laplacian pyramid detail coefficients |

## API

```cpp
// Single score
double operator()(const cv::Mat& image, BlurMethod = LAPLACIAN) const;
double compute(const cv::Mat& image, BlurMethod = LAPLACIAN) const;

// Grid of scores (returns CV_64F matrix, size gridRows x gridCols)
cv::Mat operator()(const cv::Mat& image, int gridRows, int gridCols, BlurMethod = LAPLACIAN) const;
cv::Mat computeGrid(const cv::Mat& image, int gridRows, int gridCols, BlurMethod = LAPLACIAN) const;
```

## License

MIT. See [LICENSE](LICENSE).
