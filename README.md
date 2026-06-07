# ImageBlurScore

A C++ header-only library for computing image blur and sharpness scores using OpenCV. Supports 13 algorithms and grid-based scoring.

[![CI](https://github.com/csv610/ImageBlurScore/actions/workflows/ci.yml/badge.svg)](https://github.com/csv610/ImageBlurScore/actions/workflows/ci.yml)

## Requirements

- C++17 or later
- OpenCV 4.0+

## Integration

Copy `ImageBlurScore.h` into your project and include it:

```cpp
#include "ImageBlurScore.h"

blur::ImageBlurScore blurscorer;
cv::Mat image = cv::imread("image.jpg");

// Single score (default: Laplacian method)
double score = blurscorer(image);

// Specific algorithm
double score = blurscorer(image, blur::BlurMethod::WAVELET);

// Grid of scores (e.g. 4x4 blocks)
cv::Mat grid = blurscorer(image, 4, 4);
cv::Mat grid = blurscorer.computeGrid(image, 4, 4, blur::BlurMethod::FOURIER);
```

Higher scores indicate sharper (less blurred) images.

## Build

### CMake (recommended)

```bash
cmake -B build
cmake --build build
./build/blur_test path/to/image.jpg
```

### g++ (direct)

```bash
g++ -std=c++17 main.cpp -o blur_test `pkg-config --cflags --libs opencv4`
./blur_test path/to/image.jpg
```

## API

```cpp
// Single score
double operator()(const cv::Mat& image, blur::BlurMethod = blur::BlurMethod::LAPLACIAN) const;
double compute(const cv::Mat& image, blur::BlurMethod = blur::BlurMethod::LAPLACIAN) const;

// Grid of scores (returns CV_64F matrix, size gridRows x gridCols)
cv::Mat operator()(const cv::Mat& image, int gridRows, int gridCols, blur::BlurMethod = blur::BlurMethod::LAPLACIAN) const;
cv::Mat computeGrid(const cv::Mat& image, int gridRows, int gridCols, blur::BlurMethod = blur::BlurMethod::LAPLACIAN) const;
```

## Algorithms

| Method | Enum | Description |
|--------|------|-------------|
| Laplacian | `blur::BlurMethod::LAPLACIAN` | Variance of Laplacian operator |
| Fourier | `blur::BlurMethod::FOURIER` | Log-magnitude variance in frequency domain |
| Gradient | `blur::BlurMethod::GRADIENT_MAGNITUDE` | Sobel gradient magnitude standard deviation |
| Contrast | `blur::BlurMethod::CONTRAST` | Grayscale intensity standard deviation |
| Phase Correlation | `blur::BlurMethod::PHASE_CORRELATION` | Peak height of phase correlation response |
| Wavelet | `blur::BlurMethod::WAVELET` | Multi-level Laplacian pyramid detail coefficients |
| Brenner | `blur::BlurMethod::BRENNER` | Squared horizontal difference (step 2) standard deviation |
| Tenengrad | `blur::BlurMethod::TENENGRAD` | Sum of squared Sobel responses standard deviation |
| SML | `blur::BlurMethod::SML` | Sum of Modified Laplacian — absolute separated second derivatives |
| EOL | `blur::BlurMethod::EOL` | Energy of Laplacian — squared Laplacian values |
| Scharr | `blur::BlurMethod::SCHARR` | Scharr gradient magnitude (rotationally symmetric) |
| Histogram Entropy | `blur::BlurMethod::HISTOGRAM_ENTROPY` | Shannon entropy of the grayscale histogram |
| DCT Energy Ratio | `blur::BlurMethod::DCT_ENERGY_RATIO` | Ratio of high-frequency to total energy in 8x8 DCT blocks |

## Tests

```bash
cmake --build build
ctest --test-dir build
```

Runs 29 tests covering every algorithm on synthetic gradient and uniform images, plus grid dimension validation.

## License

MIT. See [LICENSE](LICENSE).
