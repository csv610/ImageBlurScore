# ImageBlurScore

A C++ header-only library for computing image blur scores using OpenCV. This library provides multiple algorithms to assess the amount of blur in an image.

## Features

- **Laplacian-based blur detection**: Uses variance of Laplacian operator to detect blur
- **Fourier-based analysis**: Analyzes blur using frequency domain (DFT)
- **Gradient magnitude scoring**: Measures image sharpness through gradient analysis
- **Contrast scoring**: Evaluates image contrast levels
- **Wavelet analysis**: Uses multi-level wavelet decomposition to detect blur
- **Dark channel method**: Implements dark channel-based analysis
- **Phase correlation**: Analyzes blur using phase correlation techniques

## Installation

### Requirements

- C++11 or higher
- OpenCV 4.0+

### Usage

Simply include the header file in your project:

```cpp
#include "ImageBlurScore.h"

// Create an instance
ImageBlurScore blurscorer;

// Load an image with OpenCV
cv::Mat image = cv::imread("image.jpg");

// Compute blur score
double score = blurscorer(image);
std::cout << "Blur Score: " << score << std::endl;
```

Higher scores generally indicate sharper (less blurred) images.

## Building the Test Program

### Using CMake (Recommended):

```bash
mkdir build && cd build
cmake ..
make
./blur_test path/to/image.jpg
```

### Using g++ directly:

```bash
g++ -std=c++11 main.cpp -o blur_test `pkg-config --cflags --libs opencv4`
./blur_test path/to/image.jpg
```

## Methods

The library provides several blur detection algorithms:

- `operator()()` - Uses the default Laplacian-based method
- `computeBlurScore()` - Laplacian variance method
- `computeFourierBlurScore()` - Frequency domain analysis
- `computeGradientMagnitudeScore()` - Gradient-based sharpness
- `computeContrastScore()` - Contrast measurement
- `computeWaveletBlurScore()` - Multi-level wavelet analysis
- `computePhaseCorrelationScore()` - Phase correlation method
- `computeDarkChannel()` - Dark channel computation

## License

[Add your chosen license here]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
