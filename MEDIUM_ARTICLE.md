# ImageBlurScore: A Header-Only C++ Library for Blur Detection

When building computer vision pipelines, you often need a quick way to measure how sharp or blurry an image is — whether for auto-focus, quality control, or filtering training data. **ImageBlurScore** is a lightweight C++17 header-only library that gives you 13 different blur-detection algorithms in a single file.

---

## What It Does

Drop `ImageBlurScore.h` into your project, include it, and you get:

```cpp
#include "ImageBlurScore.h"
#include <opencv2/opencv.hpp>

blur::ImageBlurScore scorer;
cv::Mat img = cv::imread("photo.jpg");

// Single score (default: Laplacian variance)
double score = scorer(img);

// Specific algorithm
double score = scorer(img, blur::BlurMethod::WAVELET);

// 4x4 grid of regional scores
cv::Mat grid = scorer(img, 4, 4, blur::BlurMethod::FOURIER);
```

Higher score = sharper image. Works on color or grayscale `cv::Mat` inputs.

---

## The 13 Algorithms

| Method | Technique | Best For |
|--------|-----------|----------|
| **LAPLACIAN** | Variance of Laplacian | General purpose, fast |
| **FOURIER** | Log-magnitude DFT variance | Frequency-domain analysis |
| **GRADIENT_MAGNITUDE** | Sobel gradient std dev | Edge-aware sharpness |
| **CONTRAST** | Grayscale intensity std dev | Simple contrast check |
| **PHASE_CORRELATION** | Phase correlation peak | Shift-invariant blur |
| **WAVELET** | Multi-level Laplacian pyramid | Multi-scale detail |
| **BRENNER** | Horizontal diff² (step 2) | Fast focus measure |
| **TENENGRAD** | Sum of squared Sobel | Gradient-based focus |
| **SML** | Sum of Modified Laplacian | Second-derivative detail |
| **EOL** | Energy of Laplacian (Laplacian²) | High-frequency energy |
| **SCHARR** | Scharr gradient magnitude | Rotationally symmetric gradient |
| **HISTOGRAM_ENTROPY** | Shannon entropy of histogram | Information content |
| **DCT_ENERGY_RATIO** | High-freq / total DCT energy | JPEG-like frequency analysis |

All methods return `double` — higher = sharper. No normalization; compare scores within the same method.

---

## Grid Scoring

Need regional sharpness (e.g., for autofocus windows or tilt detection)?

```cpp
// 4x4 grid, Laplacian method
cv::Mat grid = scorer.computeGrid(image, 4, 4);
// Returns CV_64F matrix: 4 rows × 4 cols

// With specific method
cv::Mat grid = scorer(image, 3, 5, blur::BlurMethod::TENENGRAD);
```

Throws `std::invalid_argument` if grid dimensions exceed image size or are ≤ 0.

---

## Building & Testing

**CMake (recommended):**
```bash
cmake -B build
cmake --build build
./build/blur_test path/to/image.jpg   # demo CLI
ctest --test-dir build                # 29 unit tests
```

**Direct compile:**
```bash
g++ -std=c++17 main.cpp -o blur_test `pkg-config --cflags --libs opencv4`
```

CI runs on Ubuntu and macOS via GitHub Actions.

---

## Design Decisions Worth Knowing

| Choice | Rationale |
|--------|-----------|
| **Header-only** | Zero build friction; drop into any CMake/msbuild/Bazel project |
| **OpenCV dependency only** | No extra deps; OpenCV is standard in CV stacks |
| **`cv::Mat` by const ref** | Avoids copies; works with `UMat`, `cuda::GpuMat` via `cv::InputArray` |
| **No internal state** | Thread-safe, trivially copyable |
| **Exceptions on bad grid args** | Fail-fast; no silent wrong-sized outputs |

---

## What It Doesn't Do

- **No normalization** — scores aren't comparable across methods
- **No ground-truth calibration** — not a "blur classifier" with thresholds
- **No GPU acceleration** — pure CPU OpenCV (though `cv::cuda` wrappers work)
- **No ROI API** — grid divides evenly; last row/col absorbs remainder
- **No streaming/SIMD** — processes one frame at a time

---

## When to Use It

- Embedding in a larger C++ CV pipeline (robotics, inspection, mobile)
- Quick prototyping — no build system changes needed
- Comparing multiple focus measures on the same frames
- Regional sharpness maps for autofocus or stitching

## When to Look Elsewhere

- Need calibrated blur *classification* (blur/no-blur) — use a trained model
- Processing video at 60+ FPS — consider OpenCV's `quality::QualityBRISQUE` or GPU implementations
- Python-first workflow — `opencv-python` + `scikit-image` has similar measures

---

## License

MIT. [GitHub repo](https://github.com/csv610/ImageBlurScore) — issues and PRs welcome.