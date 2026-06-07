cmake_minimum_required(VERSION 3.10)
project(ImageBlurScore)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})

# Build test executable
add_executable(blur_test main.cpp)
target_link_libraries(blur_test ${OpenCV_LIBS})

# Build and register unit tests
add_executable(blur_unit_tests tests.cpp)
target_link_libraries(blur_unit_tests ${OpenCV_LIBS})

# CTest integration
enable_testing()
add_test(NAME unit_tests COMMAND blur_unit_tests)
