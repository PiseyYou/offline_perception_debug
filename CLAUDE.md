# Offline Perception Debug Project Guide

## Project Context
An offline perception debugging tool that processes stereo image pairs to generate point clouds and 3D views. It's a non-ROS2 version of a ROS2 perception node, focusing on OpenCV and PCL.

## Build and Run
- **Build**: `./build.sh` (use `./build.sh clean` for a fresh build)
- **Run**: `./run.sh` (processes images from `data/input/` to `data/output/`)
- **Test**: `./test_demo.sh` or `./verify.sh`

## Code Style & Standards
- **Language**: Modern C++ (C++17/20)
- **Dependencies**: OpenCV 4.x, PCL 1.10+, Eigen3
- **Concurrency**: Use `nproc` for parallel builds in scripts
- **Logging**: Use `std::cout` and `std::cerr` for output (replacing ROS2 logging)
- **Naming**: Snake case for files and local variables, PascalCase for classes

## Critical Workflows
- **Integration**: When moving code from ROS2, remove `<rclcpp/rclcpp.hpp>`, replace `RCLCPP_INFO` with `std::cout`, and replace `sensor_msgs` with `cv::Mat`.
- **Task Management**: Always use `planning-with-files` skill for non-trivial changes.
- **Brainstorming**: Use the `brainstorming` skill before any design change.