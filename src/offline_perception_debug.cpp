/**
 * @file offline_perception_debug.cpp
 * @brief 离线感知调试工具 - 按照ROS2 mode (0-8) 逻辑实现
 */

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <pcl/io/pcd_io.h>
#include <string>
#include <vector>

#include "unified_perception_processor.h"
#include "offline_utils.hpp"

namespace fs = std::filesystem;
using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    cout << "==================================================" << endl;
    cout << "  Offline Perception Debug Tool (Unified)         " << endl;
    cout << "==================================================" << endl;

    offline_perception::Config config;
    // TODO: Load from config.yaml
    config.infer_mode = 7;
    config.erode_pixel = 205;
    config.model_dir = "../models/";

    string input_dir = "/home/youfeng/debug/03/06/bug/userdata/rosbag_record/rosbag_LK-MR6P1US000107_navigation_202603061404/stereo_output_rosbag_LK-MR6P1US000107_navigation_202603061404_0_filtered_20260306_1405_to_20260306_1408/images/no_point/";

    config.finalPicDir = input_dir + "/output_pics/";
    config.finalPcdDir = input_dir + "/output_pcd/";

    fs::create_directories(config.finalPicDir);
    fs::create_directories(config.finalPcdDir);

    offline_perception::UnifiedPerceptionProcessor processor(config);
    if (!processor.init())
    {
        cerr << "[Error] Failed to initialize perception processor" << endl;
        return -1;
    }

    vector<string> image_files;
    for (const auto& entry : fs::directory_iterator(input_dir))
    {
        if (!entry.is_regular_file()) continue;
        string ext = entry.path().extension().string();
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp")
        {
            image_files.push_back(entry.path().string());
        }
    }
    sort(image_files.begin(), image_files.end());

    cout << "Found " << image_files.size() << " images." << endl;

    for (size_t i = 0; i < image_files.size(); i++)
    {
        const string& image_path = image_files[i];
        Mat full_img = imread(image_path);
        if (full_img.empty()) continue;

        Mat left_img = full_img.clone();
        Mat right_img;

        if (full_img.cols == 1280 && full_img.rows == 480)
        {
            left_img = full_img(Rect(0, 0, 640, 480)).clone();
            right_img = full_img(Rect(640, 0, 640, 480)).clone();
        }

        string base_name = fs::path(image_path).stem().string();
        cout << "Processing [" << (i+1) << "/" << image_files.size() << "]: " << base_name << endl;

        auto result = processor.process(left_img, right_img, i, base_name);

        // Save visualization
        if (!result.visualization.empty())
        {
            string out_path = config.finalPicDir + base_name + "_vis.jpg";
            imwrite(out_path, result.visualization);
        }

        // Save point cloud
        if (!result.point_cloud.empty())
        {
            string pcd_path = config.finalPcdDir + base_name + ".pcd";
            savePcdfile_with_rgb_label(result.point_cloud, pcd_path.substr(0, pcd_path.length()-4));
        }
    }

    cout << "Done. Results in " << config.finalPicDir << endl;
    return 0;
}
