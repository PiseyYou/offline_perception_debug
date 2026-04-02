/**
 * @file dsg_mono_perception.cpp
 * @brief 单目图片DSG感知调试工具
 * @description 输入640x480单目左目图片，裁剪640x432，resize到640x384送入DSG模型，
 *              输出三张640x432图片水平拼接：原图 | 分割着色图 | 融合图
 */

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "dsg_perception.h"

namespace fs = std::filesystem;
using namespace cv;
using namespace std;

// 颜色查找表 (BGR顺序)
static const std::array<cv::Vec3b, 256> buildColorLUT()
{
  std::array<cv::Vec3b, 256> lut;
  lut.fill(cv::Vec3b(0, 0, 0));
  lut[0] = cv::Vec3b(0, 0, 0);
  lut[1] = cv::Vec3b(200, 0, 0);
  lut[2] = cv::Vec3b(102, 255, 100);
  lut[3] = cv::Vec3b(0, 89, 118);
  lut[4] = cv::Vec3b(0, 255, 255);
  lut[5] = cv::Vec3b(0, 0, 255);
  lut[6] = cv::Vec3b(0, 165, 255);
  lut[7] = cv::Vec3b(147, 20, 255);
  lut[8] = cv::Vec3b(255, 255, 0);
  lut[9] = cv::Vec3b(48, 130, 245);
  lut[10] = cv::Vec3b(128, 64, 0);
  lut[11] = cv::Vec3b(34, 139, 34);
  lut[12] = cv::Vec3b(203, 192, 255);
  lut[13] = cv::Vec3b(226, 43, 138);
  return lut;
}

static void applyColorLUT(const cv::Mat &img_lab, cv::Mat &parsing_img)
{
  static const auto lut = buildColorLUT();
  for (int i = 0; i < img_lab.rows; ++i)
  {
    const uchar *row_ptr = img_lab.ptr<uchar>(i);
    cv::Vec3b *out_ptr = parsing_img.ptr<cv::Vec3b>(i);
    for (int j = 0; j < img_lab.cols; ++j)
      out_ptr[j] = lut[row_ptr[j]];
  }
}

// 生成着色图和融合图，返回纯着色图
static cv::Mat drawResult(cv::Mat &img_src, cv::Mat &img_lab, cv::Mat &img_seg_show)
{
  cv::Mat parsing_img(img_lab.size(), CV_8UC3);
  applyColorLUT(img_lab, parsing_img);

  if (parsing_img.size() != img_src.size())
    cv::resize(parsing_img, parsing_img, img_src.size(), 0, 0, cv::INTER_NEAREST);

  // 消除底部 label==2 绿色伪影带
  {
    int artifact_start = img_lab.rows;
    for (int i = img_lab.rows - 1; i >= 0; --i)
    {
      const uchar *row_ptr = img_lab.ptr<uchar>(i);
      int cnt2 = 0;
      for (int j = 0; j < img_lab.cols; ++j)
        if (row_ptr[j] == 2) ++cnt2;
      if (static_cast<float>(cnt2) / img_lab.cols > 0.95f)
        artifact_start = i;
      else
        break;
    }
    if (artifact_start < img_lab.rows)
    {
      float row_scale = static_cast<float>(parsing_img.rows) / img_lab.rows;
      int ps_start = static_cast<int>(artifact_start * row_scale);
      img_src(cv::Rect(0, ps_start, img_src.cols, img_src.rows - ps_start))
          .copyTo(parsing_img(cv::Rect(0, ps_start, parsing_img.cols, parsing_img.rows - ps_start)));
    }
  }

  cv::addWeighted(img_src, 0.6f, parsing_img, 0.4f, 0.0, img_seg_show);

  return parsing_img;
}

int main(int argc, char **argv)
{
  cout << "=================================================="  << endl;
  cout << "  DSG Mono Perception Debug Tool                 "  << endl;
  cout << "=================================================="  << endl;

  // 配置（优先从环境变量读取）
  const char* env_input  = std::getenv("MONO_INPUT_DIR");
  const char* env_output = std::getenv("MONO_OUTPUT_DIR");
  const char* env_model  = std::getenv("MONO_MODEL_PATH");
  string input_dir = env_input  ? env_input  : (argc > 1 ? argv[1] : "/home/youfeng/debug/03/0330/rosbag_LK-MR6P1US000111_navigation_202603282229/left_image_pcl_202603282229/images/");
  string dsg_model  = env_model  ? env_model  : (argc > 3 ? argv[3] : "../models/dsg_20260211_640x384.bin");
  string output_dir = env_output ? env_output : (argc > 2 ? argv[2] : input_dir + "/dsg_mono_432/");

  fs::create_directories(output_dir);

  cout << "Input dir:  " << input_dir  << endl;
  cout << "Output dir: " << output_dir << endl;
  cout << "Model:      " << dsg_model  << endl;

  // 初始化DSG模型
  dsg_perception dsgPerception;
  dsgPerception.perception_init(dsg_model.c_str());
  cout << "[✓] DSG model initialized" << endl;

  // 扫描输入图片
  vector<string> image_files;
  try
  {
    for (const auto &entry : fs::directory_iterator(input_dir))
    {
      if (!entry.is_regular_file()) continue;
      string ext = entry.path().extension().string();
      if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp")
        image_files.push_back(entry.path().string());
    }
  }
  catch (const fs::filesystem_error &e)
  {
    cerr << "[Error] Cannot access input directory: " << e.what() << endl;
    return -1;
  }

  if (image_files.empty())
  {
    cout << "[Warning] No image files found in: " << input_dir << endl;
    return 0;
  }

  sort(image_files.begin(), image_files.end());
  cout << "Found " << image_files.size() << " image(s)." << endl;

  for (size_t i = 0; i < image_files.size(); ++i)
  {
    const string &image_path = image_files[i];
    string base_name = fs::path(image_path).stem().string();

    Mat full_img = imread(image_path);
    if (full_img.empty())
    {
      cerr << "[Error] Cannot read: " << image_path << endl;
      continue;
    }

    // 若为双目拼接图(1280x480)，取左半部分
    Mat left_img;
    if (full_img.cols == 1280 && full_img.rows == 480)
      left_img = full_img(Rect(0, 0, 640, 480)).clone();
    else
      left_img = full_img.clone();

    if (left_img.cols != 640 || left_img.rows != 480)
    {
      cerr << "[Skip] Expected 640x480 input, got "
           << left_img.cols << "x" << left_img.rows
           << " : " << image_path << endl;
      continue;
    }

    cout << "\n[" << (i+1) << "/" << image_files.size() << "] " << base_name << endl;

    auto t0 = chrono::high_resolution_clock::now();

    // Step 1: 裁剪 640x480 -> 640x432
    Mat cropped432 = left_img(Rect(0, 0, 640, 432)).clone();

    // Step 2: resize 640x432 -> 640x384 送入模型
    Mat resized384;
    cv::resize(cropped432, resized384, Size(640, 384), 0, 0, INTER_LINEAR);

    // Step 3: DSG模型推理，输出640x384标签图
    cv::Mat dst_label384(384, 640, CV_8UC1, cv::Scalar(1));
    dsgPerception.process_infer_match(resized384, dst_label384);

    auto t1 = chrono::high_resolution_clock::now();
    double infer_ms = chrono::duration<double, milli>(t1 - t0).count();
    cout << "  Inference: " << fixed << setprecision(1) << infer_ms << " ms" << endl;

    // Step 4: 标签图 640x384 -> resize回 640x432
    cv::Mat label432;
    cv::resize(dst_label384, label432, Size(640, 432), 0, 0, INTER_NEAREST);

    // Step 5: 原图 640x384 resize成 640x432 用于展示
    Mat origin432;
    cv::resize(resized384, origin432, Size(640, 432), 0, 0, INTER_LINEAR);

    // Step 6: 生成着色图和融合图 (均为640x432)
    Mat img_seg_show;
    Mat pure_seg_mat = drawResult(origin432, label432, img_seg_show);

    // Step 7: 水平拼接三张640x432图片 -> 1920x432
    Mat result;
    cv::hconcat(origin432, pure_seg_mat, result);
    cv::hconcat(result, img_seg_show, result);

    // 保存
    string out_path = output_dir + base_name + "_dsg_mono.jpg";
    imwrite(out_path, result);
    cout << "  Saved: " << out_path << endl;
  }

  cout << "\n=================================================="  << endl;
  cout << "  All done! Output: " << output_dir                     << endl;
  cout << "=================================================="  << endl;

  return 0;
}
