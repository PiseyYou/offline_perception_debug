#if !defined(ARUCO_DETECTOR_HPP)
#define ARUCO_DETECTOR_HPP

#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <array>
#include <opencv2/calib3d.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/aruco/dictionary.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>  
#include <vector>
#include <mutex>
#include <atomic>
#include <rclcpp/rclcpp.hpp>

struct DetectionVariables
{
    std::vector<int> detectedIds;
    std::vector<std::vector<cv::Point2f>> detectedCorners;
    std::vector<std::vector<cv::Point2f>> rejectedCorners;

    void clear()
    {
        detectedIds.clear();
        detectedCorners.clear();
        rejectedCorners.clear();
    }
};

class ArucoDetector
{
private:
    // DetectionVariables variable_;
    // Aruco Tag Parameters
    cv::Ptr<cv::aruco::Dictionary> aruco_dict_;
    cv::Ptr<cv::aruco::GridBoard> board_;
    cv::Ptr<cv::aruco::DetectorParameters> detectorParam_;
    cv::Mat bitsMat0_, bitsMat1_;
    float marker_length_m_;

public:
    std::vector<int> tag_id_; // 充电桩二维码的id

    float tag_size_;
    float tag_padding_;

    cv::Mat Pl_;
    cv::Mat Pr_;
    cv::Mat Kl_;
    cv::Mat Kr_;
    cv::Mat Dl_;
    cv::Mat Dr_;

    std::atomic<bool> is_camera_param_init_;

    void createBoard();
    void setIntrinsics(const cv::Mat &Pl, const cv::Mat &Pr);
    void displayIntrinsics() const;
    int detectAndRefineMarkers(cv::Mat &src,
                               DetectionVariables &det_variables, std::vector<cv::Point2f> &outPoly, bool &is_get_aruco_area_success);
    bool estimatePose(cv::Mat &rvec,
                      cv::Mat &tvec, DetectionVariables &det_variables);
    void drawDetection(cv::Mat &src,
                       cv::Mat &rvec,
                       cv::Mat &tvec, DetectionVariables &det_variables);

    ArucoDetector();
    ~ArucoDetector();
};

#endif // ARUCO_DETECTOR_HPP