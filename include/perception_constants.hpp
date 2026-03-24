#ifndef PERCEPTION_CONSTANTS_HPP
#define PERCEPTION_CONSTANTS_HPP

namespace perception_constants {

// 图像配置
constexpr int EXPECTED_IMAGE_WIDTH = 640;
constexpr int EXPECTED_IMAGE_HEIGHT = 480;

// 时间配置
constexpr double TIME_DELAY_THRESHOLD = 0.25;  // 图像延迟阈值（秒）
constexpr double TIME_LEFT_RIGHT_DELAY = 0.01;  // 左右图像同步阈值（秒）
constexpr int IMAGE_TIMEOUT_SECONDS = 5;  // 图像处理超时（秒）
constexpr int STATUS_PUBLISH_INTERVAL_SECONDS = 10;  // 状态发布间隔（秒）

// 轮询配置
constexpr int POLL_INTERVAL_MS = 2;  // 轮询间隔（毫秒）

// 其他配置
constexpr int OVER_AREA_THRESHOLD = 2500;  // 区域阈值
constexpr const char* MODEL_VERSION = "v1.6.0";

// 错误码定义
enum class ErrorCode : int {
    OK = 150,
    MODEL_LOAD_ERROR = 151,
    NO_IMAGE_TIMEOUT = 160,
    SINGLE_IMAGE_ERROR = 161,
    IMAGE_DELAY_ERROR = 162
};

// 错误描述
inline const char* getErrorDescription(ErrorCode code) {
    switch (code) {
        case ErrorCode::OK:
            return "====perception status [OK]=====";
        case ErrorCode::MODEL_LOAD_ERROR:
            return "====perception model load error=====";
        case ErrorCode::NO_IMAGE_TIMEOUT:
            return "====No image input (timeout)=====";
        case ErrorCode::SINGLE_IMAGE_ERROR:
            return "====pic input error=====";
        case ErrorCode::IMAGE_DELAY_ERROR:
            return "====image has delay=====";
        default:
            return "====unknown error=====";
    }
}

// 双边图像错误描述
inline const char* getBothImageErrorDescription() {
    return "====pic input error (both empty or size mismatch)=====";
}

} // namespace perception_constants

#endif // PERCEPTION_CONSTANTS_HPP
