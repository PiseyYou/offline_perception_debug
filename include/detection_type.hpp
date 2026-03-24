#ifndef DETECTION_TYPE_HPP
#define DETECTION_TYPE_HPP

#include <opencv2/core/core.hpp>
#include "perception_common.h"

namespace cv {
    template<>
    class DataType<Detection> {
    public:
        typedef Detection value_type;
        typedef value_type work_type;
        typedef value_type channel_type;
        typedef value_type vec_type;
        enum {
            generic_type = 0,
            depth = 0,
            channels = 1,
            fmt = 0,
            type = CV_MAKETYPE(0, 1)
        };
    };
}

#endif // DETECTION_TYPE_HPP
