#ifndef _BASE_PERCEPTION_H_
#define _BASE_PERCEPTION_H_

#include <vector>
#include <string>
#include <memory>
#include <opencv2/opencv.hpp>

#define YJ_MODEL_OUTPUT_WIDTH 640
#define YJ_MODEL_OUTPUE_HEIGHT 480

const uint8_t yj_seg_bgr_putpalette[19] = {
        119, 119, 119,     200, 0, 0,      102, 255,  102,      0,  89, 118,     0, 255, 255,     0, 0, 255};

// Bounding box definition
struct Bbox {
    float xmin;
    float ymin;
    float xmax;
    float ymax;

    Bbox() : xmin(0), ymin(0), xmax(0), ymax(0) {}
    Bbox(float x1, float y1, float x2, float y2) : xmin(x1), ymin(y1), xmax(x2), ymax(y2) {}
};

// Detection result structure
struct Detection {
    int id;
    float score;
    Bbox bbox;
    float area;
    int skip;
    const char *class_name = nullptr;

    Detection() : id(0), score(0), area(0), skip(0) {}
    Detection(int id, float score, Bbox bbox) : id(id), score(score), bbox(bbox), area(0), skip(0) {}
    Detection(int id, float score, Bbox bbox, const char *name) : id(id), score(score), bbox(bbox), area(0), skip(0), class_name(name) {}

    friend bool operator>(const Detection &lhs, const Detection &rhs) {
        return (lhs.score > rhs.score);
    }
};

// Classification result structure
struct Classification {
    int id;
    float score;
    const char *class_name;

    Classification() : id(0), score(0), class_name(nullptr) {}
    Classification(int id, float score, const char *name) : id(id), score(score), class_name(name) {}

    friend bool operator>(const Classification &lhs, const Classification &rhs) {
        return (lhs.score > rhs.score);
    }
};

// Parsing result for segmentation
struct Parsing {
    std::vector<int8_t> seg;
    int32_t num_classes = 0;
    int32_t width = 0;
    int32_t height = 0;
};

// Horizon-specific Perception wrapper for backward compatibility
struct Perception {
    std::vector<Detection> det;
    std::vector<Classification> cls;
    Parsing seg;
    float h_base = 1;
    float w_base = 1;

    enum Type {
        DET = (1 << 0),
        CLS = (1 << 1),
        SEG = (1 << 2),
        MASK = (1 << 3),
    } type;
};

#endif // _BASE_PERCEPTION_H_