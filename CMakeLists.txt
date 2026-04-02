cmake_minimum_required(VERSION 3.10)
project(offline_perception_debug VERSION 1.0.0)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 设置编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

## 默认为Release模式
#if(NOT CMAKE_BUILD_TYPE)
##    set(CMAKE_BUILD_TYPE Release)
#    set(CMAKE_BUILD_TYPE Debug)
#endif()

set(DEPS_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../deps_gcc11.3/x86)
message(STATUS "DEPS_ROOT: ${DEPS_ROOT}")

# 查找OpenCV
find_package(OpenCV 4 REQUIRED)
if(OpenCV_FOUND)
    message(STATUS "✓ Found OpenCV version: ${OpenCV_VERSION}")
    message(STATUS "  Include dirs: ${OpenCV_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "✗ OpenCV not found!")
endif()

# 查找PCL
find_package(PCL 1.10 REQUIRED COMPONENTS common io filters)
if(PCL_FOUND)
    message(STATUS "✓ Found PCL version: ${PCL_VERSION}")
    message(STATUS "  Include dirs: ${PCL_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "✗ PCL not found!")
endif()

# 查找Eigen3
find_package(Eigen3 REQUIRED)
if(Eigen3_FOUND)
    message(STATUS "✓ Found Eigen3 version: ${Eigen3_VERSION}")
else()
    message(FATAL_ERROR "✗ Eigen3 not found!")
endif()

# 包含目录
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${DEPS_ROOT}/dnn_x86/include
)

# 链接目录
link_directories(
    ${PCL_LIBRARY_DIRS}
    ${DEPS_ROOT}/dnn_x86/lib
)

SET(LINK_libs dnn hbdk_sim_x86 z dl rt pthread)

# 添加PCL定义
add_definitions(${PCL_DEFINITIONS})

# 源文件
set(SOURCES
    src/offline_perception_debug.cpp
    src/offline_utils.cpp
    src/stereo_multi_match.cpp
    src/multi_sub_perception.cpp
    src/seg_perception.cpp
    src/det_perception.cpp
    src/cdt_perception.cpp
    src/qr_cs_perception.cpp
    src/cls_perception.cpp
    src/multiscale_filter.cpp
    src/stereo_point_cloud_rgbl.cpp
)

# 可执行文件
add_executable(${PROJECT_NAME} ${SOURCES})

# 链接库
target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}  ${PCL_LIBRARIES} ${LINK_libs} ${JSONCPP_LIBRARIES}
)

# 安装规则
install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION bin
)

# 源文件
set(SOURCES
        src/offline_perception_debug_432.cpp
        src/offline_utils.cpp
        src/stereo_multi_match.cpp
        src/multi_sub_perception.cpp
        src/seg_perception.cpp
        src/det_perception.cpp
        src/cdt_perception.cpp
        src/qr_cs_perception.cpp
        src/cls_perception.cpp
        src/multiscale_filter.cpp
        src/stereo_point_cloud_rgbl.cpp
)

# 可执行文件
add_executable(offline_perception_debug_432 ${SOURCES})

# 链接库
target_link_libraries(offline_perception_debug_432  ${OpenCV_LIBS}  ${PCL_LIBRARIES} ${LINK_libs} ${JSONCPP_LIBRARIES}
)

install(TARGETS offline_perception_debug_432
        RUNTIME DESTINATION bin
)

set(SOURCES
        src/offline_perception_debug_color.cpp
        src/offline_utils.cpp
        src/stereo_multi_match.cpp
        src/multi_sub_perception.cpp
        src/seg_perception.cpp
        src/det_perception.cpp
        src/cdt_perception.cpp
        src/qr_cs_perception.cpp
        src/cls_perception.cpp
        src/multiscale_filter.cpp
        src/stereo_point_cloud_rgbl.cpp
)

# 可执行文件
add_executable(offline_perception_debug_color ${SOURCES})

# 链接库
target_link_libraries(offline_perception_debug_color  ${OpenCV_LIBS}  ${PCL_LIBRARIES} ${LINK_libs} ${JSONCPP_LIBRARIES}
)

install(TARGETS offline_perception_debug_color
        RUNTIME DESTINATION bin
)


set(SOURCES
        src/offline_perception_debug_432_sob.cpp
        src/offline_utils.cpp
        src/stereo_multi_match.cpp
        src/multi_sub_perception.cpp
        src/seg_perception.cpp
        src/det_perception.cpp
        src/cdt_perception.cpp
        src/qr_cs_perception.cpp
        src/cls_perception.cpp
        src/dsg_perception.cpp
        src/multiscale_filter.cpp
        src/stereo_point_cloud_rgbl.cpp
)

# 可执行文件
add_executable(offline_perception_debug_432_sob ${SOURCES})

# 链接库
target_link_libraries(offline_perception_debug_432_sob  ${OpenCV_LIBS}  ${PCL_LIBRARIES} ${LINK_libs} ${JSONCPP_LIBRARIES}
)

install(TARGETS offline_perception_debug_432_sob
        RUNTIME DESTINATION bin
)

set(SOURCES
        src/dsg_mono_perception.cpp
        src/dsg_perception.cpp
)

add_executable(dsg_mono_perception ${SOURCES})

target_link_libraries(dsg_mono_perception ${OpenCV_LIBS} ${PCL_LIBRARIES} ${LINK_libs} ${JSONCPP_LIBRARIES})

install(TARGETS dsg_mono_perception
        RUNTIME DESTINATION bin
)

# 打印配置信息
message(STATUS "")
message(STATUS "========================================")
message(STATUS "  ${PROJECT_NAME} v${PROJECT_VERSION}")
message(STATUS "========================================")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "  OpenCV: ${OpenCV_VERSION}")
message(STATUS "  PCL: ${PCL_VERSION}")
message(STATUS "  Eigen3: ${Eigen3_VERSION}")
message(STATUS "========================================")
message(STATUS "")
