cmake_minimum_required(VERSION 3.4.1)


# set(CMAKE_CXX_FLAGS "-frtti -fexceptions -std=c++11 -fopenmp")

add_library(face SHARED
            src/main/cpp/face.cpp
            src/main/cpp/utils.cpp
            src/main/cpp/VideoFaceDetector.cpp)

# 添加系统库
find_library(ANDROID_LOG log)

# 添加 ncnn 库
add_library(ncnn STATIC IMPORTED)
set_target_properties(ncnn PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/libs/${ANDROID_ABI}/libncnn.a)


# 添加 opencv 库
add_library(opencv_core STATIC IMPORTED)
set_target_properties(opencv_core PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/libs/${ANDROID_ABI}/libopencv_core.a)
add_library(opencv_imgproc STATIC IMPORTED)
set_target_properties(opencv_imgproc PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/libs/${ANDROID_ABI}/libopencv_imgproc.a)
add_library(opencv_objdetect STATIC IMPORTED)
set_target_properties(opencv_objdetect PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/libs/${ANDROID_ABI}/libopencv_objdetect.a)
add_library(opencv_imgcodecs STATIC IMPORTED)
set_target_properties(opencv_imgcodecs PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/libs/${ANDROID_ABI}/libopencv_imgcodecs.a)


## 添加 opencv 依赖第三方
#add_library(tbb STATIC IMPORTED)
#set_target_properties(tbb PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/libs/${ANDROID_ABI}/libtbb.a)
#add_library(tegra_hal STATIC IMPORTED)
#set_target_properties(tegra_hal PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/libs/${ANDROID_ABI}/libtegra_hal.a)


# 添加 dlib 库

#add_library(android_dlib SHARED IMPORTED)
#set_target_properties(android_dlib PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/libs/${ANDROID_ABI}/libandroid_dlib.so)


# 添加 本地库 头文件
include_directories(${CMAKE_SOURCE_DIR}/include/)

target_link_libraries( face ncnn opencv_objdetect opencv_imgcodecs opencv_imgproc opencv_core ${ANDROID_LOG} z)