//
// Created by gjs on 17-10-17.
//

#ifndef MYFACE_UTILS_H
#define MYFACE_UTILS_H

#include <android/log.h>

#define LOG_TAG "Face"
#define LOGV(...) __android_log_print(ANDROID_LOG_VERBOSE, LOG_TAG, __VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG  , LOG_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO   , LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN   , LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR  , LOG_TAG, __VA_ARGS__)
#define LOG(...) __android_log_print(ANDROID_LOG_ERROR   , LOG_TAG, __VA_ARGS__)

extern "C"
void YUVtoBGRA(int8_t *yuv, jint width, jint height, int32_t *rgb);
//void YUVtoARBG(int8_t *yuv, jint width, jint height, int32_t *rgb);

#endif //MYFACE_UTILS_H
