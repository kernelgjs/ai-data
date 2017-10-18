//
// Created by gjs on 17-10-16.
//
#include <jni.h>

#include <android/log.h>

#include <vector>
#include <unistd.h>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "VideoFaceDetector.h"

#include "ncnn/net.h"
#include "ncnn/mat.h"

#include "utils.h"

using namespace std;

VideoFaceDetector face_detector;
ncnn::Net net;

extern "C"
void pMat(cv::Mat &img){
    LOGI("pix:start");

    for( size_t nrow = 0; nrow < 3/*img.rows*/; nrow++)
    {
        for(size_t ncol = 0; ncol < 3/*img.cols*/; ncol++)
        {
            cv::Vec4b pix = img.at<cv::Vec4b>(nrow,ncol);//用Vec3b也行
            LOGI("pix:%d\t%d\t%d\t%d ", pix.val[0], pix.val[1], pix.val[2], pix.val[3] );
        }
    }
}


extern "C"
JNIEXPORT void JNICALL
Java_com_gjs_facelibrary_Face_faceInit(JNIEnv *env, jobject instance, jstring modelDir_) {
    const char *modelDir = env->GetStringUTFChars(modelDir_, 0);

    char haarModelFile[256] = {0};
    char ncnnModelBinFile[256] = {0};
    char ncnnModelParamFile[256] = {0};

    LOGI("modelDir:%s", modelDir);

    //face detection 加载资源
    sprintf(haarModelFile, "%s/%s", modelDir, "haarcascade_frontalface_alt2.xml");
    face_detector.setFaceCascade(haarModelFile);

    // ncnn 加载资源
    sprintf(ncnnModelBinFile, "%s/%s", modelDir, "face_alig.bin");
    sprintf(ncnnModelParamFile, "%s/%s", modelDir, "face_alig.param");
    net.load_param(ncnnModelParamFile);
    net.load_model(ncnnModelBinFile);

    // 释放 jvm 资源
    env->ReleaseStringUTFChars(modelDir_, modelDir);
}

extern "C"
JNIEXPORT void JNICALL
Java_com_gjs_facelibrary_Face_getLandmark(JNIEnv * env, jobject instance, jint rotation, jbyteArray yuv420sp, jint width, jint height, jintArray rgbOut){
    // 获取 jvm 数据（分配内存）
    int32_t *bgra = (jint*) (env->GetPrimitiveArrayCritical( rgbOut, 0));
    int8_t *yuv = (jbyte*) env->GetPrimitiveArrayCritical( yuv420sp, 0);

    LOGI("getLandmark rotation:%d width:%d height:%d", rotation, width, height);

    // 转化维 rgba 格式
    YUVtoBGRA(yuv, width, height, bgra);

    // 转化维 Mat 格式 // 这里传入的 宽高是display rotaion的, 实际这里没有旋转是 sensor的原始方向
    cv::Mat matBgra( height, width, CV_8UC4, bgra);

    // test
    if (false){
        cv::rectangle(matBgra, cv::Rect(200, 50, 10, 10), cv::Scalar(255, 255, 0), 3);
        cv::rectangle(matBgra, cv::Rect(200, 300, 10, 10), cv::Scalar(255, 255, 0), 3);
        cv::rectangle(matBgra, cv::Rect(200, 400, 10, 10), cv::Scalar(255, 255, 0), 3);
        // 释放 jvm 内存占用
        env->ReleasePrimitiveArrayCritical( rgbOut, bgra, 0);
        env->ReleasePrimitiveArrayCritical( yuv420sp, yuv, 0);
        return;
    }

    cv::Mat matBgraT = cv::Mat(matBgra);
    // 转置图像,且X镜像
    if(0 == rotation){
        ;
    }else if(90 == rotation){
        matBgraT = matBgraT.t();
        cv::flip(matBgraT, matBgraT, 1);
    }else if(180 == rotation){
        cv::flip(matBgraT, matBgraT, 0);
        cv::flip(matBgraT, matBgraT, 1);
    }else if(270 == rotation){
        matBgraT = matBgraT.t();
        cv::flip(matBgraT, matBgraT, 0);
    }else{
        LOGE("unkown rotation:%d", rotation);
        return;
    }

    // 获得灰度图
    cv::Mat matGray;
    cv::cvtColor(matBgraT, matGray, CV_BGRA2GRAY);

    // 脸部识别
    double t = (double)cvGetTickCount();
    face_detector.detect(matGray);
    t = (double)cvGetTickCount() - t;
    LOGI("face detect use time = %g ms\n",  t/(cvGetTickFrequency()*1000) );//毫秒

    // 没有找到
    if (!face_detector.isFaceFound()){
        LOGW("no face found");
    }else{ // 找到
        LOGI("face found");
        // 截取面部方形
        cv::Rect faceRect = face_detector.face();
        // 缩放到最大面部
        // 对角点到边缘的绝对值
        int x_b_distance = matBgraT.cols - faceRect.x - faceRect.width;
        int y_b_distance = matBgraT.rows - faceRect.y - faceRect.height;
        // 选取 四个距离中最小 x,y,x_b_distance,y_b_distance 采用两两比较
        int x_w = x_b_distance > faceRect.x ? faceRect.x : x_b_distance;
        int y_h = y_b_distance > faceRect.y ? faceRect.y : y_b_distance;
        int min_distance = x_w > y_h ? y_h : x_w;
        // 保证最小距离 不大于原图像的 0.3 倍数
        float ratio = 0.1;
        if(min_distance/(float)faceRect.height > ratio){
            min_distance = (int)(faceRect.height*ratio);
        }
        faceRect.x -= min_distance;
        faceRect.y -= min_distance;
        faceRect.width += (2*min_distance);
        faceRect.height += (2*min_distance);

        LOGI("gjs: width:%d height:%d \n", faceRect.width, faceRect.height);
        cv::Mat roi = matGray(faceRect);
        cv::Mat face = cv::Mat(faceRect.size(), matGray.type());
        roi.copyTo(face);

        // 缩放到 40*40
        cv::Mat face_40 = cv::Mat(cv::Size(40, 40), face.type());
        cv::resize(face, face_40, cv::Size(40, 40));

        // 获取均值 标准准差
        cv::Mat mean, stddev;
        cv::meanStdDev(face_40, mean, stddev);
        float fmean = *mean.data;
        float fnorm = (float)1. / *stddev.data;

        // 计算结果
        ncnn::Mat in = ncnn::Mat::from_pixels(face_40.data, ncnn::Mat::PIXEL_GRAY, face_40.cols, face_40.rows);
        const float mean_vals[3] = {fmean, fmean, fmean};
        const float norm_vals[3] = {fnorm, fnorm, fnorm};
        in.substract_mean_normalize(mean_vals, norm_vals);

        ncnn::Mat out;
        t = (double)cvGetTickCount(); // 开始面部识别计时
        ncnn::Extractor ex = net.create_extractor();
        ex.set_light_mode(true);
        ex.input("data", in);
        ex.extract("Dense2", out);
        t = (double)cvGetTickCount() - t; // 计算面部识别时间
        LOGI("face alignment use time = %g ms\n",  t/(cvGetTickFrequency()*1000) ); //毫秒
//        LOGI("result c:%d w:%d h:%d ", out.c, out.w, out.h);

        for (int j=0; j<out.c; )
        {
            // 获取预测数值
            float x = *(out.data + out.cstep * j);
            float y = *(out.data + out.cstep * (j+1));
            j += 2;

            // 计算原始label
            int rx = (int)(faceRect.width * x + faceRect.x);
            int ry = (int)(faceRect.height * y + faceRect.y);

            int ox = rx;
            int oy = ry;
            if(0 == rotation){
                ;
            }else if(90 == rotation){
                ox = ry;
                oy = matBgra.rows - rx;
            }else if(180 == rotation){
                ox = matBgra.cols - rx;
                oy = matBgra.rows - ry;
            }else if(270 == rotation){
                ox = matBgra.cols - ry;
                oy = rx;
            }else{
                ;
            }
            // 绘制特征点到原图像
            cv::circle(matBgra, cv::Point(ox, oy), 1, cv::Scalar(0, 255, 0));
        }
        // 绘制漫步识别方框到原图像
        cv::Rect r = cv::Rect(faceRect);
        if(0 == rotation){
            ;
        }else if(90 == rotation){
            // 算出原始四个坐标
            /*
             * faceRect.x_max = faceRect.x + faceRect.width;
             * faceRect.y_max = faceRect.y + faceRect.height;
             *
             * r.x_max = faceRect.y_max
             * =faceRect.y + faceRect.height
             *
             * r.y_max = matBgra.rows - faceRect.x_max
             * = matBgra.rows - faceRect.x - faceRect.width
             *
             * r.w = r.x_max - r.x
             * = faceRect.y + faceRect.height - faceRect.y
             * = faceRect.height
             *
             * r.h = r.y_max - r.y
             * = matBgra.rows - faceRect.x - faceRect.width - matBgra.rows + faceRect.x
             * = - faceRect.width
             */

            r.x = faceRect.y;
            r.y = matBgra.rows - faceRect.x - faceRect.width;
            r.width = faceRect.height;
            r.height = faceRect.width;
        }else if(180 == rotation){
            r.x = matBgra.cols - faceRect.x - faceRect.width;
            r.y = matBgra.rows = faceRect.y - faceRect.height;
        }else if(270 == rotation){
            r.x = matBgra.cols - faceRect.y - faceRect.height;
            r.y = faceRect.x;
            r.width = faceRect.height;
            r.height = faceRect.width;
        }else{

        }
        cv::rectangle(matBgra, r, cv::Scalar(255, 255, 0));
    }

    // 释放 jvm 内存占用
    env->ReleasePrimitiveArrayCritical( rgbOut, bgra, 0);
    env->ReleasePrimitiveArrayCritical( yuv420sp, yuv, 0);
}
