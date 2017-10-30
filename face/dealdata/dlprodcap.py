#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import caffe as cf
import sys
import dlib
import os

'''
绘制图形
'''
def drawimg(img, dm=None, lm=None, name='', txt=''):
    if name == '':
        cl = (255, 0, 0)
    elif name == '1':
        cl = (0, 255, 0)
    elif name == '2':
        cl = (0, 0, 255)
    else:
        cl = (255, 255, 0)

    if lm is not None:
        tlm = np.array(lm, dtype=np.int)
        tlm = tlm.reshape(68, 2)
        for i, point in enumerate(tlm):
            point = tuple(point)
            cv2.circle(img, point, 2, cl, -1)
    if txt != '':
        cv2.putText(img, txt, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255)
    if dm is not None:
        cv2.rectangle(img, (dm[0], dm[1]), (dm[2], dm[3]), cl)
    # cv2.resizeWindow(name)
    # cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    if name != '1':
        cv2.imshow(name, img)
    # cv2.waitKey(0)


def getorglm(prelm, dm):
    ow = dm[2]-dm[0]
    tlm = prelm * ow
    tlm[0::2]+=dm[0]
    tlm[1::2]+=dm[1]
    return np.array(tlm, dtype=int)


def loss(lm, prodlm):
    tlm = lm.reshape(1, -1)
    tprodlm = prodlm.reshape(1, -1)
    selfdiff = np.zeros_like(lm, dtype=np.float32)
    y_true = tlm
    y_pred = tprodlm
    delX = y_true[:, 72] - y_true[:, 90]  # del X size 16
    delY = y_true[:, 73] - y_true[:, 91]  # del y size 16
    selfinterOc = (1e-6 + (delX * delX + delY * delY) ** 0.5).T  # Euclidain distance
    diff = (y_pred - y_true).T  # Transpose so we can divide a (16,10) array by (16,1)

    selfdiff[...] = (diff / selfinterOc).T  # We transpose back to (16,10)
    return np.sum(selfdiff ** 2) / tlm.shape[0] / 2.  # Loss is scalar

def getdmface(faces):
    # 转换为dlib 方式, 后续检测关键点,采用dlib
    face = dlib.rectangle(0, 0, 0, 0)
    for k, fc in enumerate(faces):
        left = fc.left()
        top = fc.top()
        right = fc.right()
        bottom = fc.bottom()
        # 仅比较宽
        if (right-left) > (face.right()-face.left()):
            face = dlib.rectangle(left, top, right, bottom)
    dm = np.array([ face.left(), face.top(), face.right(), face.bottom()])
    return dm, face


def getlm(img, face):
    lm = np.array([[p.x, p.y] for p in predictor(img, face).parts()])
    return lm.reshape(-1)


cap = cv2.VideoCapture(0)
hc = cv2.CascadeClassifier("/home/gjs/self-github/ai-data/face/dealdata/model/haarcascade_frontalface_alt2.xml")
predictor = dlib.shape_predictor('/home/gjs/self-github/ai-data/face/dealdata/model/shape_predictor_68_face_landmarks.dat')
detector = dlib.get_frontal_face_detector()

if __name__ == '__main__':
    # img dm lm 都以 np 表示
    # img: BGR
    # dm: l t r b
    if len(sys.argv) != 9:
        print ('usage: python dlprodcap.py caffemodel caffeproto netimgsize netimgchannel dataneme labelname mean stdev')
        exit()
    caffemodelpath = sys.argv[1]
    caffeprotopath = sys.argv[2]
    netimgsize = int(sys.argv[3])
    netimgchannel = int(sys.argv[4])
    dataname = sys.argv[5]
    labelname = sys.argv[6]
    smean = sys.argv[7]
    sstdev = sys.argv[8]

    mean = np.array(smean.split(','), dtype=np.float32)
    stdev = np.array(sstdev.split(','), dtype=np.float32)

    net = cf.Net(caffeprotopath, caffemodelpath, cf.TEST)
    cf.set_mode_cpu()

    while True:
        ret, img = cap.read()
        if img is None:
            print 'cap.read img is none continue'
            continue
        faces = detector(img, 1)
        dm, face = getdmface(faces)
        lm = getlm(img, face)

        # 获取面部
        roiimg = img[dm[1]:dm[3]+1, dm[0]:dm[2]+1]
        roilm = np.array(lm)
        roilm[0::2] -= dm[0]
        roilm[1::2] -= dm[1]
        # 缩放 灰度
        roiimg = cv2.resize(roiimg, (netimgsize, netimgsize), interpolation=cv2.INTER_CUBIC)
        roilm = roilm * netimgsize / (dm[2] - dm[0])
        if (netimgchannel == 1):
            roiimg = cv2.cvtColor(roiimg, cv2.COLOR_BGR2GRAY)
        # 均一化
        # mean, std_dev = cv2.meanStdDev(roiimg)
        roiimg = (roiimg - mean) / (0.000001 + stdev)
        # 网络预测
        net.blobs[dataname].data[...] = roiimg.reshape(netimgchannel, netimgsize, netimgsize)
        net.forward()
        # 获得预测结果
        prodlm = net.blobs[labelname].data.reshape(-1)
        prodlm = getorglm(prodlm, dm)
        lossvalue = loss(lm, prodlm)
        drawimg(img, dm=dm, lm=lm, name='1')  # 原图
        drawimg(img, dm=dm, lm=prodlm, name='2', txt=str(lossvalue))  # 预测图
        code = cv2.waitKey(30)
        if code&0xFF == ord('q'):
            exit()











