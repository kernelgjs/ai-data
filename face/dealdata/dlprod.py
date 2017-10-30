#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import caffe as cf
import sys
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
    cv2.imshow(name, img)
    # cv2.waitKey(0)


def getorglm(prelm, dm):
    ow = dm[2]-dm[0]
    tlm = prelm * ow
    tlm[0::2]+=dm[0]
    tlm[1::2]+=dm[1]
    return np.array(tlm, dtype=int)

'''
返回 map key:path value:(dm, lm) 类型 ndarry
'''
def readimgdmlmlist(imglmdm, start, end):
    with open(imglmdm, 'r') as tf:
        imglmdmlist = tf.readlines()
    imglmdmlist = imglmdmlist[start:end]  # 开区间
    imglmdmlist = [b.strip('\n') for b in imglmdmlist ]  # 去除换行
    mapimgdmlm = {}
    for line in imglmdmlist:
        sline = line.split(',')
        if len(sline) != 141:
            print('line format error len not 141 '+line)
            continue
        imgpath = sline[0]
        dm = np.array(sline[1:5], dtype=np.int)  # dm 为 int
        lm = np.array(sline[5:], dtype=np.float32)
        mapimgdmlm[imgpath]=(dm, lm)
    return mapimgdmlm

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

if __name__ == '__main__':
    # img dm lm 都以 np 表示
    # img: BGR
    # dm: l t r b
    if len(sys.argv) != 8:
        print ('usage: python dlprod.py start end imgdmlmlist.txt caffemodel caffeproto netimgsize netimgchannel')
        exit()
    start = int(sys.argv[1])
    end = int(sys.argv[2])
    imgdmlmlistpath = sys.argv[3]
    caffemodelpath = sys.argv[4]
    caffeprotopath = sys.argv[5]
    netimgsize = int(sys.argv[6])
    netimgchannel = int(sys.argv[7])

    net = cf.Net(caffeprotopath, caffemodelpath, cf.TEST)
    cf.set_mode_cpu()

    mapimgdmlm = readimgdmlmlist(imgdmlmlistpath, start, end)
    count = 0

    for imgpath, dmlm in mapimgdmlm.items():
        img = cv2.imread(imgpath)
        dm = dmlm[0]
        lm = dmlm[1]
        count += 1
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
        mean, std_dev = cv2.meanStdDev(roiimg)
        roiimg = (roiimg - mean[0][0]) / (0.000001 + std_dev[0][0])
        # 网络预测
        net.blobs['data'].data[...] = roiimg.reshape(netimgchannel, netimgsize, netimgsize)
        net.forward()
        # 获得预测结果
        prodlm = net.blobs['Dense3'].data.reshape(-1)
        prodlm = getorglm(prodlm, dm)
        lossvalue = loss(lm, prodlm)
        drawimg(img, lm=lm, name='1')  # 原图
        drawimg(img, lm=prodlm, name='2', txt=str(lossvalue))  # 预测图
        cv2.waitKey(0)

    print ('deal over')
    exit()











