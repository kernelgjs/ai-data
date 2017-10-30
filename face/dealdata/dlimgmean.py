#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import cv2
import traceback
import numpy as np


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

if __name__ == '__main__':
    if len(sys.argv) != 7:
        print ('usage: python dlimgmean.py start end imgdmlmlist.txt netimgsize netimgchannel meanstdev.txt')
        exit()
    start = int(sys.argv[1])
    end = int(sys.argv[2])
    imglistpath = sys.argv[3]
    netimgsize = int(sys.argv[4])
    netimgchannel = int(sys.argv[5])
    meanstdevpath = sys.argv[6]

    mapimgdmlm = readimgdmlmlist(imglistpath, start, end)

    # 分别统计B G R 三通道mean
    sumcout = 0
    if netimgchannel == 1:
        meansum = np.array([0.0])
        stdevsum = np.array([0.0])
    else:
        meansum = np.array([0.0, 0.0, 0.0])
        stdevsum = np.array([0.0, 0.0, 0.0])

    for imgpath, dmlm in mapimgdmlm.items():
        try:
            print('img:{}'.format(imgpath))
            img = cv2.imread(imgpath)
            dm = dmlm[0]
            if img is None:
                print ('error read img:' + imgpath)
                continue
            # 获取面部
            roiimg = img[dm[1]:dm[3]+1, dm[0]:dm[2]+1]
            # 缩放 灰度
            roiimg = cv2.resize(roiimg, (netimgsize, netimgsize), interpolation=cv2.INTER_CUBIC)
            if (netimgchannel == 1):
                roiimg = cv2.cvtColor(roiimg, cv2.COLOR_BGR2GRAY)
            meansum += np.mean(roiimg, axis=(0, 1))
            stdevsum += np.std(roiimg, axis=(0, 1))
            sumcout += 1
            print (' sumcount:{} meansum:{} stdevsum:{}'.format(sumcout, meansum, stdevsum))
        except Exception as e:
            print (traceback.format_exc())
    mean = meansum/sumcout
    stdev = stdevsum/sumcout
    with open(meanstdevpath, 'w') as f:
        f.write('mean:'+str(mean))
        f.write('stdev:'+str(stdev))
    print ('mean:'+str(mean))
    print ('stdev:'+str(stdev))


