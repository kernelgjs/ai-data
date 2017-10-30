#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import traceback
import cv2
import dlib
import numpy as np
import time


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


def drawimg(img, dm=None, lm=None, name=''):
    if lm is not None:
        tlm = np.array(lm, dtype=np.int)
        tlm = tlm.reshape(68, 2)
        for i, point in enumerate(tlm):
            point = tuple(point)
            cv2.circle(img, point, 2, (255, 255, 0), -1)
    # cv2.putText(img, str(i+1), point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)
    if name == '':
        cl = (255, 0, 0)
    elif name == '1':
        cl = (0, 255, 0)
    elif name == '2':
        cl = (0, 0, 255)
    else:
        cl = (255, 255, 0)

    if dm is not None:
        cv2.rectangle(img, (dm[0], dm[1]), (dm[2], dm[3]), cl)
    # cv2.resizeWindow(name)
    # cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, img)
    # cv2.waitKey(0)



if __name__ == '__main__':
    # img dm lm 都以 np 表示
    # img: BGR
    # dm: l t r b
    if len(sys.argv) != 5:
        print ('usage: python dlpickimg.py start end imgdmlmlist.txt pickimgdmlmlist.txt')
        exit()
    start = int(sys.argv[1])
    end = int(sys.argv[2])
    imgdmlmlistpath = sys.argv[3]
    pickimgdmlmlistpath = sys.argv[4]

    quitexe = 113  # 'q'
    imgyes = 10  # enter
    imgno = 32  # 空格

    sf = open(pickimgdmlmlistpath, 'a')
    mapimgdmlm = readimgdmlmlist(imgdmlmlistpath, start, end)
    pickcount = 0
    dealcount = 0
    for imgpath, dmlm in mapimgdmlm.items():
        img = cv2.imread(imgpath)
        dm = dmlm[0]
        lm = dmlm[1]
        dealcount += 1
        while True:
            drawimg(img, dm, lm)
            code = cv2.waitKey(0) & 0xFF
            if code == quitexe:
                print ('quitexe')
                exit()
            elif code == imgyes:
                sdm = np.array(dm, dtype=np.str)
                slm = np.array(lm, dtype=np.str)
                line = ','.join([imgpath] + sdm.tolist() + slm.tolist()) + '\n'
                sf.write(line)
                sf.flush()
                pickcount += 1
                print ('pass deal {} pick {}'.format(dealcount, pickcount))
                break
            elif code == imgno:
                print ('nopass deal {} pick {}'.format(dealcount, pickcount))
                break
            else:
                print 'unkown key'
