#!/usr/bin/python
# -*- coding: utf-8 -*-


import cv2
import dlib
import sys
import os
import numpy as np


gimgpath = '/home/gjs/download/tmp/color000001-0.png'
glm = [140.0,250.0,141.666656494,293.333312988,146.666656494,338.333312988,153.333328247,381.666656494,170.0,423.333312988,196.666656494,460.0,225.0,491.666656494,260.0,515.0,298.333312988,521.666625977,338.333312988,515.0,375.0,493.333312988,405.0,460.0,430.0,421.666656494,446.666656494,376.666656494,453.333312988,328.333312988,458.333312988,283.333312988,458.333312988,236.666656494,163.333328247,220.0,181.666656494,195.0,211.666656494,190.0,241.666656494,196.666656494,271.666656494,210.0,316.666656494,210.0,343.333312988,196.666656494,373.333312988,188.333328247,405.0,191.666656494,426.666656494,210.0,295.0,250.0,296.666656494,281.666656494,298.333312988,313.333312988,300.0,345.0,270.0,370.0,285.0,373.333312988,300.0,376.666656494,315.0,371.666656494,330.0,368.333312988,195.0,258.333312988,213.333328247,248.333328247,235.0,250.0,250.0,265.0,231.666656494,270.0,210.0,270.0,341.666656494,260.0,356.666656494,243.333328247,378.333312988,241.666656494,398.333312988,250.0,381.666656494,261.666656494,360.0,265.0,246.666656494,425.0,265.0,413.333312988,285.0,406.666656494,301.666656494,411.666656494,316.666656494,405.0,335.0,411.666656494,355.0,425.0,336.666656494,441.666656494,316.666656494,451.666656494,301.666656494,453.333312988,283.333312988,451.666656494,265.0,445.0,256.666656494,426.666656494,285.0,425.0,301.666656494,426.666656494,318.333312988,423.333312988,345.0,425.0,316.666656494,426.666656494,300.0,430.0,285.0,426.666656494]
# glm = np.array(glm, dtype=int).reshape((68, 2))

gdm = np.array([113,161,485,533])



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
        imgpath = sline[0]
        dm = np.array(sline[1:5], dtype=np.int)  # dm 为 int
        lm = np.array(sline[5:], dtype=np.float32)
        mapimgdmlm[imgpath]=(dm, lm)
    return mapimgdmlm



if __name__ == '__main__':
    # img dm lm 都以 np 表示
    # img: BGR
    # dm: l t r b
    if True:
        img = cv2.imread(gimgpath)
        drawimg(img, gdm, glm)
        cv2.waitKey(0)
        exit()

    if len(sys.argv) != 4:
        print ('usage: python drwaimgdmlm.py start end imgdmlmlist.txt')
        exit()
    start = int(sys.argv[1])
    end = int(sys.argv[2])
    imgdmlmlist = sys.argv[3]
    mapimgdmlm = readimgdmlmlist(imgdmlmlist, start, end)
    for imgpath, dmlm in mapimgdmlm.items():
        img = cv2.imread(imgpath)
        dm = dmlm[0]
        lm = dmlm[1]
        drawimg(img, dm, lm)
        cv2.waitKey(0)
