#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import traceback
import cv2
import dlib
import time


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print ('usage: python dljoindmlm.py dmlist.txt lmlist.txt imgdmlmlist.txt')
        exit()
    boxfilelistFile = sys.argv[1]
    lmfilelistFile = sys.argv[2]
    storeFile = sys.argv[3]

    # 获取所有文件路径
    ballfiles = []
    lallfiles = []
    with open(boxfilelistFile, 'r') as bf:
        ballfiles = bf.readlines()
    ballfiles = [b.strip('\n') for b in ballfiles ]
    mballfiles = {}
    for bfile in ballfiles:
        if '.png' in bfile:
            end = bfile.find('.png')+len('.png')
        elif '.jpg' in bfile:
            end = bfile.find('.jpg')+len('.jpg')
        else:
            print ('unfind jpg png')
            continue
        imgpath = bfile[:end]
        box = bfile[end:].strip('\n')
        mballfiles[imgpath]=box

    with open(lmfilelistFile, 'r') as bf:
        lallfiles = bf.readlines()
    lallfiles = [l for l in lallfiles ]

    sf = open(storeFile, 'w')
    lballfiles = []
    count = 0
    sucessCount = 0
    for lfile in lallfiles:
        try:
            count += 1
            if '.png' in lfile:
                end = lfile.find('.png')+len('.png')
            elif '.jpg' in lfile:
                end = lfile.find('.jpg')+len('.jpg')
            else:
                print ('unfind jpg png')
                continue
            imgpath = lfile[:end]
            lm = lfile[end:]

            print ('deal img:'+imgpath)
            if imgpath in mballfiles:
                box = mballfiles[imgpath]
            else:
                print ('cant find {} in box file '.format(imgpath))
                continue
            # 检测点是否在方框内（多人脸）

            lbox = [ int(float(x)) for x in box.strip(',').split(',')]
            llandmark = [int(float(x)) for x in lm.strip(',').split(',')]
            faceRect = dlib.rectangle(lbox[0], lbox[1], lbox[2], lbox[3])
            center = dlib.point(llandmark[68], llandmark[69])
            if not faceRect.contains(center):
                print ('!!! center out box ,mey more face')
                continue

            line = imgpath + box + lm
            lballfiles.append(line)
            sucessCount += 1
            print ('deal {} success {}'.format(count, sucessCount))

            sf.write(line)
            sf.flush()
        except Exception as e:
            print (traceback.format_exc())




