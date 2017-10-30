#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import cv2
import dlib
import time


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print ('usage: python getBox.py filelistFile sotreFIle')
        exit()
    filelistFile = sys.argv[1]
    storeFile = sys.argv[2]

    # 获取所有文件路径
    allfiles = []
    with open(filelistFile, 'r') as f:
        allfiles = f.readlines()
    allfiles = [s.strip('\n') for s in allfiles ]

    sf = open(storeFile, 'a')
    lmFiles = []
    count = 0
    sucessCount = 0
    for imgfile in allfiles:
        count += 1
        print ('deal img:'+imgfile)
        if imgfile.endswith('.jpg'):
            ptsfile = imgfile.replace('.jpg', '.pts')
        elif imgfile.endswith('.png'):
            ptsfile = imgfile.replace('.png', '.pts')
        else:
            print ('unkown endwith ')
            continue
        with open(ptsfile, 'r') as f:
            lines = f.readlines()
        lines = [l.strip('\n') for l in lines]
        lines = [x for point in lines[3:71] for x in point.split(' ')]
        line = ','.join(tuple([imgfile]+lines))+'\n'
        lmFiles.append(line)
        sucessCount += 1
        print ('deal '+str(count)+' sucess ' +str(sucessCount))
        sf.write(line)
        sf.flush()


