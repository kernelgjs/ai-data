#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import traceback
import cv2
import dlib
import numpy as np
import time


dependency_predictor = '/home/gjs/self-github/ai-data/face/dealdata/model/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(dependency_predictor)


def landmark_detection(img):
    rects = detector(img, 1)
    faceframe = np.array([])
    if (len(rects) != 0):
        landmarks = np.array([[p.x, p.y] for p in predictor(img, rects[0]).parts()], dtype=np.float32);
        for d in rects:
            faceframe = np.array([[d.left(), d.top()], [d.right(), d.bottom()]])
    else:
        return (None, None)
    return faceframe.reshape(-1), landmarks.reshape(-1)


if __name__ == '__main__':

    if len(sys.argv) != 5:
        print ('usage: python dldmlmlist.py start end imglist.txt imgdmlmlist.txt')
        exit()
    start = int(sys.argv[1])
    end = int(sys.argv[2])
    imglistpath = sys.argv[3]
    storefile = sys.argv[4]

    # 获取所有文件路径
    with open(imglistpath, 'r') as f:
        imglist = f.readlines()
        imglist = imglist[start:end]
        imglist = [s.strip('\n') for s in imglist]
        print (start, end, len(imglist))

    mapimgdmlm = {}
    sf = open(storefile, 'a')
    boxFiles = []
    count = 0
    sucessCount = 0
    for imgpath in imglist:
        try:
            count += 1
            print ('deal img:' + imgpath)
            img = cv2.imread(imgpath)
            if img is None:
                print ('error read img:' + imgpath)
                continue
            t = time.time()
            dm, lm = landmark_detection(img)
            if dm is None:
                print ('error no face found:' + imgpath)
                continue
            print ('deal dm lm time:{}'.format(time.time()-t))
            if True:
                sdm = np.array(dm, dtype=np.str)
                slm = np.array(lm, dtype=np.str)
                line = ','.join([imgpath] + sdm.tolist() + slm.tolist()) + '\n'
                sf.write(line)
                sf.flush()
            sucessCount += 1
            print ('deal '+str(count)+' sucess ' +str(sucessCount))
        except Exception as e:
            print (traceback.format_exc())
