#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import traceback
import cv2
import dlib
import time

detector = dlib.get_frontal_face_detector()
def get_faces(img):
    # 获取面部方框
    t = time.time()
    faces = detector(img, 1)
    print ('detector time:'+str(time.time()-t))
    return faces


def get_center_face(faces, center):
    # 转换为dlib 方式, 后续检测关键点,采用dlib
    face = None
    center = dlib.point(center[0], center[1])
    for faceRect in faces:
        if faceRect.contains(center):
            return faceRect
    return face


if __name__ == '__main__':

    if len(sys.argv) != 5:
        print ('usage: python getBox.py start end lmlist.txt dmlist.txt')
        exit()
    start = int(sys.argv[1])
    end = int(sys.argv[2])
    lmlistfilepath = sys.argv[3]
    storeFile = sys.argv[4]

    # 获取所有文件路径
    lmlist = []
    with open(lmlistfilepath, 'r') as f:
        lmlist = f.readlines()
    lmlist = [s.strip('\n') for s in lmlist]
    lmlist = lmlist[start:end]

    mfiles = {}
    # 生成 imgpath center map结构便于处理
    for imglm in lmlist:
        try:
            imglm = imglm.split(',')
            imgpath = imglm[0]
            lm = imglm[1:]
            center = (int(float(lm[68])), int(float(lm[69])))
            mfiles[imgpath] = center
        except Exception as e:
            print (traceback.format_exc())

    sf = open(storeFile, 'w')
    boxFiles = []
    count = 0
    sucessCount = 0
    for timgpath, center in mfiles.items():
        count += 1
        print ('deal img:' + timgpath)
        img = cv2.imread(timgpath)
        if img is None:
            print ('error read img:' + timgpath)
            continue
        faces = get_faces(img)
        if faces is None or len(faces) < 1:
            print ('error no face found:' + timgpath)
            continue
        face = get_center_face(faces, center)
        if face is None:
            print ('canot find faceRect contans lm')
            continue
        sucessCount += 1
        print ('deal '+str(count)+' sucess ' +str(sucessCount))
        imglm = ','.join((timgpath, str(face.left()), str(face.top()), str(face.right()), str(face.bottom()))) + '\n'
        sf.write(imglm)
        sf.flush()



