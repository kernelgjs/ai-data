# -*- coding: utf-8 -*-

import os
import sys
import time
import datetime
import cv2
import dlib
import numpy as np
import h5py


# 引入shape_predictor_68_face_landmarks.dat文件
dependency_predictor = '/traindata/dealdata/tools/shape_predictor_68_face_landmarks.dat'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(dependency_predictor)


def landmark_detection(filename):
    img = cv2.imread(filename, 3)
    rects = detector(img, 1)
    faceframe = np.array([])
    if (len(rects) != 0):
        landmarks = np.array([[p.x, p.y] for p in predictor(img, rects[0]).parts()]);
        for d in rects:
            faceframe = np.array([[d.left(), d.top()], [d.right(), d.bottom()]])
    else:
        return (None, None, None)
    return img, faceframe.reshape(-1), landmarks.reshape(-1)


def resize(img):
    img_resized = cv2.resize(img, (227, 227), interpolation=cv2.INTER_CUBIC)
    # cv2.imwrite(filename,img_resized)
    return img_resized


class DealHdf5:
    def __init__(self, batchsize=10000, cachesize=1000, prefix=''):
        self.prefix = prefix  # hdf5 文件名称
        self.cacheSize = cachesize
        self.dsetSize = batchsize
        self.fhandle = None  # hdf5 文件handle
        self.dsetImg = None  # hdf5 img 数据库
        self.dsetLabel = None  # hdf5 label 数据库
        self.dsetIter = 0  # 当前数据库游标
        self.cacheIter = 0  # 缓存游标
        self.cacheImg = np.empty((cachesize, 3, 227, 227))
        self.cacheLabel = np.empty((cachesize, 1, 1, 136))
        # BGR
        self.mean = np.array([85.27525835,96.94723657,119.02201938]).reshape(3, 1, 1)

    def to_hdf5(self, image, label):
        timg = image.transpose(2, 0, 1) - self.mean
        tlabel = label.reshape(1, 1, -1) / 250.0
        self.cacheImg[self.cacheIter, :, :, :] = timg
        self.cacheLabel[self.cacheIter, :, :, :] = tlabel
        self.cacheIter += 1
        if self.cacheIter == self.cacheSize:
            self.flush_cache()

    def flush_cache(self, isover=False):
        if self.cacheIter == 0:
            return
        if self.fhandle is None:
            curtime = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            self.fhandle = h5py.File(self.prefix+'_data_'+curtime+'.hd5', 'w')
            self.dsetImg = self.fhandle.create_dataset('data', shape=(self.dsetSize, 3, 227, 227), maxshape=(None, 3, 227, 227),
                                                       compression='gzip', compression_opts=4)
            self.dsetLabel = self.fhandle.create_dataset('label', shape=(self.dsetSize, 1, 1, 136), compression='gzip',
                                                         compression_opts=4)
        self.dsetImg[self.dsetIter:self.dsetIter+self.cacheIter, :, :, :] = self.cacheImg[:self.cacheIter, :, :, :]
        self.dsetLabel[self.dsetIter:self.dsetIter+self.cacheIter, :, :, :] = self.cacheLabel[:self.cacheIter, :, :, :]
        self.dsetIter += self.cacheIter
        self.cacheIter = 0
        if self.dsetSize - self.dsetIter < self.cacheSize or isover:
            self._close_hdf5()

    def _close_hdf5(self):
        if self.fhandle is None:
            return
        if self.dsetIter != self.dsetSize:
            self.dsetImg.resize((self.dsetIter, 3, 227, 227))
            self.dsetLabel.resize((self.dsetIter, 1, 1, 136))
        self.fhandle.close()
        self.fhandle = None
        self.dsetIter = 0


# 获取所有文件路径
def get_files(fileListFile, start, end):
    allfiles = []
    with open(fileListFile, 'r') as f:
        allfiles = f.readlines()
    allfiles = [s.strip('\n') for s in allfiles ]
    return allfiles[start:end]


# 开始处理
if __name__ == '__main__':
    if len(sys.argv) != 4:
        print ('usage: python deal.py start end filelistfile')
        exit()
    start = int(sys.argv[1])
    end = int(sys.argv[2])
    fileListFile = sys.argv[3]
    files = get_files(fileListFile, start, end)

    dealHdfe = DealHdf5(2000, 500, prefix=str(start)+'-'+str(end))
    sucessFileHandle = open(str(start) + '-' + str(end) + '-landmark-filelist.txt', 'a')
    sucessCount = 0
    # 循环处理
    for i, imgfile in enumerate(files):
        try:
            print ('start deal file:'+imgfile)
            '''
            des: 计算特征点
            in：imgfile
            out：img , dm(decetion mark), lm(landmark)
            '''
            t = time.time()
            img, dm, lm = landmark_detection(imgfile)
            print ('landmark time:'+str(time.time()-t))
            if img is None:
                print 'error no face found'
                continue
            sucessFileHandle.write(files[i]+','.join([str(m) for m in ['']+ dm.tolist() + lm.tolist()])+'\n')
            sucessCount += 1
            '''
            des: 缩放图像
            in: img
            out: img (resized)
            '''
            t = time.time()
            img = resize(img)
            print ('resize time:'+str(time.time()-t))

            '''
            des: 转换为hdf5 图像减去均值,坐标除以250
            in: img, lanmark
            out:
            '''
            # 存储 hdf5 坐标归一，图像去均值
            t = time.time()
            dealHdfe.to_hdf5(img, lm)
            print ('dealHdfe time:'+str(time.time()-t))
            print ('deal '+str(i)+' success '+str(sucessCount))
            sys.stdout.flush()
        except Exception as e:
            print("Unexpected Error: {}".format(e))
    # 最后刷新 hdf5
    dealHdfe.flush_cache(isover=True)
    print("deal isvoer")



