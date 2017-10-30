#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import traceback
import sys
import time
import datetime
import cv2
import dlib
import numpy as np
import h5py



class DealHdf5:
    def __init__(self, batchsize, cachesize, imgsize, imgchannel, dataname, labelname, prefix=''):
        self.prefix = prefix  # hdf5 文件名称
        self.cachesize = cachesize
        self.dsetSize = batchsize
        self.imgsize = imgsize
        self.imgchannel = imgchannel
        self.labelsize = 136
        self.dataname = dataname
        self.labelname = labelname
        self.fhandle = None  # hdf5 文件handle
        self.dsetImg = None  # hdf5 img 数据库
        self.dsetLabel = None  # hdf5 label 数据库
        self.dsetIter = 0  # 当前数据库游标
        self.cacheIter = 0  # 缓存游标
        self.cacheImg = np.empty((self.cachesize, self.imgchannel, self.imgsize, self.imgsize))
        self.cacheLabel = np.empty((self.cachesize, self.labelsize))
        # BGR
        # self.mean = np.array([85.27525835,96.94723657,119.02201938]).reshape(3, 1, 1)

    def to_hdf5(self, image, label):
        # mean, std_dev = cv2.meanStdDev(image)
        timg = (image - gmean) / (0.000001 + gstdev)
        tlabel = label.reshape(-1) / (self.imgsize*1.0)
        if self.imgchannel == 3:
            timg = timg.transpose((2, 0, 1))
        self.cacheImg[self.cacheIter,:, :, :] = timg
        self.cacheLabel[self.cacheIter, :] = tlabel
        self.cacheIter += 1
        if self.cacheIter == self.cachesize:
            self.flush_cache()

    def flush_cache(self, isover=False):
        if self.cacheIter == 0:
            return
        if self.fhandle is None:
            curtime = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            self.fhandle = h5py.File(self.prefix+'_data_'+curtime+'.hd5', 'w')
            self.dsetImg = self.fhandle.create_dataset(self.dataname, shape=(self.dsetSize, self.imgchannel, self.imgsize, self.imgsize), maxshape=(None, self.imgchannel, self.imgsize, self.imgsize),
                                                       compression='gzip', compression_opts=4)
            self.dsetLabel = self.fhandle.create_dataset(self.labelname, shape=(self.dsetSize, self.labelsize), maxshape=(None, self.labelsize),
                                                         compression='gzip', compression_opts=4)
        self.dsetImg[self.dsetIter:self.dsetIter+self.cacheIter, :, :, :] = self.cacheImg[:self.cacheIter, :, :, :]
        self.dsetLabel[self.dsetIter:self.dsetIter+self.cacheIter, :] = self.cacheLabel[:self.cacheIter, :]
        self.dsetIter += self.cacheIter
        self.cacheIter = 0
        if self.dsetSize - self.dsetIter < self.cachesize or isover:
            self._close_hdf5()

    def _close_hdf5(self):
        if self.fhandle is None:
            return
        if self.dsetIter != self.dsetSize:
            self.dsetImg.resize((self.dsetIter, self.imgchannel, self.imgsize, self.imgsize))
            self.dsetLabel.resize((self.dsetIter, self.labelsize))
        self.fhandle.close()
        self.fhandle = None
        self.dsetIter = 0



def drawdmlm(img, dm=None, lm=None):
    if lm is not None:
        lm = np.array(lm, dtype=int).reshape(68, 2)
        for i, point in enumerate(lm):
            point = (point[0], point[1])
            # point = tuple(point)
            cv2.circle(img, point, 2, (255, 255, 0), -1)
            # cv2.putText(img, str(i + 1), point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)
    if dm is not None:
        cv2.rectangle(img, (int(dm[0]),int(dm[1])), (int(dm[2]),int(dm[3])), (255, 255,0))
    # cv2.resizeWindow()
    # cv2.namedWindow('', cv2.WINDOW_NORMAL)
    cv2.imshow('', img)
    # cv2.imwrite('test.png', img)
    cv2.waitKey(0)

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

# roi 区域放大倍数/缩小为负
gradio = 0
#
ghdf5size = 1000
gcachesize = 100
gnetimgsize = 60
gnetimgchannel = 1
gdataname = 'data'
glabelname = 'landmarks'
gmean = None
gstdev = None

# 开始处理
if __name__ == '__main__':

    if len(sys.argv) != 12:
        print ('usage: python dlimgdmlm2hd5.py start end imgdmlmlist.txt hdf5size cachesize netimgsize netimgchannel dataname labelname mean stdev\n'
               'ex: dlimgdmlm2hd5.py 0 10000 imgdmlmlist.txt 1000 100 60 1 data landmarks 1.1,1.2,1.3 0.1,0.2,0.3')
        exit()
    start = int(sys.argv[1])
    end = int(sys.argv[2])
    imgdmlmlistpath = sys.argv[3]
    ghdf5size = int(sys.argv[4])
    gcachesize = int(sys.argv[5])
    gnetimgsize = int(sys.argv[6])
    gnetimgchannel = int(sys.argv[7])
    gdataname = sys.argv[8]
    glabelname = sys.argv[9]
    strmean = sys.argv[10]
    strstdev = sys.argv[11]

    if gnetimgchannel!=1 and gnetimgchannel!=3:
        print ('error imgchannel must 1 or 3')
        exit()
    if gnetimgchannel!=len(strmean.split(',')) or gnetimgchannel!=len(strstdev.split(',')):
        print ('error netimgchannel:{} mean:{} stdev:{}'.format(gnetimgchannel, strmean, strstdev))
        exit()
    else:
        gmean = np.array(strmean.split(','), dtype=np.float32)
        gstdev = np.array(strstdev.split(','), dtype=np.float32)

    print ('start={}\nend={}\nhdf5size={}\ncachesize={}\nnetimgszie={}\nnetimgchannel={}\ndataname={}\nlabelname={}\nmean={}\nstdev={}\n'.format(start, end, ghdf5size, gcachesize, gnetimgsize, gnetimgchannel, gdataname, glabelname, gmean, gstdev))
    while False:
        print 'please input y/n'
        res = raw_input()
        if res == 'y':
            break
        elif res == 'n':
            exit()
        else:
            pass


    mapimgdmlm = readimgdmlmlist(imgdmlmlistpath, start, end)
    dealHdfe = DealHdf5(ghdf5size, gcachesize, gnetimgsize, gnetimgchannel, gdataname, glabelname, prefix=str(start)+'-'+str(end))
    sucessCount = 0
    # 循环处理
    count = 0
    for imgpath, dmlm in mapimgdmlm.items():
        try:
            count += 1
            print ('start deal file:'+imgpath)
            img = cv2.imread(imgpath, cv2.IMREAD_COLOR)
            dm = dmlm[0]
            lm = dmlm[1]
            # draw_dm_lm(img, dm=dm, lm=lm)
            if img is None:
                print 'error read img'
                continue
            '''
            des:裁剪图像 lm 变换
            '''
            faceRect = dlib.rectangle(int(dm[0]), int(dm[1]), int(dm[2]), int(dm[3]))
            roi = img[faceRect.top():faceRect.bottom()+1, faceRect.left():faceRect.right()+1, ]
            lm[0::2] -= faceRect.left()
            lm[1::2] -= faceRect.top()
            # draw_dm_lm(roi, lm=lm)
            '''
            des: 缩放图像
            '''
            roiresized = cv2.resize(roi, (gnetimgsize, gnetimgsize), interpolation=cv2.INTER_CUBIC)
            lmresized = lm * gnetimgsize / (faceRect.right()-faceRect.left())
            if (gnetimgchannel == 1):
                roiresized = cv2.cvtColor(roiresized, cv2.COLOR_BGR2GRAY)
                # drawdmlm(roiresized, lm=lmresized)
            '''
            des: 转换为hdf5 图像减去均值,坐标除以60
            in: img, lanmark
            '''
            # 存储 hdf5 坐标归一，图像去均值
            t = time.time()
            dealHdfe.to_hdf5(roiresized, lmresized)
            sucessCount += 1
            print ('deal '+str(count)+' success '+str(sucessCount))
            sys.stdout.flush()
        except Exception as e:
            print("Unexpected Error: {}".format(e))
            print (traceback.format_exc())
    # 最后刷新 hdf5
    dealHdfe.flush_cache(isover=True)
    print("deal isvoer")
