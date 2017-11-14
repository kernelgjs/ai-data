# coding=utf-8


import sys

import cv2
import caffe
import numpy as np
import traceback
import random
import cPickle as pickle


'''

仅仅保留 负样本(0) 正样本(1)
用于计算 脸\非脸 分类损失
bottom[pred, label]
top：[validpred, validlabel]

'''

'''
class clsdatafilterlayer(caffe.Layer):
    def setup(self, bottom, top):
        if len(bottom) != 2:
            raise Exception("Need 2 Inputs")

    def reshape(self, bottom, top):
        if True:
            print ('clsdatafilterlayer')
        if False:
            print ('len(bottom):{} len(top):{}'.format(len(bottom), len(top)))
            print ( bottom[0].data.shape)
            print ( bottom[1].data.shape)
        # 获取训练label,转换为 np 处理
        label = np.array(bottom[1].data, dtype=np.int)
        # 获取有效样本（正负）坐标
        self.valid_index = np.where((label==0)|(label==1))[0]
        self.count = len(self.valid_index)
        if self.count == 0:
            print ('!!! valid count is 0 label:{}'.format(label))
        else:
            print ('valid count is:{}'.format(self.count))
        # 设置输出 pred label 的shape

        top[0].reshape(bottom[0].data.shape[0], 2, 1, 1)  # 四维 batch_size, 2(正负）, x,y(卷积坐标)(12*12的是1)
        top[1].reshape(bottom[1].data.shape[0], 1)


    def forward(self, bottom, top):
        if False:
            print ('len(bottom):{} len(top):{}'.format(len(bottom), len(top)))
            print ( bottom[0].data.shape)
            print ( bottom[1].data.shape)
            print ( top[0].data.shape)
            print ( top[1].data.shape)
        # 把有效数据 赋值给 top 前 count
        if self.count == 0:
            top[0].data[:, 0, :, :] = 0
            top[0].data[:, 1, :, :] = -10000  # 为了让loss 计算正确
            top[1].data[...] = 0
        else:
            # 初始化所有输出为 0
            top[0].data[:, 0, :, :] = 0
            top[0].data[:, 1, :, :] = -10000  # 为了让loss 计算正确
            top[1].data[...] = 0
            top[0].data[0:self.count] = bottom[0].data[self.valid_index]
            top[1].data[0:self.count] = bottom[1].data[self.valid_index]
        if False:
            print ('top[0].shape:{} top[1].shape:{}'.format(top[0].data.shape, top[1].data.shape))
            print ('top[0].data:{}'.format(str(np.array(top[0].data))))
            print ('top[1].data:{}'.format(str(np.array(top[1].data))))

    def backward(self, top, propagate_down, bottom):
        # 如果反向传播开着，则初始化diff（loss）为0，然后对应坐标赋值对应的 loss
        bottom[0].diff[...] = 0
        if propagate_down[0] and self.count != 0:
            bottom[0].diff[self.valid_index] = top[0].diff[0:self.count]
        bottom[1].diff[...] = 0
        if propagate_down[1] and self.count != 0:
            bottom[1].diff[self.valid_index] = top[1].diff[0:self.count]
'''


'''
仅仅保留 正样本(1) 部分样本(2)
用于计算  ROI 损失
bottom[pred, label]
'''
'''
class clslosslayer(caffe.Layer):
    def setup(self, bottom, top):
        if len(bottom) != 2:
            raise Exception("Need 2 Inputs")

    def reshape(self, bottom, top):
        if True:
            print ('clslosslayer')
        if False:
            print ('len(bottom):{} len(top):{}'.format(len(bottom), len(top)))
            print ( bottom[0].data.shape)
            print ( bottom[1].data.shape)
        # 获取训练label
        predlabel = np.array(bottom[0].data)
        label = np.array(bottom[1].data, dtype=np.int)

        self.batch_size = predlabel.shape[0]
        self.label_num = predlabel.shape[1]

        # 获取有效样本（正负）坐标
        self.valid_index = np.where((label==0)|(label==1))[0]
        self.count = len(self.valid_index)
        if self.count == 0:
            print ('!!! valid count is 0 label:{}'.format(label))
        else:
            print ('valid count is:{}'.format(self.count))
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss 值
        top[0].reshape(1)

        #
        self.batch_size = bottom[0].data.shape[0]
        self.label_num = bottom[0].data.shape[1]
        self.prob = np.zeros((self.batch_size, self.label_num))

    def forward(self, bottom, top):
        if False:
            print ('len(bottom):{} len(top):{}'.format(len(bottom), len(top)))
            print ( bottom[0].data.shape)
            print ( bottom[1].data.shape)
             # 初始化所有输出为 0

        predlabel = np.array(bottom[0].data)[self.valid_index, ...]
        label = np.array(bottom[1].data)[self.valid_index, ...]

        self.diff[...] = 0
        top[0].data[...] = 0
        # 计算 diff 和 loss
        if self.count != 0:
            mmax = -100000000
            tmp = predlabel.reshape(self.batch_size, self.label_num).max(axis=1)
            tmp[np.where(tmp < mmax)] = mmax
            prob = predlabel[:, :, 0, 0] - tmp.reshape(self.batch_size, 1)
            prob_exp = np.exp(prob)
            tmp = prob_exp.sum(axis=1)
            prob = np.exp(prob) / tmp.reshape(self.batch_size, 1)
            tmpProb = prob[np.arange(self.batch_size), label]
            problog = np.log(tmpProb) * -1
            loss = np.sum(problog) / self.batch_size

            # 计算梯度
            diff = np.zeros_like(predlabel)
            # 选取所有label
            prob[np.arange(self.batch_size), label] = prob[np.arange(self.batch_size), label] - 1
            diff[:, :, 0, 0] = prob[...]

            self.diff[self.valid_index] = diff
            top[0].data[...] = loss

    def backward(self, top, propagate_down, bottom):
        # 如果反向传播开着，则初始化diff（loss）为0，然后对应坐标赋值对应的 loss
        for i in range(2):
            if not propagate_down[i] or self.count == 0:
                bottom[i].diff[...] = self.diff / bottom[i].num
'''


'''
仅仅使用 正样本(1) 部分样本(2)
用于计算  ROI 损失
bottom[pred, roi, label]
top：[validpred, validroid]
'''


class roilosslayer(caffe.Layer):
    def setup(self, bottom, top):
        if len(bottom) != 3:
            raise Exception("Need 2 Inputs")

    def reshape(self, bottom, top):
        if False:
            print ('poilosslayer')
        if False:
            print ('len(bottom):{} len(top):{}'.format(len(bottom), len(top)))
            print ( bottom[0].data.shape)
            print ( bottom[1].data.shape)
            print ( bottom[2].data.shape)
        # 获取训练label
        label = np.array(bottom[2].data, dtype=np.int)
        # 获取有效样本（正负）坐标
        self.valid_index = np.where((label==1)|(label==2))[0]
        self.count = len(self.valid_index)
        if False:
            strlabel = (np.array(label, dtype=np.str).reshape(-1).tolist())
            print ('lable:{}'.format(strlabel))
        # if self.count == 0:
        #     print ('!!! valid count is 0 ')
        # else:
        #     print ('valid count is:{}'.format(self.count))
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss 值
        top[0].reshape(1)

    def forward(self, bottom, top):
        if False:
            print ('len(bottom):{} len(top):{}'.format(len(bottom), len(top)))
            print ( bottom[0].data.shape)
            print ( bottom[1].data.shape)
             # 初始化所有输出为 0

        predroi = np.array(bottom[0].data)[self.valid_index]
        roi = np.array(bottom[1].data)[self.valid_index]

        self.diff[...] = 0
        top[0].data[...] = 0
        # 计算 diff 和 loss
        if self.count != 0:
            self.diff[self.valid_index] = predroi - roi.reshape(predroi.shape)
            top[0].data[...] = np.sum(self.diff ** 2) / self.count / 2.
            # top[0].data[...] = np.sum(self.diff ** 2) / bottom[0].num / 2.

    def backward(self, top, propagate_down, bottom):
        # 如果反向传播开着，则初始化diff（loss）为0，然后对应坐标赋值对应的 loss
        for i in range(2):
            if not propagate_down[i] or self.count == 0:
                continue
            if i == 0:
                sign = 1
            else:
                sign = -1
            bottom[i].diff[...] = sign * self.diff / self.count

'''
仅仅使用 特征点样本(3)
用于计算 特征点损失

'''


class ptslosslayer(caffe.Layer):
    def setup(self, bottom, top):
        if len(bottom) != 3:
            raise Exception("Need 2 Inputs")

    def reshape(self, bottom, top):
        if False:
            print ('ptslosslayer')
        if False:
            print ('len(bottom):{} len(top):{}'.format(len(bottom), len(top)))
            print (bottom[0].data.shape)
            print (bottom[1].data.shape)
            # print (bottom[2].data.shape)
        # 获取训练label
        label = np.array(bottom[2].data, dtype=np.int)
        # 获取有效样本（正负）坐标
        self.valid_index = np.where((label == 3))[0]
        self.count = len(self.valid_index)
        # if self.count == 0:
        #     print ('!!! valid count is 0')
        # else:
        #     print ('valid count is:{}'.format(self.count))
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss 值
        top[0].reshape(1)

    def forward(self, bottom, top):
        if False:
            print ('len(bottom):{} len(top):{}'.format(len(bottom), len(top)))
            print (bottom[0].data.shape)
            print (bottom[1].data.shape)

        predpts = np.array(bottom[0].data)[self.valid_index]
        pts = np.array(bottom[1].data)[self.valid_index]

        self.diff[...] = 0
        top[0].data[...] = 0
        # 计算 diff 和 loss
        if self.count != 0:
            # 获得两眼距离
            delX = ((pts[:, 84] + pts[:, 90]) / 2) - ((pts[:, 72] + pts[:, 78]) / 2)
            delY = ((pts[:, 85] + pts[:, 91]) / 2) - ((pts[:, 73] + pts[:, 79]) / 2)
            self.interOc = (1e-6 + (delX * delX + delY * delY) ** 0.5).T  # Euclidain distance
            diff = (predpts - pts.reshape(predpts.shape)).T
            self.diff[self.valid_index] = (diff / self.interOc).T
            top[0].data[...] = np.sum(self.diff ** 2) / bottom[0].num / 2.  # Loss is scalar

    def backward(self, top, propagate_down, bottom):
        # 如果反向传播开着，则初始化diff（loss）为0，然后对应坐标赋值对应的 loss
        for i in range(2):
            if not propagate_down[i] or self.count == 0:
                continue
            if i == 0:
                sign = 1
            else:
                sign = -1
            bottom[i].diff[...] = sign * self.diff / bottom[i].num
