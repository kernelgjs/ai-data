#!/usr/bin/python
# coding:utf-8

# 直接通过 proto 处理 caffemodel
# 1. 显示 txt（所有）
# 2. 显示 训练时 prototxt（供参考）
# 3. 修改模型 col2row major
# 4. 按照其他方式修改模型（ex：某个层的名字 等）

# 卷积层 计算 [http://caffe.berkeleyvision.org/tutorial/layers/convolution.html]
# in: n * c_i * h_i * w_i
# out: n * c_o * h_o * w_o, where h_o = (h_i + 2 * pad_h - kernel_h) / stride_h + 1 and w_o likewise.

# TODO all

import caffe.proto.caffe_pb2 as caffe_pb2      # 载入caffe.proto编译生成的caffe_pb2文件

# 载入模型
caffemodel_filename = '/home/gjs/work/self/mtcnn/ncnn/mtcnn/row-major/det2.caffemodel'
ZFmodel = caffe_pb2.NetParameter()        # 为啥是NetParameter()而不是其他类，呃，目前也还没有搞清楚，这个是试验的
f = open(caffemodel_filename, 'rb')
ZFmodel.ParseFromString(f.read())
f.close()

# noob阶段，只知道print输出
print ZFmodel.name
# print ZFmodel

with open('tmp.txt', 'w') as f:
    f.write(str(ZFmodel))
