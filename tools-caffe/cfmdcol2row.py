# coding:utf-8


import caffe.proto.caffe_pb2 as caffe_pb2      # 载入caffe.proto编译生成的caffe_pb2文件

# 载入模型
caffemodel_filename = '/home/chris/py-faster-rcnn/imagenet_models/ZF.v2.caffemodel'
ZFmodel = caffe_pb2.NetParameter()        # 为啥是NetParameter()而不是其他类，呃，目前也还没有搞清楚，这个是试验的
f = open(caffemodel_filename, 'rb')
ZFmodel.ParseFromString(f.read())
f.close()

# noob阶段，只知道print输出
print ZFmodel.name
print ZFmodel.input
