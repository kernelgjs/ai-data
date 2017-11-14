#!/usr/bin/python
# coding:utf-8

import sys
import caffe.proto.caffe_pb2 as caffe_pb2

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print ('usage: cfmd2txt caffemodel(path) txt(path)')
        exit()

    cfmdname = sys.argv[1]
    txtname = sys.argv[2]

    model = caffe_pb2.NetParameter()
    with open(cfmdname, 'r') as f:
        model.ParseFromString(f.read())
    with open(txtname, 'w') as f:
        f.write(str(model))
