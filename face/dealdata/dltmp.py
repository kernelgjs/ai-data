#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import traceback
import cv2
import dlib
import time
import numpy as np


if __name__ == '__main__':
    imgpath = 'pic/000001.png'

    # img = np.arange(12).reshape(2,2,3)
    a = np.ones((2,2,3), dtype=np.float32)
    b = np.array([2,3,4], dtype=np.float32)
    c = np.array([2], dtype=np.float32)
    print a.transpose(2,0,1).shape
    bb = a - b
    cc = bb/c
    print c