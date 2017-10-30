#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print ('usage: python getAllFile.py searchDir storeFile')
        exit()
    searchDir = sys.argv[1]
    storeFile = sys.argv[2]
    allfiles = []
    curdir = os.getcwd()
    for root, dirs, files in os.walk(searchDir):
        for f in files:
            if f.endswith(('.jpg','.png')):
                allfiles.append(os.path.join(curdir,root,f)+'\n')
    with open(storeFile, 'w') as f:
        f.writelines(sorted(allfiles))
