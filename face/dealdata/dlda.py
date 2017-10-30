#!/usr/bin/python
# -*- coding: utf-8 -*-


from PIL import Image, ImageEnhance, ImageOps, ImageFile
import numpy as np
import sys
import os
import cv2


def pil2cv(img):
    t = np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0], 3)  # h * w * c
    return cv2.cvtColor(t, cv2.COLOR_RGB2BGR)

def cv2pil(arr):
    t = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(t, 'RGB')

def drawimg(img, dm=None, lm=None, name=''):
    if lm is not None:
        tlm = np.array(lm, dtype=np.int)
        tlm = tlm.reshape(68, 2)
        for i, point in enumerate(tlm):
            point = tuple(point)
            cv2.circle(img, point, 2, (255, 255, 0), -1)
    # cv2.putText(img, str(i+1), point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)
    if name == '':
        cl = (255, 0, 0)
    elif name == '1':
        cl = (0, 255, 0)
    elif name == '2':
        cl = (0, 0, 255)
    else:
        cl = (255, 255, 0)

    if dm is not None:
        cv2.rectangle(img, (dm[0], dm[1]), (dm[2], dm[3]), cl)
    # cv2.resizeWindow(name)
    # cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, img)
    # cv2.waitKey(0)


class pda: # PILDataAugmentation

    def __init__(self):
        pass

    @staticmethod
    def openImage(image):
        return Image.open(image)

    @staticmethod
    def randomColor(image):
        """
        对图像进行颜色抖动
        :param image: PIL的图像image
        :return: 有颜色色差的图像image
        """
        random_factor = np.random.randint(0, 31) / 10.  # 随机因子
        color_image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度
        random_factor = np.random.randint(10, 21) / 10.  # 随机因子
        brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # 调整图像的亮度
        random_factor = np.random.randint(10, 21) / 10.  # 随机因1子
        contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # 调整图像对比度
        random_factor = np.random.randint(0, 31) / 10.  # 随机因子
        return ImageEnhance.Sharpness(contrast_image).enhance(random_factor)  # 调整图像锐度

    @staticmethod
    def saveImage(image, path):
        image.save(path)



class cda:
    def __init__(self):
        pass

    @staticmethod
    def openImage(imagepath):
        return cv2.imread(imagepath, cv2.IMREAD_COLOR)

    @staticmethod
    def saveImage(filename, image):
        return cv2.imwrite(filename, image)


    @staticmethod
    def noise(image):
        timg = image.copy()
        # 随机生成5000个椒盐噪声
        for i in range(5000):
            x = np.random.randint(0, timg.shape[0])
            y = np.random.randint(0, timg.shape[1])
            timg[x, y, :] = 255
        return timg

    # 水平变换 需要变换 包围框 关键点
    @staticmethod
    def flip(image, dm, lm):
        timg = image.copy()
        tdm = dm.copy()
        tlm = lm.copy()
        timg = cv2.flip(timg, 1)
        # 镜像后 包围框 左上角,右下角 发生变换
        fdm = tdm.copy()
        fdm[0] = timg.shape[1]-tdm[2]
        fdm[2] = timg.shape[1]-tdm[0]
        # 镜像后 面部坐标及顺发生变化
        tlm[0::2]=timg.shape[1]-tlm[0::2]
        '''
        ## 面部 17 -> 1
        ## 双眉 27 -> 18
        ## 鼻梁 28 -> 31
        ## 鼻下 36 -> 32
        ## 左眼 46 -> 43 48 47
        ## 右眼 40 -> 37 42 41
        ## 外嘴唇 55 -> 49 60 -> 56
        ## 内嘴唇 65 -> 61 68 -> 66
        ## 根据以上注释顺序 生成新的特征点顺序
        '''
        tlm = tlm.reshape(68, 2)
        flm = np.zeros_like(tlm)
        flm[0:17, ] = tlm[0:17, ][::-1]
        flm[17:27, ] = tlm[17:27, ][::-1]
        flm[27:31, ] = tlm[27:31, ]
        flm[31:36, ] = tlm[31:36, ][::-1]
        flm[36:40, ] = tlm[42:46, ][::-1]
        flm[40:42, ] = tlm[46:48, ][::-1]
        flm[42:46, ] = tlm[36:40, ][::-1]
        flm[46:48, ] = tlm[40:42, ][::-1]
        flm[48:55, ] = tlm[48:55, ][::-1]
        flm[55:60, ] = tlm[55:60, ][::-1]
        flm[60:65, ] = tlm[60:65, ][::-1]
        flm[65:68, ] = tlm[65:68, ][::-1]

        return timg, fdm, flm.reshape(-1)

    # 移动包围框
    @staticmethod
    def shiftbox(image, dm):
        timg = image.copy()
        tdm = dm.copy()
        random_left_up = np.random.randint(0, 3)  # 0 左右移动， 1上下移动, 2 左右上下同时移动
        random_factor = np.random.randint(-10, 11) / 100.0  # 随机因子 偏移范围[-0.2, 0.2]
        deta = int((dm[2]-dm[0])*random_factor)  # 移动距离
        if random_left_up == 0:
            tdm[0::2] += deta
        elif random_left_up == 1:
            tdm[1::2] += deta
        else:
            tdm[:] += deta
        # 判断坐标是否超出图像
        if tdm[0]<0 or tdm[1]<0 or tdm[2]>timg.shape[1] or tdm[3]>timg.shape[0]:
            return dm
        return tdm


    # 缩放包围框
    @staticmethod
    def zoombox(image, dm):
        timg = image.copy()
        tdm = dm.copy()
        random_factor = np.random.randint(-10, 11) / 100.0  # 随机因子 偏移范围[-0.2, 0.2]
        deta = int((dm[2]-dm[0])*random_factor)  # 移动距离
        tdm[0:2]-=deta
        tdm[2:4]+=deta
        # 判断坐标是否超出图像
        if tdm[0]<0 or tdm[1]<0 or tdm[2]>timg.shape[1] or tdm[3]>timg.shape[0]:
            return dm
        return tdm

    # 获取最大正方形,用于裁剪不是1:1的图片 dm lm 变化
    @staticmethod
    def maxboximg(image, dm, lm):
        timg = image.copy()
        tdm = dm.copy()
        tlm = lm.copy()
        imgh = timg.shape[0]
        imgw = timg.shape[1]
        if imgh < imgw:
            centerx = (tdm[0]+tdm[2])/2
            if centerx-imgh/2 < 0: # 往右边延伸多些
                l = 0
                r = imgh
            elif centerx+imgh/2 > imgw:  # 往左边边延伸多些
                l = imgw -imgh
                r = imgw
            else:  # 两边相等延伸
                l = centerx - imgh/2
                r = centerx + imgh/2
            t = 0
            b = imgh
            tdm[0::2]-=l
            tlm[0::2]-= l

        else:
            centery = (tdm[1]+tdm[3])/2
            if centery-imgw/2 < 0: # 往上边延伸多些
                t = 0
                b = imgw
            elif centery+imgw/2 > imgh:  # 往下边边延伸多些
                t = imgh -imgw
                b = imgh
            else:  # 两边相等延伸
                t = centery - imgw / 2
                b = centery + imgw / 2
            l = 0
            r = imgw
            tdm[1::2]-=t
            tlm[1::2]-= t
        timg = timg[t:b+1, l:r+1, ]

        # 缩放到 600 × 600
        timg = cv2.resize(timg, (600, 600), interpolation=cv2.INTER_CUBIC)
        # 所有坐标按比例缩放
        tdm = np.array(tdm, dtype=np.float32)
        radio = 600.0/(r-l)
        tdm*=radio
        tdm = np.array(tdm, dtype=np.int)
        tlm*=radio

        return timg, tdm, tlm




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


def wirteimgdmlmlist(mapimgdmlm, storename):
    imglmdmlist = []
    for imagepath, dmlm in mapimgdmlm.items():
        dm = np.array(dmlm[0], dtype=np.str)
        lm = np.array(dmlm[1], dtype=np.str)
        line = ','.join([imagepath]+dm.tolist()+lm.tolist())+'\n'
        imglmdmlist.append(line)
    with open(storename, 'w') as tf:
        tf.writelines(imglmdmlist)



# /home/gjs/data/orgdata/ibug-68/300W/01_Indoor/indoor_001.png
imgpath = '/home/gjs/data/orgdata/ibug-68/300W/01_Indoor/indoor_001.png'
imgpath = '/home/gjs/data/orgdata/ibug-68/300W/01_Indoor/indoor_002.png'
if __name__ == '__main__':
    # img dm lm 都以 np 表示
    # img: BGR
    # dm: l t r b
    # !!! 建议分别基于 600*600 水平变换
    if len(sys.argv) != 7:
        print ('usage: python dlda.py method start end imgdmlmlist.txt storedir muti\n'
               'method:crop flip shift zoom color noise ')
        exit()
    method = sys.argv[1]
    start = int(sys.argv[2])
    end = int(sys.argv[3])
    imgdmlmlist = sys.argv[4]
    storedir = sys.argv[5]
    muti = int(sys.argv[6])

    # start = 0
    # end = 10
    # imgdmlmlist = '/home/gjs/data/orgdata/ibug600/imgdmlmlist.txt'
    # storedir = '/home/gjs/data/orgdata/ibug600flipx/'

    mapimgdmlm = readimgdmlmlist(imgdmlmlist, start, end)
    mapnewimgdmlm = {}
    scount = 0

    '''
    裁剪为 600 × 600
    '''
    if 'crop' == method:
        for imgpath, dmlm in mapimgdmlm.items():
            img = cda.openImage(imgpath)
            dm = dmlm[0]
            lm = dmlm[1]
            scount+=1
            # basename = os.path.basename(imgpath)
            # name, ext = os.path.splitext(basename)
            name = '%06d' % scount + '.png'
            sotrefilename = os.path.join(storedir, name)
            timg, tdm, tlm = cda.maxboximg(img, dm, lm)
            mapnewimgdmlm[sotrefilename] = (tdm, tlm)
            cda.saveImage(sotrefilename, timg)
            print ('deal imgpath:{} scount:{} sotrename:{}'.format(imgpath, scount, sotrefilename))
            sys.stdout.flush()
        wirteimgdmlmlist(mapnewimgdmlm, os.path.join(storedir, 'imgdmlmlist.txt'))
        print ('deal crop over')
        exit()
    '''
    镜像 flip
    '''
    if 'flip' == method:
        for imgpath, dmlm in mapimgdmlm.items():
            img = cda.openImage(imgpath)
            dm = dmlm[0]
            lm = dmlm[1]
            scount+=1
            basename = os.path.basename(imgpath)
            # name, ext = os.path.splitext(basename)
            sotrefilename = os.path.join(storedir, method+basename)
            timg, tdm, tlm = cda.flip(img, dm, lm)
            mapnewimgdmlm[sotrefilename] = (tdm, tlm)
            cda.saveImage(sotrefilename, timg)
            print ('deal imgpath:{} scount:{} sotrename:{}'.format(imgpath, scount, sotrefilename))
            sys.stdout.flush()
        wirteimgdmlmlist(mapnewimgdmlm, os.path.join(storedir, 'imgdmlmlist.txt'))
        print ('deal flip over')
        exit()
    '''
    颜色扰动
    '''
    if 'color' == method:
        for imgpath, dmlm in mapimgdmlm.items():
            img = pda.openImage(imgpath)
            dm = dmlm[0]
            lm = dmlm[1]
            scount+=1
            for i in range(muti):
                basename = os.path.basename(imgpath)
                name, ext = os.path.splitext(basename)
                sotrefilename = os.path.join(storedir, method+name+'-'+str(i)+ext)
                timg = pda.randomColor(img)
                mapnewimgdmlm[sotrefilename] = (dm, lm)
                pda.saveImage(timg, sotrefilename)
                print ('deal imgpath:{} scount:{} sotrename:{}'.format(imgpath, scount, sotrefilename))
                sys.stdout.flush()
        wirteimgdmlmlist(mapnewimgdmlm, os.path.join(storedir, 'imgdmlmlist.txt'))
        print ('deal random color over')
        exit()
    '''
    包围框扰动：偏移
    '''
    if 'shift' == method:
        for imgpath, dmlm in mapimgdmlm.items():
            img = cda.openImage(imgpath)
            dm = dmlm[0]
            lm = dmlm[1]
            scount+=1
            for i in range(muti):
                basename = os.path.basename(imgpath)
                name, ext = os.path.splitext(basename)
                sotrefilename = os.path.join(storedir, method+name+'-'+str(i)+ext)
                tdm = cda.shiftbox(img, dm)
                mapnewimgdmlm[sotrefilename] = (tdm, lm)
                cda.saveImage(sotrefilename, img)
                print ('deal imgpath:{} scount:{} sotrename:{}'.format(imgpath, scount, sotrefilename))
                sys.stdout.flush()
        wirteimgdmlmlist(mapnewimgdmlm, os.path.join(storedir, 'imgdmlmlist.txt'))
        print ('deal box shift over')
        exit()
    '''
    包围框扰动：缩放
    '''
    if 'zoom' == method:
        for imgpath, dmlm in mapimgdmlm.items():
            img = cda.openImage(imgpath)
            dm = dmlm[0]
            lm = dmlm[1]
            scount+=1
            for i in range(muti):
                basename = os.path.basename(imgpath)
                name, ext = os.path.splitext(basename)
                sotrefilename = os.path.join(storedir, method+name+'-'+str(i)+ext)
                tdm = cda.zoombox(img, dm)
                mapnewimgdmlm[sotrefilename] = (tdm, lm)
                cda.saveImage(sotrefilename, img)
                print ('deal imgpath:{} scount:{} sotrename:{}'.format(imgpath, scount, sotrefilename))
                sys.stdout.flush()
        wirteimgdmlmlist(mapnewimgdmlm, os.path.join(storedir, 'imgdmlmlist.txt'))
        print ('deal box zoom over')
        exit()
    '''
    椒盐噪声
    '''
    if 'noise' == method:
        for imgpath, dmlm in mapimgdmlm.items():
            img = cda.openImage(imgpath)
            dm = dmlm[0]
            lm = dmlm[1]
            scount+=1
            for i in range(muti):
                basename = os.path.basename(imgpath)
                name, ext = os.path.splitext(basename)
                sotrefilename = os.path.join(storedir, method+name+'-'+str(i)+ext)
                timg = cda.noise(img)
                mapnewimgdmlm[sotrefilename] = (dm, lm)
                cda.saveImage(sotrefilename, timg)
                print ('deal imgpath:{} scount:{} sotrename:{}'.format(imgpath, scount, sotrefilename))
                sys.stdout.flush()
        wirteimgdmlmlist(mapnewimgdmlm, os.path.join(storedir, 'imgdmlmlist.txt'))
        print ('deal noise over')
        exit()

    print('unsport method')