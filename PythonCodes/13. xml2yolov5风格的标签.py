# -*- coding: utf-8 -*-
# @Time    : 2024/3/29 9:14
# @Author  : yblir
# @File    : 13. xml2yolov5风格的标签.py
# explain  : 
# =======================================================
import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

sets = ['train', 'test', ]
classes = ["fire", "smoke"]  # 这里输入你的数据集类别


def convert(size, box):  # 读取xml文件中的数据，xywh
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


def convert_annotation(image_id):
    in_file = open('Annotations/%s.xml' % (image_id), encoding='utf-8')  # 这里是读取xml的文件夹
    out_file = open('Annotations/%s.txt' % (image_id), 'w', encoding='utf-8')  # 存入txt文件的文件夹
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


wd = getcwd()

for image_set in sets:
    # if not os.path.exists('labels/'):
    #     os.makedirs('labels/')
    image_ids = open('ImageSets/Main/%s.txt' % (
        image_set)).read().strip().split()  # 读取train.txt或者test.txt从而找到每个xml文件的文件名，这里的train.txt中仅包含文件名，不包好路径。
    # list_file = open('%s.txt'%(image_set), 'w')
    for image_id in image_ids:
        # list_file.write('/root/object-detection/yolov5-master/data/police_obj/images/%s.jpg\n'%(image_id))#从写train.txt或者test.txt文件，把图片文件的绝对路径写入，方便读取图片
        convert_annotation(image_id)
    # list_file.close()
