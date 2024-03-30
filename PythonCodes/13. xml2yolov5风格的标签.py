# -*- coding: utf-8 -*-
# @Time    : 2024/3/29 9:14
# @Author  : yblir
# @File    : 13. xml2yolov5风格的标签.py
# explain  : 
# =======================================================
import shutil
import xml.etree.ElementTree as ET

# import pickle
import os
from os import listdir, getcwd
# from os.path import join
import glob

classes = ["person", "helmet"]


def convert(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


def convert_annotation(image_name):
    # in_file = open(image_name[:-3] + 'xml')  # xml文件路径
    out_file = open('./work/lables/train_v/' + image_name[:-3] + 'txt', 'w')  # 转换后的txt文件存放路径
    with open('./work/' + image_name[:-3] + 'xml', "r") as f:
        xml_text = f.read()
        root = ET.fromstring(xml_text)
        f.close()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls not in classes:
            print(cls)
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


wd = getcwd()

if __name__ == '__main__':
    os.makedirs('./work/lables/train_v', exist_ok=True)
    os.makedirs("./work/images/train_v", exist_ok=True)

    for image_path in glob.glob("./work/*.jpg"):  # 每一张图片都对应一个xml文件这里写xml对应的图片的路径
        shutil.copy(image_path, "./work/images/train_v")
        image_name = image_path.split(os.sep)[-1]
        convert_annotation(image_name)
