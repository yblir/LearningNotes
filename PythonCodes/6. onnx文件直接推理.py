# -*- coding: utf-8 -*-
# @Time    : 2023/12/2 11:16
# @Author  : yblir
# @File    : 6. onnx文件直接推理.py
# explain  : 
# =======================================================
import cv2
import numpy as np
import onnxruntime
import torch

session = onnxruntime.InferenceSession("xxx.onnx")
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name


def resize_img(img, target_size):
    return img


if __name__ == '__main__':
    img = cv2.imread(r"C:\Users\Administrator\Desktop\app_fakes\20231102142902.jpg")
    img = resize_img(img, 224)
    # BGR->RGB
    img = img[..., ::-1]
    # 变换通道并使内存连续, 必须有 ascontiguousarray 连续操作,不然tensorrt推理不对
    img = np.ascontiguousarray(img.transpose(2, 0, 1))
    # (3,224,224) -> (1,3,224,224)
    img = img[None]

    # output = session.run([output_name], {input_name: img})
    # 让内部程序自己判断输出?
    output = session.run(None, {input_name: img})
    print(output)
    # print(img.shape)
