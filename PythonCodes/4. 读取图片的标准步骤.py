import os
import cv2
from PIL import Image, ImageFile
import numpy as np
from loguru import logger

# 防止出现image file is truncated这种问题
ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_img_back(img_path):
    '''
    输入图片路径,返回bgr格式的numpy矩阵
    :param img_path:
    :return:
    '''
    if not os.path.exists(img_path):  # 检查图片路径是否存在
        raise ValueError('img_path is not exists')

    img = cv2.imread(img_path)
    if img is None:  # 读取失败,就用Image再读一次
        try:  # PIL因版本问题,也有很多坑!
            img = Image.open(img_path)
        except Exception as e:
            raise ValueError(f'Image.open() read image failure,error:{e}')

        # 检查图片格式
        if img.mode != 'RGB':
            img = img.convert('RGB')
        # 转换为numpy格式
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    else:  # 读取成功,再校验下是否为bgr
        if len(img.shape) == 2:  # 灰度图,转为bgr
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif len(img.shape) != 3:  # 通道数不为3,不知什么问题,反正不是正常图片
            raise ValueError('image channel not 3')
        elif img.shape[2] == 4:  # png转为bgr
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        else:
            raise ValueError('image error !')
    return img


def read_img(img_path):
    '''
    输入图片路径,返回bgr格式的numpy矩阵,或None
    :param img_path:
    :return:
    '''
    if not os.path.exists(img_path):  # 检查图片路径是否存在
        # 如果路径不存在,那么大概率其他图片路径也有问题,直接抛出异常
        raise ValueError('img_path is not exists')

    try:  # PIL因版本问题,有很多坑! 但cv2坑更多,所以还是优先使用PIL
        img = Image.open(img_path)
    except Exception as e1:
        logger.info(f'Image.open() read image failure:{img_path},error:{e1}, now use cv2 read')
        try:
            img = cv2.imread(img_path)
        except Exception as e2:
            logger.error(f'Image,cv2 read all failure: {img_path},error:{e2}')
            # 图片读取失败,直接返回None
            return None

    if isinstance(img, np.ndarray):  # 如果是cv2读取成功
        if len(img.shape) == 2:  # 灰度图,转为bgr
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif len(img.shape) != 3:  # 通道数不为3,不知什么问题,反正不是正常图片
            logger.error('image channel not 3')
            # 当前图片错误,返回None, 这样函数就有3个return出口,这样写是否规范?
            return None
        elif img.shape[2] == 4:  # png转为bgr
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    else:  # 如果是Image读取成功
        # 检查图片格式
        if img.mode != 'RGB':
            img = img.convert('RGB')
        # 转换为numpy格式
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    return img


if __name__ == '__main__':
    path = r'C:\Users\FH\Desktop\style_bases\d53a081f5c8a5f5072f7db876ed6c6bb.gif'
    img2 = read_img(path)
    print(img2.shape)
