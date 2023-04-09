import csv
import cv2
import numpy as np
from pathlib2 import Path
import os
import shutil


def read_coord_from_csv():
    csv_path = Path(r'F:\cartoon_data\personai_icartoonface_dettrain\icartoonface_dettrain.csv')
    img_dir_path = Path(r'F:\cartoon_data\personai_icartoonface_dettrain\icartoonface_dettrain')

    with open(str(csv_path), 'r', encoding='utf-8') as f:
        data = f.readlines()

    for item in data:
        info = item.split(',')
        img_path = img_dir_path / info[0]
        coord = [int(i.strip()) for i in info[1:]]
        # x1y1  x2y2 x是水平方向 传入必须是元祖,不能是列表
        point1 = (coord[0], coord[1])
        point2 = (coord[2], coord[3])

        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        print(img.shape)
        cv2.rectangle(img, point1, point2, (0, 255, 0), thickness=2)
        save_path = img_dir_path.parent / 'box_images' / info[0]
        cv2.imwrite(str(save_path), img)
        # cv2.imshow('aaa', img)
        # cv2.waitKey()
        # break


def write_coord_from_txt():
    '''
    :return:
    '''
    txt_path = Path(r'F:\cartoon_data\personai_icartoonface_dettrain\train_data2\train_data\labels\train2022')
    img_dir_path = Path(r'F:\cartoon_data\personai_icartoonface_dettrain\icartoonface_dettrain')

    for item in txt_path.iterdir():
        with open(str(item), 'r', encoding='utf-8') as f:
            data = f.readlines()
        img_path = img_dir_path / item.stem
        img = cv2.imread(str(img_path) + '.jpg', cv2.IMREAD_UNCHANGED)
        for xy in data:
            info = xy.split(' ')

            coord = [int(i.strip()) for i in info[1:]]
            # x1y1  x2y2 x是水平方向 传入必须是元祖,不能是列表
            point1 = (coord[0], coord[1])
            point2 = (coord[2], coord[3])
            cv2.rectangle(img, point1, point2, (0, 255, 0), thickness=2)

        save_path = img_dir_path.parent / 'box_images' / item.stem
        cv2.imwrite(str(save_path) + '.jpg', img)


def csv_to_txt():
    '''
    把csv中文件写入txt中
    :return:
    '''
    csv_path = Path(r'F:\cartoon_data\personai_icartoonface_dettrain\icartoonface_dettrain.csv')
    img_dir_path = Path(r'F:\cartoon_data\personai_icartoonface_dettrain\icartoonface_dettrain')
    train_data = csv_path.parent / 'train_data' / 'images' / 'train2022'
    label_data = csv_path.parent / 'train_data' / 'labels' / 'train2022'
    os.makedirs(str(train_data), exist_ok=True)
    os.makedirs(str(label_data), exist_ok=True)

    with open(str(csv_path), 'r', encoding='utf-8') as f:
        data = f.readlines()

    for i, item in enumerate(data):
        info = item.split(',')
        img_path = img_dir_path / info[0]
        coord = np.array([float(i.strip()) for i in info[1:]])
        coord = xyxy2xywh(coord)
        img_name = img_path.stem

        # shutil.copy2(str(img_path), str(train_data / img_path.name))
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(str(train_data.parent / 'wrong_imgs' / img_path.name), img)
            # shutil.copy2(str(img_path), str(train_data.parent/'wrong_imgs' / img_path.name))
            print(img.shape)
            print(img_path)

        h, w, _ = img.shape

        coord[..., [0, 2]] = coord[..., [0, 2]] / w
        coord[..., [1, 3]] = coord[..., [1, 3]] / h

        with open(str(label_data / img_name) + '.txt', 'a') as f:
            f.writelines('0 ' + str(coord.tolist())[1:-1].replace(', ', ' ') + '\n')

        if (i + 1) % 500 == 0:
            print(f'cur process: {i + 1}')


def xyxy2xywh(coord):
    '''
    把xyxy格式坐标转化为xywh格式.
    coord: x1,y1,x2,y2
    normalized: 是否归一化
    '''
    box_xy = (coord[..., 0:2] + coord[..., 2:4]) / 2
    box_wh = coord[..., 2:4] - coord[..., 0:2]

    coord[..., :2] = box_xy
    coord[..., 2:4] = box_wh

    return coord


def write_coord_from_txt2():
    '''
    从txt文件读取坐标,在图片上画框,并过滤小框
    :return:
    '''
    txt_path = Path(r'F:\cartoon_data\VOCdevkit\txt_out')
    img_dir_path = Path(r'F:\cartoon_data\VOCdevkit\VOC2012\JPEGImages')
    flag = False
    i = 0
    for item in txt_path.iterdir():
        i += 1
        with open(str(item), 'r', encoding='utf-8') as f:
            data = f.readlines()
        img_path = img_dir_path / item.stem
        img = cv2.imread(str(img_path) + '.jpg', cv2.IMREAD_UNCHANGED)
        for xy in data:
            if xy == '\n':  # 如果是空行,跳过
                continue
            info = xy.split(',')
            try:
                coord = [int(i.strip()) for i in info]
            except:
                print(item)
                raise
            # 判断框是否太小
            if coord[2] - coord[0] < 25 or coord[3] - coord[1] < 25:
                continue
            # x1y1  x2y2 x是水平方向 传入必须是元祖,不能是列表
            point1 = (coord[0], coord[1])
            point2 = (coord[2], coord[3])
            save_txt_path = txt_path.parent / 'txt_out2' / item.stem
            with open(str(save_txt_path) + '.txt', 'a') as f:
                f.write(xy)
            cv2.rectangle(img, point1, point2, (0, 255, 0), thickness=2)
            flag = True

        save_path = txt_path.parent / 'img_out2' / item.stem
        if flag is True:  # 只保存画框的图片
            cv2.imwrite(str(save_path) + '.jpg', img)
            flag = False
        if i % 200 == 0:
            print('当前进度 =', i)


if __name__ == '__main__':
    # csv_to_txt()
    # read_coord_from_csv()
    write_coord_from_txt2()
