import cv2
import numpy as np
from pathlib2 import Path
from PIL import Image
import warnings

warnings.filterwarnings("ignore")


def letterbox_image(image, size):
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image


if __name__ == '__main__':
    boxes = [[395.68646240234375,
              228.71900939941406,
              588.1580200195312,
              522.3549194335938,
              ],
             [46.462135314941406,
              221.62551879882812,
              273.6243591308594,
              532.8731689453125,
              ]]
    img = cv2.imread(r'E:\GitHub\TensorRTModelDeployment\imgs\2007_000925.jpg')
    # img = Image.open(r'E:\GitHub\TensorRTModelDeployment\imgs\2007_000925.jpg')
    img = letterbox_image(Image.fromarray(img), (640, 640))
    img = np.asarray(img)
    # img=np.ascontiguousarray(img[...,::-1])
    # img.astype('uint8')
    for box in boxes:
        box = [int(i) for i in box]
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1)
    cv2.imshow('aa', img)
    cv2.waitKey()
