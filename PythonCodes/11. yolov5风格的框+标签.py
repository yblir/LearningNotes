# -*- coding: utf-8 -*-
# @Time    : 2024/3/8 10:36
# @Author  : yblir
# @File    : 11. yolov5风格的框+标签.py
# explain  : 从yolov5源码中抽取, 提炼成绘图模块, 用于测试其他检测模型的输出
# =======================================================
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


def xywh2xyxy(x):
    """Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right."""
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        """
        Initializes the Colors class with a palette derived from Ultralytics color scheme, converting hex codes to RGB.

        Colors derived from `hex = matplotlib.colors.TABLEAU_COLORS.values()`.
        """
        hexs = (
            "FF3838", "FF9D97", "FF701F", "FFB21D", "CFD231", "48F90A", "92CC17", "3DDB86", "1A9334", "00D4BB",
            "2C99A8", "00C2FF", "344593", "6473FF", "0018EC", "8438FF", "520085", "CB38FF", "FF95C8", "FF37C7",
        )
        self.palette = [self.hex2rgb(f"#{c}") for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        """Returns color from palette by index `i`, in BGR format if `bgr=True`, else RGB; `i` is an integer index."""
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):
        """Converts hexadecimal color `h` to an RGB tuple (PIL-compatible) with order (R, G, B)."""
        return tuple(int(h[1 + i: 1 + i + 2], 16) for i in (0, 2, 4))


def draw_images(image, boxes, classes, scores, colors, xyxy=True):
    """
    对单张图片进行多个框的绘制
    Args:
        image: pillow与numpy格式都行, 反正都会转成pillow格式,h,w,c
        boxes: tensor与numpy格式都行, 最后都会转成numpy格式
        xyxy: 默认是xyxy格式, 如果为False,就是xywh格式,需要进行一次格式转换
        classes: 每个框的类别,list, 与boxes框对应
        scores: 每个预测类别得分,list, 与boxes框对应
        colors: 每个类别框颜色
    Returns:
    """
    if isinstance(image, torch.Tensor):
        image = Image.fromarray(image.cpu().float().numpy()).convert("RGB")
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image).convert("RGB")
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()

    # 转换box格式
    if not xyxy:
        boxes = xywh2xyxy(boxes)
    
    # 设置字体,pillow 绘图环节
    font = ImageFont.truetype(font=r'E:\SourceCodes\yolox-pytorch-main\model_data\simhei.ttf',
                              size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    # 多次画框的次数,根据图片尺寸不同,把框画粗
    thickness = max((image.size[0] + image.size[1]) // 300, 1)
    draw = ImageDraw.Draw(image)

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        color = colors[i]
        
        label = '{}:{:.2f}'.format(classes[i], scores[i])
        # 提取标签框宽高
        tx1, ty1, tx2, ty2 = font.getbbox(label)
        tw, th = tx2 - tx1, ty2 - tx1

        # 这是标签框起始位置
        text_x1y1 = np.array([x1, y1 - th]) if y1 - th >= 0 else np.array([x1, y1 + 1])

        # 在目标框周围偏移几个像素多画几次, 让边框变粗
        for j in range(thickness):
            draw.rectangle((x1 + j, y1 + j, x2 - j, y2 - j), outline=color)

        # 画标签
        draw.rectangle((text_x1y1[0], text_x1y1[1], text_x1y1[0] + tw, text_x1y1[1] + th), fill=color)
        draw.text(text_x1y1, label, fill=(0, 0, 0), font=font)

    return image


if __name__ == '__main__':
    labels = [1, 1, 2, 4]
    scores = [0.7936255931854248, 0.7047973871231079, 0.5894179940223694, 0.5288568735122681]
    boxes = [[337.5139, 205.9804, 600.7680, 365.0697],
             [318.4250, 289.5251, 638.9167, 477.7193],
             [367.7424, 306.2967, 398.3298, 316.3928],
             [188.3082, 19.3642, 200.0569, 47.6136]]

    colors_ = Colors()
    colors = [colors_(cls) for cls in labels]

    image = draw_images(
            image=Image.open(r"E:\interesting\GLIP-main\docs\demo.jpg"),
            boxes=boxes,
            classes=labels,
            scores=scores,
            colors=colors
    )

    image.show()
