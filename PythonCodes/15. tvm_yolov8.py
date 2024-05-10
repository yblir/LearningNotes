# -*- coding: utf-8 -*-
# @File  : tvm_yolov8.py
# @Author: yblir
# @Time  : 2024-04-15 23:57
# @Explain: 
# ======================================================================================================================
import os.path
import random
import sys
import time
import cv2
import numpy as np
import torch
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import ops
from typing import List, Tuple, Union
from numpy import ndarray

import onnx
import numpy as np
import tvm.relay as relay
import tvm
from tvm.contrib import graph_executor

import tvm.auto_scheduler as auto_scheduler
from tvm.autotvm.tuner import XGBTuner
from tvm import autotvm


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


class YOLOV8DetectionInfer:
    def __init__(self, weights, conf_thres, iou_thres,
                 target, device, is_tvm=False, tvm_param="yolov8m-autotuning.json") -> None:
        self.imgsz = 640
        self.model = AutoBackend(weights, device=device)
        self.model.eval()
        self.names = self.model.names
        self.half = False
        self.conf = conf_thres
        self.iou = iou_thres
        self.color = {"font": (255, 255, 255)}

        colors_ = Colors()
        self.color.update({self.names[i]: colors_(i) for i in range(len(self.names))})
        self.target = target
        self.device = device

        self.tvm_param = tvm_param
        self.is_tvm = is_tvm
        # 是否使用tvm
        if self.is_tvm:
            # 简单优化
            # self.tvm_module = self.init_tvm_raw(weights)
            # 深度优化
            # self.tvm_module, _ = self.init_tvm_optimize(weights)
            self.tvm_module = None
        self.tvm_out_shape = (1, 84, 8400)
        self.tvm_input_name = "images"

    @staticmethod
    def save_tvm_lib(lib, save_path):
        path_lib = save_path + os.sep + "deploy_lib.tar"
        lib.export_library(path_lib)

    def load_tvm_lib(self, lib_path):
        # 重新加载模块
        loaded_lib = tvm.runtime.load_module(lib_path)
        dev = tvm.device(str(self.target), 0)
        module = graph_executor.GraphModule(loaded_lib["default"](dev))

        return module

    def init_tvm_raw(self, weights):
        print('tvm标准优化')
        onnx_model = onnx.load(weights)

        input_name = "images"
        shape_dict = {input_name: (1, 3, 640, 640)}
        # mod: relay表示的模型计算图, 相当于函数定义. params: 模型群众参数? 为什么我这里是空的?
        mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

        # 标准优化
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=self.target, params=params)

        dev = tvm.device(str(self.target), 0)
        module = graph_executor.GraphModule(lib["default"](dev))

        return module

    # 调优tvm加载的模型
    def init_tvm_optimize(self, weights):
        print('tvm深度优化')
        onnx_model = onnx.load(weights)

        input_name = "images"
        shape_dict = {input_name: (1, 3, 640, 640)}
        # 1.加载onnx模型,高级模型语言 Relay
        mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

        if not os.path.exists(self.tvm_param):
            number = 10  # number 指定将要测试的不同配置的数量
            repeat = 10  # 指定将对每个配置进行多少次测试
            # 指定运行配置测试需要多长时间，如果重复次数低于此时间，则增加其值，在 GPU 上进
            # 行精确调优时此选项是必需的，在 CPU 调优则不是必需的，将此值设置为 0表示禁用
            min_repeat_ms = 0  # 调优 CPU 时设置为 0
            timeout = 10  # 指明每个测试配置运行训练代码的时间上限

            # 创建 TVM 运行器
            runner = autotvm.LocalRunner(
                number=number,
                repeat=repeat,
                timeout=timeout,
                min_repeat_ms=min_repeat_ms,
                enable_cpu_cache_flush=True,
            )

            tuning_option = {
                "tuner": "xgb",  # 使用 XGBoost 算法来指导搜索
                "trials": 1500,  # cpu, 对于gpu 3000-4000
                "early_stopping": 10,  # 使得搜索提前停止的试验最小值
                "measure_option": autotvm.measure_option(
                    builder=autotvm.LocalBuilder(build_func="default"), runner=runner
                ),
                "tuning_records": "yolov8m-autotuning.json",
            }

            # 首先从 onnx 模型中提取任务
            tasks = autotvm.task.extract_from_program(mod["main"], target=self.target, params=params)

            # 按顺序调优提取的任务
            for i, task in enumerate(tasks):
                prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
                tuner_obj = XGBTuner(task, loss_type="reg")

                tuner_obj.tune(
                    n_trial=min(tuning_option["trials"], len(task.config_space)),
                    early_stopping=tuning_option["early_stopping"],
                    measure_option=tuning_option["measure_option"],
                    callbacks=[
                        autotvm.callback.progress_bar(tuning_option["trials"], prefix=prefix),
                        autotvm.callback.log_to_file(tuning_option["tuning_records"]),
                    ],
                )
            self.tvm_param = tuning_option["tuning_records"]
        # 用优化的算子重新编译模型来加快计算速度
        with autotvm.apply_history_best(self.tvm_param):
            with tvm.transform.PassContext(opt_level=3, config={}):
                lib = relay.build(mod, target=self.target, params=params)

        # dev = tvm.device(str(self.target), 0)
        # module = graph_executor.GraphModule(lib["default"](dev))
        module = None
        return module, lib

    def infer(self, img_src):
        '''
        :param img_src: np.ndarray (H, W, C), BGR格式
        :return:
        '''
        img = self.precess_image(img_src, self.imgsz, self.half)
        t1 = time.time()
        if self.is_tvm:
            self.tvm_module.set_input(self.tvm_input_name, img)
            self.tvm_module.run()
            preds = self.tvm_module.get_output(0, tvm.nd.empty(self.tvm_out_shape)).numpy()
            preds = torch.tensor(preds)
        else:
            preds = self.model(img)

        t2 = time.time()
        print((t2 - t1) * 1000)
        det = ops.non_max_suppression(preds, self.conf, self.iou,
                                      classes=None, agnostic=False, max_det=300, nc=len(self.names))
        # t3 = time.time()
        return_res = []
        for i, pred in enumerate(det):
            # lw = max(round(sum(img_src.shape) / 2 * 0.003), 2)  # line width
            # tf = max(lw - 1, 1)  # font thickness
            # sf = lw / 3  # font scale
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], img_src.shape)
            results = pred.cpu().detach().numpy()
            for result in results:
                return_res.append([result[:4], result[4], int(result[5])])
                # self.draw_box(img_src, result[:4], result[4], self.names[result[5]], lw, sf, tf)

        # cv2.imwrite(os.path.join(save_path, os.path.split(img_path)[-1]), img_src)
        return return_res
        # return (t2 - t1) * 1000, (t3 - t2) * 1000

    def draw_box(self, img_src, box, conf, cls_name, lw, sf, tf):
        color = self.color[cls_name]
        label = f'{cls_name} {round(conf, 3)}'
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        # 绘制矩形框
        cv2.rectangle(img_src, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
        # text width, height
        w, h = cv2.getTextSize(label, 0, fontScale=sf, thickness=tf)[0]
        # label fits outside box
        outside = box[1] - h - 3 >= 0
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        # 绘制矩形框填充
        cv2.rectangle(img_src, p1, p2, color, -1, cv2.LINE_AA)
        # 绘制标签
        cv2.putText(img_src, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                    0, sf, self.color["font"], thickness=2, lineType=cv2.LINE_AA)

    @staticmethod
    def letterbox(im: ndarray,
                  new_shape: Union[Tuple, List] = (640, 640),
                  color: Union[Tuple, List] = (114, 114, 114),
                  stride=32) -> Tuple[ndarray, float, Tuple[float, float]]:
        # todo 640x640,加灰度图
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        # new_shape: [width, height]

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[1], new_shape[1] / shape[0])
        # Compute padding [width, height]
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[0] - new_unpad[0], new_shape[1] - new_unpad[1]  # wh padding

        # todo 这步操作,能填充一个包裹图片的最小矩形,相当于动态shape, 输出目标的置信度与较大偏差
        # dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, r, (dw, dh)

    def precess_image(self, img_src, img_size, half):
        # Padded resize
        img = self.letterbox(img_src, img_size)[0]
        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)

        img = img.half() if half else img.float()  # uint8 to fp16/32
        img = img / 255  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        return img


if __name__ == '__main__':
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    target = "llvm -mtriple=armv8l-linux-gnueabihf"
    target = tvm.target.arm_cpu('rasp4b')

    weight_path = r'yolov8m.onnx'
    # weights = r'yolov8n.pt'
    save_path = "./runs"

    model = YOLOV8DetectionInfer(weight_path, 0.45, 0.45,
                                 target=target,
                                 device=device,
                                 is_tvm=True,
                                 tvm_param="yolov8m-autotuning.json")
    # 保存编译文件
    module, lib = model.init_tvm_optimize(weight_path)
    model.save_tvm_lib(lib, '.')
    sys.exit()
    module = model.load_tvm_lib('deploy_lib.tar')
    model.tvm_module = module

    #
    img_path = r"ultralytics/assets/bus.jpg"
    img_src = cv2.imread(img_path)

    res = model.infer(img_src)
    print('-------------------------------------')
    for i in range(10):
        res = model.infer(img_src)
    #
    for i in res:
        print(i)

# 6819.404602050781 28.783559799194336
