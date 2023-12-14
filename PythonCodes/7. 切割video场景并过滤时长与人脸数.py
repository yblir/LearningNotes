# -*- coding: utf-8 -*-
# @Time    : 2023/12/14 9:44
# @Author  : yblir
# @File    : 7. 切割video场景并过滤时长与人脸数.py
# explain  : 
# =======================================================
import os
import sys
from pathlib2 import Path
from loguru import logger

import cv2
import numpy as np
import torch
import warnings
import shutil

from scenedetect import SceneManager, VideoStreamCv2
from scenedetect.detectors import ContentDetector
from scenedetect.video_splitter import split_video_ffmpeg
from decord import VideoReader

warnings.filterwarnings("ignore")

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', '..'))

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.cuda.set_device(0)


# scene_manager = SceneManager()
# scene_manager.add_detector(ContentDetector())


def split_video_scenes(raw_video_path, save_split_video_path):
    # SceneManager()对象, 每个视频创建一个, 只能在函数内. 在外部创建会出错
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())

    video_manager = VideoStreamCv2(str(raw_video_path))
    scene_manager.detect_scenes(frame_source=video_manager)

    # 保存为视频文件
    scene_list = scene_manager.get_scene_list()
    # 视频只有一个场景,不切割直接返回原视频
    if not scene_list:
        shutil.copy2(str(video_path), str(split_save_dir))
        return
    for index, scene in enumerate(scene_list):
        split_start_time = scene[0].get_timecode().replace("00:", "", 1)[:8]
        split_end_time = scene[1].get_timecode().replace("00:", "", 1)[:8]
        split_video_ffmpeg(str(video_path), [scene],
                           # rf"C:\Users\Administrator\Desktop\youtube\xidada_\{index + 1}.mp4",
                           f"{save_split_video_path}/{str(video_path.stem)}_{index}.mp4",
                           "",
                           # show_progress=True,
                           # show_output=True,
                           # suppress_output=True
                           )
        # 当前场景时长
        # 这行有报错, 用不到, 暂时不管了
        # scene_time = float(split_end_time.split("00:")[1]) - float(split_start_time.split("00:")[1])


# 只保留合规时长的视频
def filter_scene_duration(video_path, save_filtered_video_path, save_big_duration_video_path):
    try:
        vr = VideoReader(str(video_path))
    except:
        logger.error(f"VideoReader error: {str(video_path)}")
        return
        # 小于3s,过滤掉,不保存

    vr_length = len(vr)

    if vr_length < 50:
        return
        # 大于40s,过滤掉,另存在一个文件夹
    elif vr_length > 1200:
        shutil.copy2(str(video_path), str(save_big_duration_video_path))
    else:
        # 3s-40s
        shutil.copy2(str(video_path), str(save_filtered_video_path))


def get_video_frames_index(video_path):
    """
    通过随机采样方式获得视频中图片帧索引
    """
    vr = VideoReader(str(video_path))
    video_len = len(vr)

    base_idxs = np.linspace(0, video_len - 1, 150, dtype=np.int)
    base_idxs_len = len(base_idxs)

    tick = base_idxs_len / float(16)
    offsets = np.array([int(tick / 2.0 + tick * x) for x in range(16)])
    offsets = base_idxs[offsets].tolist()

    # return base_idxs, offsets
    return offsets


def filter_face_nums(video_path, save_video_dir_path):
    """
    过滤视频中人脸数量
    Args:
        video_path: 视频路径
        save_video_dir_path: 保存符合要求的视频路径文件夹
    Returns:

    """
    try:
        vr = VideoReader(str(video_path))
    except:
        print(str(video_path))
        return
    offsets = get_video_frames_index(vr)
    frames = vr.get_batch(offsets).asnumpy()

    face_count = 0
    face_num = 0
    for idx in range(len(frames)):
        img = frames[idx]
        # 人脸检测算法获得人脸数量
        res = mynet.detect(img)
        if len(res) > 1:
            face_count += 1
            break
        if len(res) == 1:
            face_num += 1

    # 当一个视频检测人脸数大于15个时,才进行保存
    if face_count == 0 and face_num >= 15:
        shutil.copy2(str(video_path), str(save_video_dir_path))


if __name__ == '__main__':
    # 原视频路径文件夹
    root_path = Path(r"E:\DeepFakeDetection\datasets\bilibili2")
    # 分割后保存路径
    split_save_dir = Path(r"E:\DeepFakeDetection\datasets\bilibili_splits2")
    # 已清理时长的视频文件夹路径
    cleaned_splits_scenes = Path(r"E:\DeepFakeDetection\datasets\bilibili2_clean_scene")
    save_big_duration_video_path = Path(r"E:\DeepFakeDetection\datasets\bilibili2_big_duration_scenes")
    save_scence_one_face = Path(r"")

    os.makedirs(str(split_save_dir), exist_ok=True)
    os.makedirs(str(cleaned_splits_scenes), exist_ok=True)
    os.makedirs(str(save_big_duration_video_path), exist_ok=True)
    # ------------------------------------------------------------------------------------------------------------------
    count = 0
    # 场景切割
    for video_path in root_path.iterdir():
        split_video_scenes(video_path, split_save_dir)
        count += 1
        logger.info(f"scene split finished: {count}")
    logger.success("scenes split success")
    # ------------------------------------------------------------------------------------------------------------------
    # 过滤场景时长
    count1 = 0
    for video_path in split_save_dir.iterdir():
        filter_scene_duration(video_path, cleaned_splits_scenes, save_big_duration_video_path)
        count1 += 1
        if count1 % 100 == 0:
            logger.info(f"filtered scene nums: {count1}")
    logger.success("scenes duration filter success")
    # ------------------------------------------------------------------------------------------------------------------
    #  过滤人脸
    count2 = 0
    for video_path in cleaned_splits_scenes.iterdir():
        filter_face_nums(video_path, save_scence_one_face)
        count2 += 1
        if count2 % 100 == 0:
            logger.info(f"filtered scene face video: {count2}")

# logger.success("success")
