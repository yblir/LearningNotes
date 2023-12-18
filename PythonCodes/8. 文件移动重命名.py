# -*- coding: utf-8 -*-
# @Time    : 2023/12/18 10:57
# @Author  : yblir
# @File    : 8. 文件移动重命名.py
# explain  : 
# =======================================================
import os
import shutil
from pathlib2 import Path

# def pathlib2_rename(dir_path):
#     """
#     使用Path重命名
#     Args:
#         dir_path: 待重名文件所在文件夹
#
#     Returns:
#
#     """

if __name__ == '__main__':
    dir_path = Path("")
    target_dir_path = Path("")

    # ------------------------------------------------------------------------------------------------------------------
    # 使用pathlib2重命名
    for i, file_path in enumerate(dir_path.iterdir()):
        if file_path.suffix == '.mp4':
            # 以下两种方式都行
            file_path.rename(file_path.parent / f"{i}.mp4")
            # file_path.with_name(f"{i}.mp4")

    # ------------------------------------------------------------------------------------------------------------------
    # 使用os重命名, ps: 似乎没有Path方法方便
    for i, file_path in enumerate(dir_path.iterdir()):
        if file_path.suffix == '.mp4':
            os.rename(str(file_path), str(file_path.parent / f"{i}.mp4"))

    # ------------------------------------------------------------------------------------------------------------------
    # 移动文件
    for i, file_path in enumerate(dir_path.iterdir()):
        if file_path.suffix == '.mp4':
            shutil.move(str(file_path), str(target_dir_path))
