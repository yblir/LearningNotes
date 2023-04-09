import os
from pathlib2 import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from queue import Queue
from multiprocessing import Queue, cpu_count, shared_memory, Process

import json
import cv2

work_nums = 30
img_queue = Queue(1000)
shm = shared_memory.ShareableList([0])


def get_img_path(img_dir):
    for img_path in img_dir.rglob('*'):
        if img_path.suffix in ('.jpg', '.png', '.jpeg'):
            img_queue.put(img_path)
    img_queue.put('#####')


def run_process(func, img_dir):
    proc = Process(target=get_img_path, args=(img_dir,))
    proc.start()

    with ProcessPoolExecutor(max_workers=work_nums) as p:
        while True:
            img_path = img_queue.get()
            if img_path == '#####':
                break
            p.submit(func, img_path)

            shm[0] = shm[0] + 1  # 使用共享内存变量计数,这样总不会出现死锁问题吧!
            if (shm[0] % 1000) == 0:
                print(f'current_nums={shm[0]}')
    proc.join()  # 结束图片读取进程


if __name__ == '__main__':
    print(shm)
    shm[0] = shm[0] + 1
    print(shm[0])
