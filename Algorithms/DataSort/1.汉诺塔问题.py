# -*- coding: utf-8 -*-
# @Time    : 2024/11/15 下午4:00
# @Author  : yblir
# @File    : 1.汉诺塔问题.py
# explain  : 
# =======================================================
# 分治策略

# n, 从a，经过b，移动到c
def hanoi(n, a, b, c):
    # 结束条件，n=0
    if n > 0:
        hanoi(n - 1, a, c, b)
        print(f'从{a}移动到{c}')
        hanoi(n - 1, b, a, c)


if __name__ == '__main__':
    hanoi(10, 'A', 'B', 'C')
