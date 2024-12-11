# -*- coding: utf-8 -*-
# @Time    : 2024/11/18 上午11:08
# @Author  : yblir
# @File    : 3.lowB排序.py
# explain  : 
# =======================================================
# 冒泡排序，如果前面比后面大，则交换这两个数字
def bubble_sort(nums):
    for i in range(len(nums) - 1):
        for j in range(0, len(nums) - i - 1):
            if nums[j] > nums[j + 1]:
                nums[j], nums[j + 1] = nums[j + 1], nums[j]


# 选择排序
def select_sort(nums):
    for i in range(len(nums) - 1):
        min_index = i
        for j in range(i + 1, len(nums)):
            if nums[j] < nums[min_index]:
                min_index = j
        nums[i], nums[min_index] = nums[min_index], nums[i]


# 插入排序
def insert_sort(nums):
    pass


if __name__ == '__main__':
    a = [2, 1, 5, 4, 3, 9, 7, 4, 0, 8]
    print(a)
    # bubble_sort(a)
    select_sort(a)
    print(a)
    en
