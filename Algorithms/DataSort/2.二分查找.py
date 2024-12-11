# -*- coding: utf-8 -*-
# @Time    : 2024/11/15 下午4:19
# @Author  : yblir
# @File    : 2.二分查找.py
# explain  : 
# =======================================================
# 双指针法
def binary_search(nums, target):
    left = 0
    right = len(nums) - 1

    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] > target:  # 说明目标值在坐标，移动右端指针
            right = mid - 1
        else:
            left = mid + 1
    else:
        return None


if __name__ == '__main__':
    nums = [34, 6, 9, 7, 8, 3, 4, 5, 0, 1]
    nums.sort()
    print(nums)
    res = binary_search(nums, 4)
    print(res)
