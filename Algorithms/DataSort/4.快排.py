# -*- coding: utf-8 -*-
# @Time    : 2024/12/17 下午3:12
# @Author  : yblir
# @File    : 4.快排.py
# explain  : 
# =======================================================
def partition(li, left, right):
    temp = li[left]
    while left < right:
        while left < right and li[right] >= temp:  # 找比temp小的数，放到left指向的空位
            right -= 1  # 如果没找到，指针向左移动一位
        li[left] = li[right]  # 如果找到了,就把右边的值赋值给左边
        # print(li)
        while left < right and li[left] <= temp:
            left += 1
        li[right] = li[left]  # 把左边的值写到右边
        # print(li)
    li[left] = temp  # 此时,left==right,把temp归为

    return left


# 时间复杂度nlongn
def quick_sort(li, left, right):
    if left < right:
        mid = partition(li, left, right)
        quick_sort(li, left, mid - 1)
        quick_sort(li, mid + 1, right)


if __name__ == '__main__':
    li = [5, 7, 4, 6, 3, 1, 2, 9, 8]
    # print(li)
    # partition(li, 0, len(li) - 1)
    # print(li)
    quick_sort(li, 0, len(li) - 1)
    print(li)
