# -*- coding: utf-8 -*-
# @Time    : 2024/12/20 上午10:55
# @Author  : yblir
# @File    : 归并排序.py
# explain  : 
# =======================================================
def merge(li, low, mid, high):
    #归并排序使用了双指针
    i = low
    j = mid + 1
    ltemp = []

    while i <= mid and j <= high:
        if li[i] < li[j]:
            ltemp.append(li[i])
            i += 1
        else:
            ltemp.append(li[j])
            j += 1
    # 看哪部分有数
    while i <= mid:
        ltemp.append(li[i])
        i += 1
    while j <= high:
        ltemp.append(li[j])
        j += 1
    li[low:high + 1] = ltemp


def merge_sort(li, low, high):
    if low < high:  # 至少有两个
        mid = (low + high) // 2
        merge_sort(li, low, mid)
        merge_sort(li, mid + 1, high)
        merge(li, low, mid, high)


if __name__ == '__main__':
    li = [2, 4, 5, 7, 1, 3, 6, 8]
    print(li)
    #merge(li, 0, 3, 7)
    merge_sort(li, 0, len(li) - 1)
    print(li)
