# -*- coding: utf-8 -*-
# @Time    : 2024/12/17 下午3:35
# @Author  : yblir
# @File    : 5.堆排序.py
# explain  : 
# =======================================================
import random

# 二叉树： 度不超过2的树
# 满二叉树：一个二叉树，每层节点数达到最大
# 完全二叉树：叶子节点只出现在最下层和次下层，在完全二叉树中，除了最底层节点可能没填满外，
# 其余每层节点数都达到最大值，并且最下面一层的节点都集中在该层最左边的若干位置

# 堆：一种完全二叉树，满足任一节点都比其子节点大，称为大跟堆，反之称为小根堆
# 用大根堆排出来的是增序

# 节点的左右子树都是堆，但自身不是堆，可以通过向下调整变成堆

'''
堆排序的过程：
1.建立堆
2.得到堆顶元素，为最大元素
3.去掉堆顶，将堆最后一个元素放到堆顶，此时可通过一次调整重新使堆有序
4.堆顶元素为第二大元素
5.重复步骤3，知道堆变空
'''


def sift(li, low, high):
    """
    Args:
        li: list
        low: 堆的根节点位置
        high: 堆的最后一个元素位置
    Returns:
    """
    i = low  # 堆顶，指向根节点
    j = 2 * i + 1  # i的左节点
    temp = li[low]  # 把堆顶元素存储起来

    while j <= high:  # 只要j指向的位置有数，就继续循环
        if j + 1 <= high and li[j + 1] > li[j]:  # 若右节点存在，且右节点大于左节点值
            j = j + 1  # 让j指向右节点
        if li[j] > temp:
            li[i] = li[j]  # 把较大子节点的值更新到堆顶
            i = j  # 更新新的堆顶和它的左节点
            j = 2 * i + 1
        else:
            li[i] = temp  # 把temp放到某个子节点空位上
            break
    else:
        li[i] = temp  # 当while条件不满足时，走完while会进入else，此时要把拿出来的堆顶元素放到空位上

# 时间复杂度 nlogn
def heap_sort(li):
    n = len(li)
    # 从最后一个叶子节点序号len(li)-1开始倒叙遍历
    # 这个最后叶子节点的父节点是len(li)-1-1)//2，整除，所以不管左右节点都适用
    for i in range((len(li) - 1 - 1) // 2, -1, -1):
        # i表示当前调整部分的根节点
        sift(li, i, n - 1)  # 建堆完成
    for i in range(n - 1, -1, -1):
        # i 指向当前堆的最后一个元素, 让堆顶元素与堆最后一个元素交换
        li[0], li[i] = li[i], li[0]
        # i-1是新的high
        sift(li, 0, i - 1)


if __name__ == '__main__':
    li = [i for i in range(20)]
    random.shuffle(li)
    print(li)
    heap_sort(li)
    print(li)
