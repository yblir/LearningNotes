# -*- coding: utf-8 -*-
# @Time    : 2024/12/27 下午5:24
# @Author  : yblir
# @File    : 二叉树.py
# explain  : 
# =======================================================
from collections import deque


class BiTreeNode:
    def __init__(self, data):
        self.data = data
        self.lchild = None
        self.rchild = None


a = BiTreeNode('A')
b = BiTreeNode('B')
c = BiTreeNode('C')
d = BiTreeNode('D')
e = BiTreeNode('E')
f = BiTreeNode('F')
g = BiTreeNode('G')

e.lchild = a
e.rchild = g
a.rchild = c
c.lchild = b
c.rchild = d
g.rchild = f

root = e


#print(root.lchild.rchild.data)
# 前序遍历：先访问根节点(自己)，再访问左右节点
def pre_order(root):
    if root:
        print(root.data, end=',')
        pre_order(root.lchild)
        pre_order(root.rchild)


# 中序遍历：先访问左子树，再访问自己，再访问右节点
def in_order(root):
    if root:
        in_order(root.lchild)
        print(root.data, end=',')
        in_order(root.rchild)


def post_order(root):
    if root:
        post_order(root.lchild)
        post_order(root.rchild)
        print(root.data, end=',')


def level_order(root):
    q = deque()
    q.append(root)
    while len(q) > 0:
        node = q.popleft()
        print(node.data, end=',')
        if node.lchild:
            q.append(node.lchild)
        if node.rchild:
            q.append(node.rchild)


if __name__ == '__main__':
    #pre_order(root)
    #in_order(root)
    #post_order(root)
    level_order(root)
