# -*- coding: utf-8 -*-
# @Time    : 2024/12/27 下午5:24
# @Author  : yblir
# @File    : 二叉树.py
# explain  : 
# =======================================================
class BiTreeNode:
    def __init__(self,data):
        self.data=data
        self.lchild=None
        self.rchild=None

a=BiTreeNode('A')
b=BiTreeNode('B')
c=BiTreeNode('C')
d=BiTreeNode('D')
e=BiTreeNode('E')
f=BiTreeNode('F')
g=BiTreeNode('G')

e.lchild=a
e.rchild=g
a.rchild=c
c.lchild=b
c.rchild=d
g.rchild=f

root=e
print(root.lchild.rchild.data)