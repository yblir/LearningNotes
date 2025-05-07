# -*- coding: utf-8 -*-
# @Time    : 2025/5/7 下午3:49
# @Author  : yblir
# @File    : 17. 协程1.py
# explain  : 
# =======================================================
import asyncio

"""
协程，也可以称为微线程，是一种用户态内的上下文切换技术，简单地说，就是通过一个线程实现 不同程序的代码块相互切换执行
实现协程方法有以下几种：
greenlet, 早期模块，
yield关键字
asyncio装饰器
async，await关键字，（目前主流）
"""


async def func1():
    print(1)
    await asyncio.sleep(1)  # 遇到io耗时操作，自动切换到tasks中其他任务
    print(2)


async def func2():
    print('222')
    await asyncio.sleep(0.5)
    print('444')


tasks = [
    asyncio.ensure_future(func1()),
    asyncio.ensure_future(func2())

]
# 事件循环：一个死循环，去检测并执行某些代码
# get_event_loop：去生成或获取一个事件循环
loop = asyncio.get_event_loop()
# 将任务放到任务列表
loop.run_until_complete(asyncio.wait(tasks))

# 协成函数：async def 函数名
# 协程对象： 执行协成函数()得到的携程对象，内部代码不会执行

# 如果要执行携程函数内部代码，必须要将携程对象交给循环事件来处理
# loop.run_until_complete(协成函数())

# 更推荐的写法, 替换loop = asyncio.get_event_loop()
# # 将任务放到任务列表
# loop.run_until_complete(asyncio.wait(tasks)) 这两行
# asyncio.run(协成函数())
print('-------------------------')
asyncio.run(func1())

# 3. await+可等待队形（携程对象，task对象，可简单理解为io等待），当io结束才往下走
# 一个携程函数中可以有多个await，遇到await，可以去执行其他任务，但当前携程函数会卡在这里，
# 直到这个await执行完才会继续
