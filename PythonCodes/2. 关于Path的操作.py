'''
该模块用于替换os.path
'''
import os

from pathlib2 import Path

# 获得当前目录,绝对路径
# cur_path=Path.cwd()

# 获得home目录,Linux时时home,windows时是C:\FH
# home_path=Path.home()

# 获得 上层路径,相当于os.path.dirname(),parents是路径列表, parent直接对应上层路径
# parents=cur_path.parents
# print(cur_path.parents[1])

'''
各种属性:
    1. cur_path.name:目录的最后一个部分,若aaa.jpg
    2. cur_path.suffix:目录中最后一个部分的扩展名,若是文件夹,则没有扩展名,返回 ''
    3. cur_path.suffixes:目录中最后一个部分的扩展名列表,若是文件夹,则没有扩展名,返回 []
    4. cur_path.stem 目录最后一个部分，没有后缀, 如aaa
'''
# 替换目录最后一个部分的文件名并返回一个新的路径
# new_path1 = example_path.with_name('def.gif')
# print(new_path1)
# 输出如下：
# /Users/Anders/Documents/def.gif

# 替换目录最后一个部分的文件名并返回一个新的路径
# new_path2 = example_path.with_suffix('.txt')
# print(new_path2)
# 输出如下：
# /Users/Anders/Documents/abc.txt

# #利用 / 可以创建子路径 / 可以替代joinpath
example_path4 = Path('/Users/Anders/Documents').joinpath('aa')
example_path5 = example_path4 / 'python_learn/pic-2.jpg'

'''
以下也只有个is_dir与is_file最常用了
is_dir() 是否是目录
is_file() 是否是普通文件
is_symlink() 是否是软链接
is_socket() 是否是socket文件
is_block_device() 是否是块设备
is_char_device() 是否是字符设备
is_absolute() 是否是绝对路径
resolve() 返回一个新的路径，这个新路径就是当前Path对象的绝对路径，如果是软链接则直接被解析
absolute() 也可以获取绝对路径，但是推荐resolve()
exists() 该路径是否指向现有的目录或文件：
'''


data_path = Path('F:\interesting\datasets')
print('===================')
print([i for i in data_path.iterdir()])  # 查看数量 相当于len(os.listdir())
for item in data_path.iterdir():  # 等同于os.listdir()
    print(item)
    # print('===========')
print('0-------------------')

for item in data_path.rglob('*'):   # 相当于os.walk操作,遍历文件内所有内容,比walk方便多了
    # if item.suffix=='.jpg':
    #     continue
    # if item.suffix=='.txt':
    #     continue
    if item.is_dir():
        print(item)
