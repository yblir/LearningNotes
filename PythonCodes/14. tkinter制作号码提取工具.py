# -*- coding: utf-8 -*-
# @Time    : 2024/3/29 16:20
# @Author  : yblir
# @File    : 14. tkinter制作号码提取工具.py
# explain  : 
# =======================================================
import re
import xlwt
import os

import tkinter as tk
from tkinter import filedialog
from tkinter import scrolledtext
from tkinter.filedialog import askopenfilename, askopenfile


def import_text():
    with askopenfile(title="上传文件",
                     # initialdir="d:",
                     filetypes=[("文本文件", ".txt")]) as f:
        left_text.insert(1.0, f.read())


def process_text():
    left_content = left_text.get('1.0', tk.END)
    # numbers = [word for word in left_content.split() if word.isdigit()]
    # right_text.delete('1.0', tk.END)
    # right_text.insert(tk.END, '\n'.join(numbers))
    num_list = []
    for line in left_content.split("\n"):
        result = re.findall(r'\d+', line)
        for res in result:
            if len(res) == 11 and res[:2] in ("13", "15", "17", "18"):
                num_list.append(res)

    right_text.delete('1.0', tk.END)
    right_text.insert(tk.END, '\n'.join(num_list))


def save_num_to_excel():
    save_path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel Files", "*.xlsx")])
    work_book = xlwt.Workbook(encoding='utf-8')
    sheet = work_book.add_sheet('号码簿')
    right_content = right_text.get('1.0', tk.END)
    # numbers=[for ]
    # 设置列宽
    sheet.col(0).width = 10 * 400
    tall_style = xlwt.easyxf("font: height 280;")

    # sheet.write(0, 0, '手机号码')
    for i, item in enumerate(right_content.split("\n")):
        first_row = sheet.row(i)
        first_row.set_style(tall_style)
        sheet.write(i, 0, item)

    # path = os.path.dirname(os.path.realpath(__file__))
    work_book.save(save_path)


def save_num_to_txt():
    save_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text Files", "*.txt")])
    right_content = right_text.get('1.0', tk.END)

    with open(save_path, "w", encoding="utf-8") as f:
        for i, item in enumerate(right_content.split("\n")):
            f.write(item + "\n")


def clear_():
    left_text.delete(1.0, tk.END)
    right_text.delete(1.0, tk.END)


if __name__ == '__main__':
    root = tk.Tk()
    root.title("文本处理器")
    root.geometry("800x400+500+300")

    frame_left = tk.Frame(root)
    frame_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    left_text = scrolledtext.ScrolledText(frame_left)
    left_text.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    delete_button = tk.Button(frame_left, text="清空", command=clear_)
    delete_button.pack(side=tk.LEFT, padx=150, pady=5)

    import_button = tk.Button(frame_left, text="导入数据", command=import_text)
    import_button.pack(side=tk.LEFT, padx=5, pady=5)

    frame_right = tk.Frame(root)
    frame_right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    right_text = scrolledtext.ScrolledText(frame_right)
    right_text.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    process_button = tk.Button(frame_left, text="提取号码", command=process_text)
    process_button.pack(side=tk.LEFT, padx=15, pady=5)

    # 导出文件
    output_button1 = tk.Button(frame_right, text="输出到Excel", command=save_num_to_excel)
    output_button1.pack(side=tk.RIGHT, padx=5, pady=5)

    output_button2 = tk.Button(frame_right, text="输出到Txt", command=save_num_to_txt)
    output_button2.pack(side=tk.LEFT, padx=5, pady=5)

    # 创建一个退出按钮
    # bt_quit = tk.Button(root, text="退出", command=root.destroy)
    # bt_quit.pack()

    root.mainloop()
