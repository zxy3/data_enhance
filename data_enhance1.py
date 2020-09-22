# -*- coding: utf-8 -*-
'''
   File Name: data_enhance1
   Author: zxy
   date: 2020/8/10
'''
def number_of_variation_sample(folder_path):
    num_dirs = 0 #路径下文件夹数量
    num_files = 0 #路径下文件数量(包括文件夹)
    num_files_rec = 0 #路径下文件数量,包括子文件夹里的文件数量，不包括空文件夹
    file_list = []
    for root,dirs,files in os.walk(folder_path):
        for each in files:
            if each[-2:] == '.o':
                # print(root,dirs,each)
                num_files_rec += 1
        for name in dirs:
                num_dirs += 1
                # print(os.path.join(root,name))
    for fn in os.listdir(folder_path):
        num_files += 1
        file_list.append(fn)
    return num_files,file_list

import os
path = os.getcwd()    #获取当前路径
print(path)
# 匹配目标路径
tar_path = "\\train"
tar_path = path + tar_path
# 数据集路径
print(tar_path)

file_num,file_list = number_of_variation_sample(tar_path)
print(file_num,file_list)
# print(file_num[1])

new_path = tar_path + os.sep + file_list[2]
print('new_path',new_path)
data = {}
import Augmentor
import pandas as pd
# 进入每个食物种类的文件夹
for i in range (file_num):
    new_path = tar_path + os.sep + file_list[i]
    print(new_path)
    file_num_1, file_list_1 = number_of_variation_sample(new_path)
    data[file_list[i]] = file_num_1

    # p=Augmentor.Pipeline(new_path)  #设置路径

# print(data)

import os
import pandas as pd
import csv
# file = open(os.getcwd()+os.sep+'data_list.txt', 'w')
# file.write(str(data))   #msg也就是下面的Hello world!
# dict_a = [data]
# pd_data = pd.DataFrame(dict_a)
# print(pd_data)
# pd_data.to_csv('pandas.csv',header=True,index=True)


# for i in range (file_num):
#     new_path = tar_path + os.sep + file_list[i]
#     # if i == 3:
#     #     p=Augmentor.Pipeline(new_path)  #设置路径
#     # elif i == 36 or i == 0 or i == 20:
#     #     p = Augmentor.Pipeline(new_path)  # 设置路径
#     # elif i == 2 or i == 32 or i == 5 or i == 12 or i == 19 or i == 17 or i == 23:
#     #     p = Augmentor.Pipeline(new_path)  # 设置路径
#     # elif i == 24 or i == 33 or i == 30 or i == 37 or i == 7 or i == 1 or i == 35:
#     #     p = Augmentor.Pipeline(new_path)  # 设置路径
#     p = Augmentor.Pipeline(new_path)  # 设置路径
#     # 0.5概率，向左最多旋转90度，向右最多旋转90度
#     p.rotate(probability=0.5,max_left_rotation=25, max_right_rotation=25)
#     num = 700 - i
#     p.sample(num)

# import shutil
# for i in range (file_num):
#     new_path = tar_path + os.sep + file_list[i] + os.sep + 'output'
#     shutil.rmtree(new_path)
