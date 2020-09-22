from PIL import Image, ImageEnhance
import numpy as np
import random

rdir = 'data/train.txt'

def sdir_path(path_list):
    sdir = ''
    for item in path_list:
        sdir = sdir + '/' + item
    return sdir

def data_enhance(currentPath):
    img_info = currentPath.split('/')
    img_name = img_info[-1]
    img_info.remove(img_name)
    img_info.remove(img_info[0])
    sdir = sdir_path(img_info)
    # print(sdir)
    new_name = sdir + '/rsi_' + img_name
    # print(new_name)
    img = Image.open(currentPath)
    if img.mode == "P":
        img = img.convert('RGB')
    random_factor = np.random.randint(0, 31) / 10.  # 随机因子
    color_image = ImageEnhance.Color(img).enhance(random_factor)  # 调整图像的饱和度
    random_factor = np.random.randint(10, 21) / 10.  # 随机因子
    brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # 调整图像的亮度
    random_factor = np.random.randint(10, 21) / 10.  # 随机因子
    contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # 调整图像对比度
    random_factor = np.random.randint(0, 31) / 10.  # 随机因子
    sharpness_image = ImageEnhance.Sharpness(contrast_image).enhance(random_factor)
    out = sharpness_image
    out.save(new_name)
    return new_name

def train_data_enhance(train_list_path):
    train_de_list = []
    with open(train_list_path, 'r') as f:
        lines = [line.strip() for line in f]
        for line in lines:
            img_path, lab = line.strip().split('\t')
            new_name = data_enhance(img_path)
            str = new_name + "\t" + lab + "\n"
            train_de_list.append(str)
    random.shuffle(train_de_list)
    with open(train_list_path, 'a') as f:
        for eval_image in train_de_list:
            f.write(eval_image)

train_data_enhance(rdir)