#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 查看当前挂载的数据集目录, 该目录下的变更重启环境后会自动还原
# View dataset directory. This directory will be recovered automatically after resetting environment. 
get_ipython().system('ls /home/aistudio/data')


# In[ ]:


# 查看工作区文件, 该目录下的变更将会持久保存. 请及时清理不必要的文件, 避免加载过慢.
# View personal work directory. All changes under this directory will be kept even after reset. Please clean unnecessary files in time to speed up environment loading.
get_ipython().system('ls /home/aistudio/work')


# In[ ]:


# 如果需要进行持久化安装, 需要使用持久化路径, 如下方代码示例:
# If a persistence installation is required, you need to use the persistence path as the following:
# !mkdir /home/aistudio/external-libraries
# !pip install beautifulsoup4 -t /home/aistudio/external-libraries


# In[ ]:


# 同时添加如下代码, 这样每次环境(kernel)启动的时候只要运行下方代码即可:
# Also add the following code, so that every time the environment (kernel) starts, just run the following code:
import sys
sys.path.append('/home/aistudio/external-libraries')


# In[ ]:


# 第一次运行时执行，后面就不运行了
# !unzip  -qo data/data23828/training.zip -d 'dataset'
# !unzip  -qo data/data23828/validation.zip -d  'dataset'
# !unzip  -qo data/data23828/valid_gt.zip -d  'dataset'


# In[ ]:


# 第一次运行时执行，后面就不执行了
# !unzip  -qo dataset/PALM-Training400/PALM-Training400.zip -d  'dataset/PALM-Training400'


# ##  数据增强
# 
# ### 原始数据集+随机翻转（上下左右4个方向）

# In[ ]:


import numpy as np
import random
from PIL import Image, ImageEnhance
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageOps, ImageFile
import os
import os.path
get_ipython().run_line_magic('matplotlib', 'inline')


# ### 获取当前路径

# In[ ]:


import os
path = os.getcwd()    #获取当前路径
print(path)
tar_path = "/dataset/PALM-Training400/PALM-Training1600-overturn_dim"
tar_path = path + tar_path
print(tar_path)


# In[ ]:


# 计算文件夹中样本个数和样本名称列表

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
    #     print ("->",fn)
    # print (num_dirs)
    # print (num_files)
    # print (num_files_rec)
    return num_files,file_list


# ### 随机旋转

# In[ ]:


# 随机翻转
def random_flip(img, thresh=0.5):
    img = np.asarray(img)
    if random.random() > thresh:
        img = img[:, ::-1, :]
    if random.random() > thresh:
        img = img[::-1 , :, :]
    img = Image.fromarray(img)
    return img
# 随机旋转图像
def random_rotate(img, thresh=0.5):
    # 任意角度旋转
    angle = np.random.randint(0, 360)
    img = img.rotate(angle)
    '''
    # 0, 90, 270, 360度旋转
    img = np.asarray(img)
    if random.random() > thresh:
        img = img[:, ::-1, :]
    if random.random() > thresh:
        img = img[::-1 , :, :]
    img = Image.fromarray(img)
    '''
    return img


# In[ ]:


# test
img = cv2.imread("/home/aistudio/dataset/PALM-Training400/PALM-Training400/N0075.jpg")
print(img.shape)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# plt.figure(figsize=(12,8),dpi=80)
img = random_flip(img)
print(img.size)
path = "/home/aistudio/dataset/PALM-Training400/PALM-Training400/H0004.jpg"
print(path)
img.save(path)

# plt.savefig(path)
plt.imshow(img)


# In[ ]:


# PALM-Training400-overturn
# 原始数据集+随机翻转（上下左右4个方向）
# 800
def overturn(img):
    img_path = tar_path+img
    img = plt.imread(img_path)
    img = Image.fromarray(img)
    i = random.randint(0,1)
    if i == 0:
        # 随机翻转
        img = random_flip(img, thresh=0.5)
    elif i == 1:
        # 随机旋转
        img = random_rotate(img, thresh=0.5)
    return img


# ### 添加高斯白噪声

# In[ ]:


# PALM-Training400-noise
# 原始数据集+随机加高斯噪声
# 800
def random_noise(img, max_sigma = 10):
    img_path = tar_path+img
    img = plt.imread(img_path)
    img = Image.fromarray(img)
    img = np.asarray(img)
    sigma = np.random.uniform(0, max_sigma)
    # print("=>",img.shape)
    noise = np.round(np.random.randn(img.shape[0], img.shape[1], 3) * sigma).astype('uint8')
    img = img + noise
    img[img > 255] = 255
    img[img < 0] = 0
    img = Image.fromarray(img)
    return img


# ### 随机亮度、饱和度、对比度

# In[ ]:


# PALM-Training400-color
# 800
# 原始数据集+
# 随机改变亮度
def random_brightness(img, lower=0.5, upper=1.5,path = tar_path):
    e = np.random.uniform(lower, upper)
    return ImageEnhance.Brightness(img).enhance(e)

# 随机改变对比度
def random_contrast(img, lower=0.5, upper=1.5):
    e = np.random.uniform(lower, upper)
    return ImageEnhance.Contrast(img).enhance(e)

# 随机改变颜色(饱和度)
def random_color(img, lower=0.5, upper=1.5):
    e = np.random.uniform(lower, upper)
    return ImageEnhance.Color(img).enhance(e)

def random_color_change(img):
    img_path = tar_path+img
    img = plt.imread(img_path)
    img = Image.fromarray(img)
    i = random.randint(0,2)
    if i == 0:
        # 随机改变亮度
        img = random_brightness(img, lower=0.5, upper=1.5)
    elif i == 1:
        # 随机改变对比度
        img = random_contrast(img, lower=0.5, upper=1.5)
    elif i == 2:
        # 随机改变颜色(饱和度)
        img = random_color(img, lower=0.5, upper=1.5)
    return img


# ### 随机裁剪

# In[ ]:


# PALM-Training400-tailor
# 800
# 原始数据集+等比例随机裁剪
def random_crop(img, max_ratio=1.5):
    #if(random.random() > 0.5):
    #    return img
    img_path = tar_path+'/'+img
    img = plt.imread(img_path)
    img = Image.fromarray(img)
    img = np.asarray(img)
    h, w, _ = img.shape
    m = random.uniform(1, max_ratio)
    n = random.uniform(1, max_ratio)
    x1 = w * (1 - 1 / m) / 2
    y1 = h * (1 - 1 / n) / 2
    x2 = x1 + w * 1 / m
    y2 = y1 + h * 1 / n
    img = Image.fromarray(img)
    img = img.crop([x1, y1, x2, y2])
    type = [Image.NEAREST,Image.BILINEAR,Image.BICUBIC,Image.ANTIALIAS]
    img = img.resize((w, h),type[random.randint(0,3)])
    return img


# ### 随机改变清晰度

# In[ ]:


# PALM-Training400-dim
# 800
# 原始数据集+随机改变清晰度
def random_dim(img, lower=0.5, upper=1.5,path = tar_path):
    img_path = path+'/'+img
    img = plt.imread(img_path)
    img = Image.fromarray(img)
    e = np.random.uniform(lower, upper)
    return ImageEnhance.Sharpness(img).enhance(e)


# ### 随机拉伸

# In[ ]:


def random_deform(img,path = tar_path):
    """图像拉伸."""
        
    # 图像完整路径
    img_path = path+'/'+img
    img = plt.imread(img_path)
    img = Image.fromarray(img)
    w, h = img.size
    
    w = int(w)
    h = int(h)
    i = random.randint(0,1)
    if i == 0:
        # 拉伸成宽为w的正方形
        out_img = img.resize((int(w), int(w)))
    elif i == 1:
        # 拉伸成宽为h的正方形
        out_img = img.resize((int(h), int(h)))
    return out_img


# ### 拷贝文件夹

# In[ ]:


# import os
# import shutil

# source_path = os.path.abspath(r'dataset/PALM-Training400/PALM-Training1600-overturn_dim')
# target_path = os.path.abspath(r'dataset/PALM-Training400/PALM-Training1600-imgaug2')

# if not os.path.exists(target_path):
#     os.makedirs(target_path)

# if os.path.exists(source_path):
#     # root 所指的是当前正在遍历的这个文件夹的本身的地址
#     # dirs 是一个 list，内容是该文件夹中所有的目录的名字(不包括子目录)
#     # files 同样是 list, 内容是该文件夹中所有的文件(不包括子目录)
#     for root, dirs, files in os.walk(source_path):
#         for file in files:
#             src_file = os.path.join(root, file)
#             shutil.copy(src_file, target_path)
#             # print(src_file)

# print('copy files finished!')


# ### 清空文件夹 临时

# In[ ]:


# 临时文件夹temp: 文件夹不存在则创建，存在则清空

# filepath = "dataset/PALM-Training400/PALM-Training800-imgaug1"
# if not os.path.exists(filepath):
#     os.mkdir(filepath)
# else:
#     shutil.rmtree(filepath)
#     os.mkdir(filepath)


# ### PALM-Training400-overturn-noise-color
# ####  随机旋转叠加随机噪声和随机色彩

# In[ ]:


def overturn_noise_color(img):
    # 随机高斯白噪声
    img1 = random_noise(img)

    # 临时文件夹temp: 文件夹不存在则创建，存在则清空
    filepath = "dataset/PALM-Training400/temp"
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    else:
        shutil.rmtree(filepath)
        os.mkdir(filepath)

    temp_img_name = str(img)[0:-4]+'_temp.jpg'
    save_img_name = temp_img_name
    new_img_name = img1
    new_img_name.save(filepath+'/'+save_img_name)
    # 随机对比度，亮度，饱和度
    img2 = random_brightness(temp_img_name,filepath)
    return img


# ### PALM-Training1600-overturn-crop-deform
# #### 随机旋转叠加裁剪和拉伸
# 
# ### 更改为：PALM-Training3200-overturn-noise-color-crop-flex-dim
# #### 随机翻转+噪声+改变亮度、饱和度、对比度+裁剪+拉伸+改变清晰度

# In[ ]:


def overturn_noise_color_crop_flex_dim(img):
    # 临时文件夹temp: 文件夹不存在则创建，存在则清空
    filepath = "dataset/PALM-Training400/temp"
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    else:
        shutil.rmtree(filepath)
        os.mkdir(filepath)

    # 随机裁剪
    img1 = random_crop(img)
    temp_img_name = str(img)[0:-4]+'_temp.jpg'
    save_img_name = temp_img_name
    new_img_name = img1
    temp_path = "dataset/PALM-Training400/temp/"
    new_img_name.save(temp_path+'/'+save_img_name)

    # 随机拉伸
    img2 = random_deform(temp_img_name,temp_path)
    temp_img_name = str(img)[0:-4]+'_temp1.jpg'
    save_img_name = temp_img_name
    new_img_name = img2
    temp_path = "dataset/PALM-Training400/temp/"
    new_img_name.save(temp_path+'/'+save_img_name)

    # 随机改变清晰度
    img3 = random_dim(temp_img_name,path =temp_path)
    return img3


# ### PALM-Training1600-overturn_dim

# In[ ]:


def overturn_dim(img):
    # 随机改变清晰度
    img = random_dim(img)
    return img


# ### PALM-Training400-imgaug1	
# #### 原始数据集+四周0-50像素随机裁剪，50%概率水平翻转，高斯模糊（sigma = 0 到 3.0）

# In[ ]:


# ! pip install scipy==1.3.1
# sys.path.append('/home/aistudio/external-libraries')


# In[ ]:


# 初次运行时需要将下面两行注释取消
# !pip install imgaug -t /home/aistudio/external-libraries

# 同时添加如下代码, 这样每次环境(kernel)启动的时候只要运行下方代码即可:
# import sys
# sys.path.append('/home/aistudio/external-libraries')


# In[ ]:


import imgaug as ia
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt

def image_augment_imgauglib(img):
    img = np.array([img])
    ia.seed(random.randint(0, 10000))

    # 示例批图像:
    # 数组有shape (N, W, H, C)和dtype uint8。
    # Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
    # e.g. 有时(0.5，高斯模糊(0.3))会大约每秒模糊一次图片

    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    
    # Define our sequence of augmentation steps that will be applied to every image.
    seq = iaa.Sequential(
        [
            #
            # Apply the following augmenters to most images.
            #
            iaa.Fliplr(0.5), # horizontally flip 50% of all images
            iaa.Flipud(0.2), # vertically flip 20% of all images
    
            # crop some of the images by 0-10% of their height/width
            sometimes(iaa.Crop(percent=(0, 0.1))),
    
            # Apply affine transformations to some of the images
            # - scale to 80-120% of image height/width (each axis independently)
            # - translate by -20 to +20 relative to height/width (per axis)
            # - rotate by -45 to +45 degrees
            # - shear by -16 to +16 degrees
            # - order: use nearest neighbour or bilinear interpolation (fast)
            # - mode: use any available mode to fill newly created pixels
            #         see API or scikit-image for which modes are available
            # - cval: if the mode is constant, then use a random brightness
            #         for the newly created pixels (e.g. sometimes black,
            #         sometimes white)
            sometimes(iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-45, 45),
                shear=(-16, 16),
                order=[0, 1],
                cval=(0, 255),
                mode=ia.ALL
            )),
    
            #
            # Execute 0 to 5 of the following (less important) augmenters per
            # image. Don't execute all of them, as that would often be way too
            # strong.
            #
            iaa.SomeOf((0, 5),
                [
                    # Convert some images into their superpixel representation,
                    # sample between 20 and 200 superpixels per image, but do
                    # not replace all superpixels with their average, only
                    # some of them (p_replace).
                    sometimes(
                        iaa.Superpixels(
                            p_replace=(0, 1.0),
                            n_segments=(20, 200)
                        )
                    ),
    
                    # Blur each image with varying strength using
                    # gaussian blur (sigma between 0 and 3.0),
                    # average/uniform blur (kernel size between 2x2 and 7x7)
                    # median blur (kernel size between 3x3 and 11x11).
                    iaa.OneOf([
                        iaa.GaussianBlur((0, 3.0)),
                        iaa.AverageBlur(k=(2, 7)),
                        iaa.MedianBlur(k=(3, 11)),
                    ]),
    
                    # Sharpen each image, overlay the result with the original
                    # image using an alpha between 0 (no sharpening) and 1
                    # (full sharpening effect).
                    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
    
                    # Same as sharpen, but for an embossing effect.
                    iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
    
                    # Search in some images either for all edges or for
                    # directed edges. These edges are then marked in a black
                    # and white image and overlayed with the original image
                    # using an alpha of 0 to 0.7.
                    sometimes(iaa.OneOf([
                        iaa.EdgeDetect(alpha=(0, 0.7)),
                        iaa.DirectedEdgeDetect(
                            alpha=(0, 0.7), direction=(0.0, 1.0)
                        ),
                    ])),
    
                    # Add gaussian noise to some images.
                    # In 50% of these cases, the noise is randomly sampled per
                    # channel and pixel.
                    # In the other 50% of all cases it is sampled once per
                    # pixel (i.e. brightness change).
                    iaa.AdditiveGaussianNoise(
                        loc=0, scale=(0.0, 0.05*255), per_channel=0.5
                    ),
    
                    # Either drop randomly 1 to 10% of all pixels (i.e. set
                    # them to black) or drop them on an image with 2-5% percent
                    # of the original size, leading to large dropped
                    # rectangles.
                    iaa.OneOf([
                        iaa.Dropout((0.01, 0.1), per_channel=0.5),
                        iaa.CoarseDropout(
                            (0.03, 0.15), size_percent=(0.02, 0.05),
                            per_channel=0.2
                        ),
                    ]),
    
                    # Invert each image's channel with 5% probability.
                    # This sets each pixel value v to 255-v.
                    iaa.Invert(0.05, per_channel=True), # invert color channels
    
                    # Add a value of -10 to 10 to each pixel.
                    iaa.Add((-10, 10), per_channel=0.5),
    
                    # Change brightness of images (50-150% of original value).
                    iaa.Multiply((0.5, 1.5), per_channel=0.5),
    
                    # Improve or worsen the contrast of images.
                    iaa.LinearContrast((0.5, 2.0), per_channel=0.5),
    
                    # Convert each image to grayscale and then overlay the
                    # result with the original with random alpha. I.e. remove
                    # colors with varying strengths.
                    iaa.Grayscale(alpha=(0.0, 1.0)),
    
                    # In some images move pixels locally around (with random
                    # strengths).
                    sometimes(
                        iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)
                    ),
    
                    # In some images distort local areas with varying strength.
                    sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05)))
                ],
                # do all of the above augmentations in random order
                random_order=True
            )
        ],
        # do all of the above augmentations in random order
        random_order=True
    )
    img = seq(images=img)
    img = img[0]
    return img
   
# test
# img = cv2.imread("/home/aistudio/dataset/PALM-Training400/PALM-Training400/H0004.jpg")
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# plt.figure(figsize=(12,8),dpi=80)
# for i in range(16):
#     plt.subplot(4,4,i + 1)
#     plt.imshow(image_augment_imgauglib(img))
#     plt.xticks([])
#     plt.yticks([])


# ### PALM-Training800-imgaug1
# #### 原始数据集+四周0-50像素随机裁剪，50%概率水平翻转，高斯模糊（sigma =0到3.0）

# In[ ]:


print(tar_path)


# In[ ]:


def imgaug1(img,path = tar_path):
        
    # 图像完整路径
    img_path = path+'/'+img
    img = plt.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = image_augment_imgauglib(img)
    img = Image.fromarray(np.uint8(img))
    return img


# ### 查看数据集中样本数量

# In[ ]:


num_files,file_list = number_of_variation_sample("dataset/PALM-Training400/PALM-Training1600-imgaug2")
print(num_files)
# print(file_list)


# In[ ]:


# def traversal_folder(folder_path):
#     num_files,file_list = number_of_variation_sample(folder_path)
#     for i in range(num_files):
        # 随机翻转
        # img = overturn(file_list[i])
        # 随机白噪声
        # img1 = random_noise(file_list[i])
        # 随机改变亮度、饱和度、对比度
        # img = random_color_change(file_list[i])
        # 随机裁剪
        # img = random_crop(file_list[i])
        # 随机模糊
        # img = random_sharpness(file_list[i])
        # 随机拉伸
        # img = random_deform(file_list[i])
        # 随机翻转+噪声+改变亮度、饱和度、对比度
        # img = overturn_noise_color(file_list[i])
        # 随机翻转+裁剪+拉伸
        # img = overturn_crop_deform(file_list[i])
        # 随机翻转+改变清晰度
        # img = overturn_dim(file_list[i])
        # 随机翻转+噪声+改变亮度、饱和度、对比度+裁剪+拉伸+改变清晰度
        # img = overturn_noise_color_crop_flex_dim(file_list[i])
        # 第三方库1
        # img = imgaug1(file_list[i])
        # new_img_name = str(file_list[i])[0:-4]+'_imgaug2_'+str(i+1)+'.jpg'
        # save_img_name = new_img_name
        # new_img_name = img
        # save_path = "dataset/PALM-Training400/PALM-Training1600-imgaug2"
        # new_img_name.save(save_path+'/'+save_img_name)
        # print('save_path:',save_path+'/'+save_img_name)


# In[ ]:


print(tar_path)


# In[ ]:


# 数据集生成以后就不再执行
# traversal_folder(tar_path)


# ### 清空文件夹

# In[ ]:


# 有隐藏文件的话删除一下
# os.rmdir("dataset/PALM-Training400/.ipynb_checkpoints")


# ### 归一化处理

# In[97]:


import cv2
import random
import numpy as np
import os

# 对读入的图像数据进行预处理
def transform_img(img):
    # 将图片尺寸缩放道 224x224
    img = cv2.resize(img, (224, 224))
    # 读入的图像数据格式是[H, W, C]
    # 使用转置操作将其变成[C, H, W]
    img = np.transpose(img, (2,0,1))
    img = img.astype('float32')
    # 将数据范围调整到[-1.0, 1.0]之间
    img = img / 255.
    img = img * 2.0 - 1.0
    return img

# 定义训练集数据读取器
def data_loader(datadir, batch_size=10, mode = 'train'):
    # 将datadir目录下的文件列出来，每条文件都要读入
    filenames = os.listdir(datadir)
    def reader():
        if mode == 'train':
            # 训练时随机打乱数据顺序
            random.shuffle(filenames)
        batch_imgs = []
        batch_labels = []
        for name in filenames:
            filepath = os.path.join(datadir, name)
            img = cv2.imread(filepath)
            img = transform_img(img)
            if name[0] == 'H' or name[0] == 'N':
                # H开头的文件名表示高度近似，N开头的文件名表示正常视力
                # 高度近视和正常视力的样本，都不是病理性的，属于负样本，标签为0
                label = 0
            elif name[0] == 'P':
                # P开头的是病理性近视，属于正样本，标签为1
                label = 1
            else:
                raise('Not excepted file name')
            # 每读取一个样本的数据，就将其放入数据列表中
            batch_imgs.append(img)
            batch_labels.append(label)
            if len(batch_imgs) == batch_size:
                # 当数据列表的长度等于batch_size的时候，
                # 把这些数据当作一个mini-batch，并作为数据生成器的一个输出
                imgs_array = np.array(batch_imgs).astype('float32')
                labels_array = np.array(batch_labels).astype('float32').reshape(-1, 1)
                yield imgs_array, labels_array
                batch_imgs = []
                batch_labels = []

        if len(batch_imgs) > 0:
            # 剩余样本数目不足一个batch_size的数据，一起打包成一个mini-batch
            imgs_array = np.array(batch_imgs).astype('float32')
            labels_array = np.array(batch_labels).astype('float32').reshape(-1, 1)
            yield imgs_array, labels_array

    return reader

# 定义验证集数据读取器
def valid_data_loader(datadir, csvfile, batch_size=2, mode='valid'):
    # 训练集读取时通过文件名来确定样本标签，验证集则通过csvfile来读取每个图片对应的标签
    # 请查看解压后的验证集标签数据，观察csvfile文件里面所包含的内容
    # csvfile文件所包含的内容格式如下，每一行代表一个样本，
    # 其中第一列是图片id，第二列是文件名，第三列是图片标签，
    # 第四列和第五列是Fovea的坐标，与分类任务无关
    # ID,imgName,Label,Fovea_X,Fovea_Y
    # 1,V0001.jpg,0,1157.74,1019.87
    # 2,V0002.jpg,1,1285.82,1080.47
    # 打开包含验证集标签的csvfile，并读入其中的内容
    filelists = open(csvfile).readlines()
    def reader():
        batch_imgs = []
        batch_labels = []
        for line in filelists[1:]:
            line = line.strip().split(',')
            name = line[1]
            label = int(line[2])
            # 根据图片文件名加载图片，并对图像数据作预处理
            filepath = os.path.join(datadir, name)
            img = cv2.imread(filepath)
            img = transform_img(img)
            # 每读取一个样本的数据，就将其放入数据列表中
            batch_imgs.append(img)
            batch_labels.append(label)
            if len(batch_imgs) == batch_size:
                # 当数据列表的长度等于batch_size的时候，
                # 把这些数据当作一个mini-batch，并作为数据生成器的一个输出
                imgs_array = np.array(batch_imgs).astype('float32')
                labels_array = np.array(batch_labels).astype('float32').reshape(-1, 1)
                yield imgs_array, labels_array
                batch_imgs = []
                batch_labels = []

        if len(batch_imgs) > 0:
            # 剩余样本数目不足一个batch_size的数据，一起打包成一个mini-batch
            imgs_array = np.array(batch_imgs).astype('float32')
            labels_array = np.array(batch_labels).astype('float32').reshape(-1, 1)
            yield imgs_array, labels_array

    return reader


# In[99]:


img = cv2.imread('dataset/PALM-Training400/PALM-Training400/H0002.jpg')
img = transform_img(img)


# In[102]:


def inference(model, params_file_path,img_file):
    with fluid.dygraph.guard():
        print('start evaluation .......')
        #加载模型参数
        model_state_dict, _ = fluid.load_dygraph(params_file_path)
        model.load_dict(model_state_dict)
        model.eval()
        img = cv2.imread(img_file)
        img = transform_img(img)
        pred=model(fluid.dygraph.to_variable(img.reshape(1,3,224,224)))
        pred = fluid.layers.sigmoid(pred)
        pred=(pred.numpy()>0.5).astype('int64')
        print(pred)


# In[104]:


# 创建模型
place=fluid.CPUPlace()
with fluid.dygraph.guard(place):
    model = ResNet()
inference(model,'res_3200-ONCCD','dataset/PALM-Training400/PALM-Training400/H0002.jpg')


# ### 连接可视化

# In[ ]:


import numpy as np
from PIL import Image
from visualdl import LogWriter
import os

#确保路径为'/home/aistudio'
os.chdir('/home/aistudio')

#创建 LogWriter 对象，将图像数据存放在 `./log/train`路径下
from visualdl import LogWriter
log_writer = LogWriter("./log/train")

#导入所需展示的图片
img1 = Image.open('dataset/PALM-Training400/PALM-Training400/N0012.jpg')
img2 = Image.open('dataset/PALM-Training400/PALM-Training400/P0095.jpg')

#将图片转化成array格式
img_n1=np.asarray(img1)
img_n2=np.asarray(img2)

#将图片数据打点至日志文件
log_writer.add_image(tag='图像样本/正样本',img=img_n2, step=5)
log_writer.add_image(tag='图像样本/负样本',img=img_n1, step=5)


# ### 使用LeNet网络进行眼疾分类

# In[ ]:


# # 导入需要的包
# import paddle
# import paddle.fluid as fluid
# import numpy as np
# from paddle.fluid.dygraph.nn import Conv2D, Pool2D, Linear

# # 定义 LeNet 网络结构
# class LeNet(fluid.dygraph.Layer):
#     def __init__(self, num_classes=1):
#         super(LeNet, self).__init__()

#         # 创建卷积和池化层块，每个卷积层使用Sigmoid激活函数，后面跟着一个2x2的池化
#         self.conv1 = Conv2D(num_channels=3, num_filters=6, filter_size=5, act='sigmoid')
#         self.pool1 = Pool2D(pool_size=2, pool_stride=2, pool_type='max')
#         self.conv2 = Conv2D(num_channels=6, num_filters=16, filter_size=5, act='sigmoid')
#         self.pool2 = Pool2D(pool_size=2, pool_stride=2, pool_type='max')
#         # 创建第3个卷积层
#         self.conv3 = Conv2D(num_channels=16, num_filters=120, filter_size=4, act='sigmoid')
#         # 创建全连接层，第一个全连接层的输出神经元个数为64， 第二个全连接层输出神经元个数为分类标签的类别数
#         self.fc1 = Linear(input_dim=300000, output_dim=64, act='sigmoid')
#         self.fc2 = Linear(input_dim=64, output_dim=num_classes)
#     # 网络的前向计算过程，定义输出每一层的结果，后续将结果写入VisualDL日志文件，实现每一层输出图片的可视化
#     def forward(self, x):
#         x1 = self.conv1(x)
#         x2 = self.pool1(x1)
#         x3 = self.conv2(x2)
#         x4 = self.pool2(x3)
#         x5 = self.conv3(x4)
#         x6 = fluid.layers.reshape(x5, [x5.shape[0], -1])
#         x7 = self.fc1(x6)
#         x8 = self.fc2(x7)
#         conv=[x,x1,x2,x3,x4,x5,x6,x7,x8]
#         return x8,conv


# ### 查看数据形状

# In[ ]:


# 查看数据形状
DATADIR = '/home/aistudio/dataset/PALM-Training400/PALM-Training800-color'
train_loader = data_loader(DATADIR, 
                           batch_size=10, mode='train')
data_reader = train_loader()
data = next(data_reader)
data[0].shape, data[1].shape


# ### Lenet

# In[ ]:


# # -*- coding: utf-8 -*-

# # LeNet 识别眼疾图片

# import os
# import random
# import paddle
# import paddle.fluid as fluid
# import numpy as np

# #创建日志文件，储存lenet训练结果
# log_writer = LogWriter("./log/lenet")

# #定义文件路径
# DATADIR = '/home/aistudio/dataset/PALM-Training400/PALM-Training400'
# DATADIR2 = '/home/aistudio/dataset/PALM-Validation400'
# CSVFILE = '/home/aistudio/dataset/PALM-Validation-GT/labels.csv'

# # 定义训练过程
# def train(model):
#     with fluid.dygraph.guard():
#         print('start training ... ')
#         model.train()
#         epoch_num = 10
#         iter=0
#         # 定义优化器
#         opt = fluid.optimizer.Momentum(learning_rate=0.001, momentum=0.9, parameter_list=model.parameters())
#         # 定义数据读取器，训练数据读取器和验证数据读取器
#         train_loader = data_loader(DATADIR, batch_size=10, mode='train')
#         valid_loader = valid_data_loader(DATADIR2, CSVFILE)
#         for epoch in range(epoch_num):
#             for batch_id, data in enumerate(train_loader()):
#                 x_data, y_data = data
#                 img = fluid.dygraph.to_variable(x_data)
#                 label = fluid.dygraph.to_variable(y_data)
#                 # 运行模型前向计算，得到预测值
#                 logits,conv = model(img)
#                 pred = fluid.layers.sigmoid(logits)
#                 pred2 = pred * (-1.0) + 1.0
#                 pred = fluid.layers.concat([pred2, pred], axis=1)
#                 #将每一层输出的图片数据转成numpy array格式并写入日志文件
#                 log_writer.add_image(tag='input_lenet/original', img=conv[0].numpy(), step=batch_id)
#                 log_writer.add_image(tag='input_lenet/conv_1', img=conv[1].numpy(), step=batch_id)
#                 log_writer.add_image(tag='input_lenet/pool_1', img=conv[2].numpy(), step=batch_id)
#                 log_writer.add_image(tag='input_lenet/conv_2', img=conv[3].numpy(), step=batch_id)
#                 log_writer.add_image(tag='input_lenet/pool_2', img=conv[4].numpy(), step=batch_id)
#                 log_writer.add_image(tag='input_lenet/conv_3', img=conv[5].numpy(), step=batch_id)
#                 log_writer.add_image(tag='input_lenet/reshape', img=conv[6].numpy(), step=batch_id)
#                 log_writer.add_image(tag='input_lenet/fc1', img=conv[7].numpy(), step=batch_id)
#                 log_writer.add_image(tag='input_lenet/fc2', img=conv[8].numpy(), step=batch_id)
#                 #计算accuracy
#                 acc = fluid.layers.accuracy(pred, fluid.layers.cast(label, dtype='int64'))
#                 # 进行loss计算
#                 loss = fluid.layers.sigmoid_cross_entropy_with_logits(logits, label)
#                 avg_loss = fluid.layers.mean(loss)
#                 #训练过程中插入作图语句，当每10个batch训练完成后，将当前损失、准确率作为一个新增的数据点储存到记录器中。
#                 if batch_id % 10 == 0:
#                     log_writer.add_scalar(tag='train/acc', step=iter, value=acc.numpy())
#                     log_writer.add_scalar(tag='train/loss', step=iter, value=avg_loss.numpy())

#                     iter+=10
#                     print("epoch: {}, batch_id: {}, loss is: {}".format(epoch, batch_id, avg_loss.numpy(),))
#                 # 反向传播，更新权重，清除梯度
#                 avg_loss.backward()
#                 opt.minimize(avg_loss)
#                 model.clear_gradients()

#             model.eval()
#             accuracies = []
#             losses = []
#             for batch_id, data in enumerate(valid_loader()):
#                 x_data, y_data = data
#                 img = fluid.dygraph.to_variable(x_data)
#                 label = fluid.dygraph.to_variable(y_data)
#                 # 运行模型前向计算，得到预测值
#                 logits,conv = model(img)
#                 # 二分类，sigmoid计算后的结果以0.5为阈值分两个类别
#                 # 计算sigmoid后的预测概率，进行loss计算
#                 pred = fluid.layers.sigmoid(logits)
#                 loss = fluid.layers.sigmoid_cross_entropy_with_logits(logits, label)
#                 avg_loss = fluid.layers.mean(loss)
#                 # 计算预测概率小于0.5的类别
#                 pred2 = pred * (-1.0) + 1.0
#                 # 得到两个类别的预测概率，并沿第一个维度级联
#                 pred = fluid.layers.concat([pred2, pred], axis=1)
#                 acc = fluid.layers.accuracy(pred, fluid.layers.cast(label, dtype='int64'))
#                 accuracies.append(acc.numpy())
#                 losses.append(loss.numpy())
#             print("[validation] accuracy/loss: {}/{}".format(np.mean(accuracies), np.mean(losses)))
#             model.train()
#             # 每轮训练结束后保存模型
#             fluid.save_dygraph(model.state_dict(), 'lenet')
#             # save optimizer state
#             fluid.save_dygraph(opt.state_dict(), 'lenet')


# # 定义评估过程
# def evaluation(model, params_file_path):
#     with fluid.dygraph.guard():
#         print('start evaluation .......')
#         #加载模型参数
#         model_state_dict, _ = fluid.load_dygraph(params_file_path)
#         model.load_dict(model_state_dict)

#         model.eval()
#         eval_loader = load_data('eval')

#         acc_set = []
#         avg_loss_set = []
#         for batch_id, data in enumerate(eval_loader()):
#             x_data, y_data = data
#             img = fluid.dygraph.to_variable(x_data)
#             label = fluid.dygraph.to_variable(y_data)
#             # 计算预测和精度
#             prediction, acc = model(img, label)
#             # 计算损失函数值
#             loss = fluid.layers.cross_entropy(input=prediction, label=label)
#             avg_loss = fluid.layers.mean(loss)
#             acc_set.append(float(acc.numpy()))
#             avg_loss_set.append(float(avg_loss.numpy()))
#         # 求平均精度
#         acc_val_mean = np.array(acc_set).mean()
#         avg_loss_val_mean = np.array(avg_loss_set).mean()

#         print('loss={}, acc={}'.format(avg_loss_val_mean, acc_val_mean))

# # 定义 LeNet 网络结构
# class LeNet(fluid.dygraph.Layer):
#     def __init__(self, num_classes=1):
#         super(LeNet, self).__init__()

#         # 创建卷积和池化层块，每个卷积层使用Sigmoid激活函数，后面跟着一个2x2的池化
#         self.conv1 = Conv2D(num_channels=3, num_filters=6, filter_size=5, act='sigmoid')
#         self.pool1 = Pool2D(pool_size=2, pool_stride=2, pool_type='max')
#         self.conv2 = Conv2D(num_channels=6, num_filters=16, filter_size=5, act='sigmoid')
#         self.pool2 = Pool2D(pool_size=2, pool_stride=2, pool_type='max')
#         # 创建第3个卷积层
#         self.conv3 = Conv2D(num_channels=16, num_filters=120, filter_size=4, act='sigmoid')
#         # 创建全连接层，第一个全连接层的输出神经元个数为64， 第二个全连接层输出神经元个数为分类标签的类别数
#         self.fc1 = Linear(input_dim=300000, output_dim=64, act='sigmoid')
#         self.fc2 = Linear(input_dim=64, output_dim=num_classes)
#     # 网络的前向计算过程，定义输出每一层的结果，后续将结果写入VisualDL日志文件，实现每一层输出图片的可视化
#     def forward(self, x):
#         x1 = self.conv1(x)
#         x2 = self.pool1(x1)
#         x3 = self.conv2(x2)
#         x4 = self.pool2(x3)
#         x5 = self.conv3(x4)
#         x6 = fluid.layers.reshape(x5, [x5.shape[0], -1])
#         x7 = self.fc1(x6)
#         x8 = self.fc2(x7)
#         conv=[x,x1,x2,x3,x4,x5,x6,x7,x8]
#         return x8,conv

# if __name__ == '__main__':
#     # 创建模型
#     with fluid.dygraph.guard():
#         model = LeNet(num_classes=1)

#     train(model)


# ### AlexNet

# 

# In[ ]:


folder_path = "dataset/PALM-Training400"
log_path = "All_model_log_txt"
num_files,file_list = number_of_variation_sample(folder_path)
num_log_files,file_log_list = number_of_variation_sample(log_path)

data_name_dict = {'0': '800-color', 
'1': '1600-overturn-crop-deform', 
'2': '3200-ONCCD',
'3': '1600-imgaug2'}

data_set_dict = {'0': 'dataset/PALM-Training400/PALM-Training800-color', 
'1': 'dataset/PALM-Training400/PALM-Training1600-overturn-crop-deform', 
'2': 'dataset/PALM-Training400/PALM-Training3200-ONCCD',
'3': 'dataset/PALM-Training400/PALM-Training1600-imgaug2'}

log_txt_dict = {'0': 'All_model_log_txt/alexnet_color.txt', 
'1': 'All_model_log_txt/alexnet_ov_cr_de.txt', 
'2': 'All_model_log_txt/alexnet_no_co_cr_fl_di.txt',
'3': 'All_model_log_txt/alexnet_imgaug2.txt'}

for i in range(4):
    path = "home/aistudio/"+data_set_dict[str(i)]
    log_path = "home/aistudio/"+log_txt_dict[str(i)]
    print(path)
    print(log_path)
    print(data_name_dict[str(i)])


# In[ ]:


# -*- coding:utf-8 -*-

# 导入需要的包
import paddle
import paddle.fluid as fluid
import numpy as np
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, Linear


# 定义 AlexNet 网络结构
class AlexNet(fluid.dygraph.Layer):
    def __init__(self, num_classes=1):
        super(AlexNet, self).__init__()
        
        # AlexNet与LeNet一样也会同时使用卷积和池化层提取图像特征
        # 与LeNet不同的是激活函数换成了‘relu’
        self.conv1 = Conv2D(num_channels=3, num_filters=96, filter_size=11, stride=4, padding=5, act='relu')
        self.pool1 = Pool2D(pool_size=2, pool_stride=2, pool_type='max')
        self.conv2 = Conv2D(num_channels=96, num_filters=256, filter_size=5, stride=1, padding=2, act='relu')
        self.pool2 = Pool2D(pool_size=2, pool_stride=2, pool_type='max')
        self.conv3 = Conv2D(num_channels=256, num_filters=384, filter_size=3, stride=1, padding=1, act='relu')
        self.conv4 = Conv2D(num_channels=384, num_filters=384, filter_size=3, stride=1, padding=1, act='relu')
        self.conv5 = Conv2D(num_channels=384, num_filters=256, filter_size=3, stride=1, padding=1, act='relu')
        self.pool5 = Pool2D(pool_size=2, pool_stride=2, pool_type='max')

        self.fc1 = Linear(input_dim=12544, output_dim=4096, act='relu')
        self.drop_ratio1 = 0.5
        self.fc2 = Linear(input_dim=4096, output_dim=4096, act='relu')
        self.drop_ratio2 = 0.5
        self.fc3 = Linear(input_dim=4096, output_dim=num_classes)

        
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.pool1(x1)
        x3 = self.conv2(x2)
        x4 = self.pool2(x3)
        x5 = self.conv3(x4)
        x6 = self.conv4(x5)
        x7 = self.conv5(x6)
        x8 = self.pool5(x7)
        x9 = fluid.layers.reshape(x8, [x8.shape[0], -1])
        x10 = self.fc1(x9)
        # 在全连接之后使用dropout抑制过拟合
        x10= fluid.layers.dropout(x10, self.drop_ratio1)
        x10 = self.fc2(x10)
        # 在全连接之后使用dropout抑制过拟合
        x10 = fluid.layers.dropout(x10, self.drop_ratio2)
        x10 = self.fc3(x10)
        conv=[x,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10]
        return x10, conv


# #### 训练网络

# In[ ]:





for i in range(4):
    dataset_path = "/home/aistudio/"+data_set_dict[str(i)]
    log_txt_path = "/home/aistudio/"+log_txt_dict[str(i)]
    log_path = "./log/"+data_name_dict[str(i)]

    DATADIR = dataset_path # 数据集存放位置
    full_path = log_txt_path  # 新创建的txt文件的存放路径
    file = open(full_path, 'w')
    
    log_writer = LogWriter(log_path) #创建储存vgg结果的日志文件夹
    #创建储存alexnet结果的日志文件夹
    log_writer = LogWriter("./log/alexnet")
    # 定义训练过程
    def train(model):
        with fluid.dygraph.guard():
            print('start training ... ')
            model.train()
            epoch_num = 30
            iter=0
            # 定义优化器
            opt = fluid.optimizer.Momentum(learning_rate=0.001, momentum=0.9, parameter_list=model.parameters())
            # 定义数据读取器，训练数据读取器和验证数据读取器
            train_loader = data_loader(DATADIR, batch_size=10, mode='train')
            
            valid_loader = valid_data_loader(DATADIR2, CSVFILE)
            for epoch in range(epoch_num):
                for batch_id, data in enumerate(train_loader()):
                    x_data, y_data = data
                    img = fluid.dygraph.to_variable(x_data)
                    label = fluid.dygraph.to_variable(y_data)
                    # 运行模型前向计算，得到预测值
                    logits,conv = model(img)
                    pred = fluid.layers.sigmoid(logits)
                    pred2 = pred * (-1.0) + 1.0
                    pred = fluid.layers.concat([pred2, pred], axis=1)
                    #将每一层输出的图片数据转成numpy array格式并写入日志文件
                    log_writer.add_image(tag='input_alexnet/original', img=conv[0].numpy(), step=batch_id)
                    log_writer.add_image(tag='input_alexnet/conv_1', img=conv[1].numpy(), step=batch_id)
                    log_writer.add_image(tag='input_alexnet/pool_1', img=conv[2].numpy(), step=batch_id)
                    log_writer.add_image(tag='input_alexnet/conv_2', img=conv[3].numpy(), step=batch_id)
                    log_writer.add_image(tag='input_alexnet/pool_2', img=conv[4].numpy(), step=batch_id)
                    log_writer.add_image(tag='input_alexnet/conv_3', img=conv[5].numpy(), step=batch_id)
                    log_writer.add_image(tag='input_alexnet/conv_4', img=conv[6].numpy(), step=batch_id)
                    log_writer.add_image(tag='input_alexnet/conv_5', img=conv[7].numpy(), step=batch_id)
                    log_writer.add_image(tag='input_alexnet/pool_5', img=conv[8].numpy(), step=batch_id)
                    log_writer.add_image(tag='input_alexnet/reshape', img=conv[9].numpy(), step=batch_id)
                    log_writer.add_image(tag='input_alexnet/fc', img=conv[10].numpy(), step=batch_id)
                    #计算accuracy
                    acc = fluid.layers.accuracy(pred, fluid.layers.cast(label, dtype='int64'))
                    # 进行loss计算
                    loss = fluid.layers.sigmoid_cross_entropy_with_logits(logits, label)
                    avg_loss = fluid.layers.mean(loss)
                    #训练过程中插入作图语句，当每10个batch训练完成后，将当前损失、准确率作为一个新增的数据点储存到记录器中。
                    if batch_id % 10 == 0:
                        log_writer.add_scalar(tag='train/acc', step=iter, value=acc.numpy())
                        log_writer.add_scalar(tag='train/loss', step=iter, value=avg_loss.numpy())
                        iter+=10
                        print("epoch: {}, batch_id: {}, loss is: {}".format(epoch, batch_id, avg_loss.numpy()))
                    # 反向传播，更新权重，清除梯度
                    avg_loss.backward()
                    opt.minimize(avg_loss)
                    model.clear_gradients()

                model.eval()
                accuracies = []
                losses = []
                for batch_id, data in enumerate(valid_loader()):
                    x_data, y_data = data
                    img = fluid.dygraph.to_variable(x_data)
                    label = fluid.dygraph.to_variable(y_data)
                    # 运行模型前向计算，得到预测值
                    logits,conv = model(img)
                    # 二分类，sigmoid计算后的结果以0.5为阈值分两个类别
                    # 计算sigmoid后的预测概率，进行loss计算
                    pred = fluid.layers.sigmoid(logits)
                    loss = fluid.layers.sigmoid_cross_entropy_with_logits(logits, label)
                    avg_loss = fluid.layers.mean(loss)
                    # 计算预测概率小于0.5的类别
                    pred2 = pred * (-1.0) + 1.0
                    # 得到两个类别的预测概率，并沿第一个维度级联
                    pred = fluid.layers.concat([pred2, pred], axis=1)
                    acc = fluid.layers.accuracy(pred, fluid.layers.cast(label, dtype='int64'))
                    accuracies.append(acc.numpy())
                    losses.append(loss.numpy())
                print("[validation] accuracy/loss: {}/{}".format(np.mean(accuracies), np.mean(losses)))
                file.write("[validation] accuracy: {}  loss: {} \n".format(np.mean(accuracies), np.mean(losses)))
                    
                model.train()
                # save params of model
                save_path = "./All_model_log_txt/"+data_name_dict[str(i)]

                # save params of model
                fluid.save_dygraph(model.state_dict(), save_path)
                # save optimizer state
                fluid.save_dygraph(opt.state_dict(), save_path)


    with fluid.dygraph.guard():
        model = AlexNet()

    train(model)
    file.close()
 


# ### 数据集读取器

# In[ ]:


folder_path = "dataset/PALM-Training400"
log_path = "num-txt"
num_files,file_list = number_of_variation_sample(folder_path)
num_log_files,file_log_list = number_of_variation_sample(log_path)

data_name_dict = {'0': 'PALM-Training400', 
'1': 'PALM-Training800-overturn', 
'2': 'PALM-Training800-noise',
'3': 'PALM-Training800-color', 
'4': 'PALM-Training800-crop', 
'5': 'PALM-Training800-dim',
'6': 'PALM-Training800-deform',
'7': 'PALM-Training1600-overturn_noise_color', 
'8': 'PALM-Training1600-overturn-crop-deform',
'9': 'PALM-Training1600-overturn_dim', 
'10': 'PALM-Training3200-ONCCD', 
'11': 'PALM-Training800-imgaug1',
'12': 'PALM-Training1600-imgaug2'}


data_set_dict = {'0': 'dataset/PALM-Training400/PALM-Training400', 
'1': 'dataset/PALM-Training400/PALM-Training800-overturn', 
'2': 'dataset/PALM-Training400/PALM-Training800-noise',
'3': 'dataset/PALM-Training400/PALM-Training800-color', 
'4': 'dataset/PALM-Training400/PALM-Training800-crop', 
'5': 'dataset/PALM-Training400/PALM-Training800-dim',
'6': 'dataset/PALM-Training400/PALM-Training800-deform',
'7': 'dataset/PALM-Training400/PALM-Training1600-overturn_noise_color', 
'8': 'dataset/PALM-Training400/PALM-Training1600-overturn-crop-deform',
'9': 'dataset/PALM-Training400/PALM-Training1600-overturn_dim', 
'10': 'dataset/PALM-Training400/PALM-Training3200-ONCCD', 
'11': 'dataset/PALM-Training400/PALM-Training800-imgaug1',
'12': 'dataset/PALM-Training400/PALM-Training1600-imgaug2'}

log_txt_dict = {'0': 'num-txt/vgg.txt', 
'1': 'num-txt/vgg_overturn.txt', 
'2': 'num-txt/vgg_noise.txt',
'3': 'num-txt/vgg_color.txt', 
'4': 'num-txt/vgg_crop.txt', 
'5': 'num-txt/vgg_dim.txt',
'6': 'num-txt/vgg_deform.txt',
'7': 'num-txt/vgg_ov_no_co.txt', 
'8': 'num-txt/vgg_ov_cr_de.txt',
'9': 'num-txt/vgg_ov_di.txt', 
'10': 'num-txt/vgg_no_co_cr_fl_di.txt', 
'11': 'num-txt/vgg_imgaug1.txt',
'12': 'num-txt/vgg_imgaug2.txt'}
for i in range(num_files):
    path = "home/aistudio/"+data_set_dict[str(i)]
    log_path = "home/aistudio/"+log_txt_dict[str(i)]
    print(path)
    print(log_path)
    print(data_name_dict[str(i)])
    


# ### VGG

# In[ ]:


# -*- coding:utf-8 -*-
#定义文件路径
DATADIR = '/home/aistudio/dataset/PALM-Training400/PALM-Training400-color'
DATADIR2 = '/home/aistudio/dataset/PALM-Validation400'
CSVFILE = '/home/aistudio/dataset/PALM-Validation-GT/labels.csv'

# VGG模型代码
import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, BatchNorm, Linear
from paddle.fluid.dygraph.base import to_variable

# 定义vgg块，包含多层卷积和1层2x2的最大池化层
class vgg_block(fluid.dygraph.Layer):
    def __init__(self, num_convs, in_channels, out_channels):
        """
        num_convs, 卷积层的数目
        num_channels, 卷积层的输出通道数，在同一个Incepition块内，卷积层输出通道数是一样的
        """
        super(vgg_block, self).__init__()
        self.conv_list = []
        for i in range(num_convs):
            conv_layer = self.add_sublayer('conv_' + str(i), Conv2D(num_channels=in_channels, 
                                        num_filters=out_channels, filter_size=3, padding=1, act='relu'))
            self.conv_list.append(conv_layer)
            in_channels = out_channels
        self.pool = Pool2D(pool_stride=2, pool_size = 2, pool_type='max')
    def forward(self, x):
        for item in self.conv_list:
            x = item(x)
        return self.pool(x)

class VGG(fluid.dygraph.Layer):
    def __init__(self, conv_arch=((2, 64), 
                                (2, 128), (3, 256), (3, 512), (3, 512))):
        super(VGG, self).__init__()
        self.vgg_blocks=[]
        iter_id = 0
        # 添加vgg_block
        # 这里一共5个vgg_block，每个block里面的卷积层数目和输出通道数由conv_arch指定
        in_channels = [3, 64, 128, 256, 512, 512]
        for (num_convs, num_channels) in conv_arch:
            block = self.add_sublayer('block_' + str(iter_id), 
                    vgg_block(num_convs, in_channels=in_channels[iter_id], 
                              out_channels=num_channels))
            self.vgg_blocks.append(block)
            iter_id += 1
        self.fc1 = Linear(input_dim=512*7*7, output_dim=4096,
                      act='relu')
        self.drop1_ratio = 0.5
        self.fc2= Linear(input_dim=4096, output_dim=4096,
                      act='relu')
        self.drop2_ratio = 0.5
        self.fc3 = Linear(input_dim=4096, output_dim=1)
        
    def forward(self, x):
        for item in self.vgg_blocks:
            x = item(x)
        x1 = fluid.layers.reshape(x, [x.shape[0], -1])
        x2 = fluid.layers.dropout(self.fc1(x1), self.drop1_ratio)
        x3 = fluid.layers.dropout(self.fc2(x2), self.drop2_ratio)
        x4 = self.fc3(x3)
        conv=[x,x1,x2,x3,x4]
        return x4,conv


# In[ ]:





# ### 创建文件

# In[ ]:


# # 测试创建文件
# full_path = "/home/aistudio/vgglog_Training400-color.txt"  # 新创建的txt文件的存放路径
# file = open(full_path, 'w')
# # test
# for i in range(30):
#     file.write("msg   \n") 
# #msg也就是下面的Hello world!
# file.close()
# # 调用函数创建一个名为mytxtfile的.txt文件，并向其写入Hello world!


# In[ ]:


# for i in range (13):
#     print(log_txt_dict[str(i)])


# In[ ]:


#创建储存vgg结果的日志文件夹
# log_writer = LogWriter("./log/VGG_Training400-color")
# full_path = "/home/aistudio/vgglog_Training400-color.txt"  # 新创建的txt文件的存放路径
# file = open(full_path, 'w')


for i in range(13):
    dataset_path = "/home/aistudio/"+data_set_dict[str(i)]
    log_txt_path = "/home/aistudio/"+log_txt_dict[str(i)]
    log_path = "./log/"+data_name_dict[str(i)]

    DATADIR = dataset_path # 数据集存放位置
    full_path = log_txt_path  # 新创建的txt文件的存放路径
    file = open(full_path, 'w')
    
    log_writer = LogWriter(log_path) #创建储存vgg结果的日志文件夹

    # 定义训练过程
    def train(model):
        with fluid.dygraph.guard():
            print('start training ... ')
            model.train()
            epoch_num = 30
            iter=0
            # 定义优化器
            opt = fluid.optimizer.Momentum(learning_rate=0.001, momentum=0.9, parameter_list=model.parameters())
            # 定义数据读取器，训练数据读取器和验证数据读取器
            train_loader = data_loader(DATADIR, batch_size=10, mode='train')
            valid_loader = valid_data_loader(DATADIR2, CSVFILE)
            for epoch in range(epoch_num):
                for batch_id, data in enumerate(train_loader()):
                    x_data, y_data = data
                    img = fluid.dygraph.to_variable(x_data)
                    label = fluid.dygraph.to_variable(y_data)
                    # 运行模型前向计算，得到预测值
                    logits,conv = model(img)
                    pred = fluid.layers.sigmoid(logits)
                    pred2 = pred * (-1.0) + 1.0
                    pred = fluid.layers.concat([pred2, pred], axis=1)
                    #将每一层输出的图片数据转成numpy array格式并写入日志文件
                    log_writer.add_image(tag='input_vgg/original', img=conv[0].numpy(), step=batch_id)
                    log_writer.add_image(tag='input_vgg/reshape', img=conv[1].numpy(), step=batch_id)
                    log_writer.add_image(tag='input_vgg/fc1', img=conv[2].numpy(), step=batch_id)
                    log_writer.add_image(tag='input_vgg/fc2', img=conv[3].numpy(), step=batch_id)
                    log_writer.add_image(tag='input_vgg/fc3', img=conv[4].numpy(), step=batch_id)
                    #计算accuracy
                    acc = fluid.layers.accuracy(pred, fluid.layers.cast(label, dtype='int64'))
                    # 进行loss计算
                    loss = fluid.layers.sigmoid_cross_entropy_with_logits(logits, label)
                    avg_loss = fluid.layers.mean(loss)
                    #训练过程中插入作图语句，当每10个batch训练完成后，将当前损失、准确率作为一个新增的数据点储存到记录器中。
                    if batch_id % 10 == 0:
                        log_writer.add_scalar(tag='train/acc', step=iter, value=acc.numpy())
                        log_writer.add_scalar(tag='train/loss', step=iter, value=avg_loss.numpy())
                        iter+=10
                        print("epoch: {}, batch_id: {}, loss is: {}".format(epoch, batch_id, avg_loss.numpy()))
                    # 反向传播，更新权重，清除梯度
                    avg_loss.backward()
                    opt.minimize(avg_loss)
                    model.clear_gradients()

                model.eval()
                accuracies = []
                losses = []
                for batch_id, data in enumerate(valid_loader()):
                    x_data, y_data = data
                    img = fluid.dygraph.to_variable(x_data)
                    label = fluid.dygraph.to_variable(y_data)
                    # 运行模型前向计算，得到预测值
                    logits,conv = model(img)
                    # 二分类，sigmoid计算后的结果以0.5为阈值分两个类别
                    # 计算sigmoid后的预测概率，进行loss计算
                    pred = fluid.layers.sigmoid(logits)
                    loss = fluid.layers.sigmoid_cross_entropy_with_logits(logits, label)
                    avg_loss = fluid.layers.mean(loss)
                    # 计算预测概率小于0.5的类别
                    pred2 = pred * (-1.0) + 1.0
                    # 得到两个类别的预测概率，并沿第一个维度级联
                    pred = fluid.layers.concat([pred2, pred], axis=1)
                    acc = fluid.layers.accuracy(pred, fluid.layers.cast(label, dtype='int64'))
                    accuracies.append(acc.numpy())
                    losses.append(loss.numpy())
                print("[validation] accuracy/loss: {}/{}".format(np.mean(accuracies), np.mean(losses)))
                file.write("[validation] accuracy: {}  loss: {} \n".format(np.mean(accuracies), np.mean(losses)))
                
                model.train()
                # save params of model
                save_path = "./save_model/"+data_name_dict[str(i)]
                fluid.save_dygraph(model.state_dict(), save_path)
                # 保存优化器状态
                fluid.save_dygraph(opt.state_dict(), save_path)

    with fluid.dygraph.guard():
        model = VGG()

    train(model)
    file.close()





# ### 确定数据集后的数据集读取器

# In[ ]:


folder_path = "dataset/PALM-Training400"
log_path = "All_model_log_txt"
num_files,file_list = number_of_variation_sample(folder_path)
num_log_files,file_log_list = number_of_variation_sample(log_path)

data_name_dict = {'0': '800-color', 
'1': '1600-overturn-crop-deform', 
'2': '3200-ONCCD',
'3': '1600-imgaug2'}

data_set_dict = {'0': 'dataset/PALM-Training400/PALM-Training800-color', 
'1': 'dataset/PALM-Training400/PALM-Training1600-overturn-crop-deform', 
'2': 'dataset/PALM-Training400/PALM-Training3200-ONCCD',
'3': 'dataset/PALM-Training400/PALM-Training1600-imgaug2'}

log_txt_dict = {'0': 'All_model_log_txt/google_color.txt', 
'1': 'All_model_log_txt/google_ov_cr_de.txt', 
'2': 'All_model_log_txt/google_no_co_cr_fl_di.txt',
'3': 'All_model_log_txt/google_imgaug2.txt'}

for i in range(4):
    path = "home/aistudio/"+data_set_dict[str(i)]
    log_path = "home/aistudio/"+log_txt_dict[str(i)]
    print(path)
    print(log_path)
    print(data_name_dict[str(i)])
    


# ### GoogleNet

# In[ ]:


class Inception(fluid.dygraph.Layer):
    def __init__(self, c1, c2, c3, c4, **kwargs):
        '''
        Inception模块的实现代码，
        
        c1,  图(b)中第一条支路1x1卷积的输出通道数，数据类型是整数
        c2，图(b)中第二条支路卷积的输出通道数，数据类型是tuple或list, 
               其中c2[0]是1x1卷积的输出通道数，c2[1]是3x3
        c3，图(b)中第三条支路卷积的输出通道数，数据类型是tuple或list, 
               其中c3[0]是1x1卷积的输出通道数，c3[1]是3x3
        c4,  图(b)中第一条支路1x1卷积的输出通道数，数据类型是整数
        '''
        super(Inception, self).__init__()
        # 依次创建Inception块每条支路上使用到的操作
        self.p1_1 = Conv2D(num_filters=c1, 
                           filter_size=1, act='relu')
        self.p2_1 = Conv2D(num_filters=c2[0], 
                           filter_size=1, act='relu')
        self.p2_2 = Conv2D(num_filters=c2[1], 
                           filter_size=3, padding=1, act='relu')
        self.p3_1 = Conv2D(num_filters=c3[0], 
                           filter_size=1, act='relu')
        self.p3_2 = Conv2D(num_filters=c3[1], 
                           filter_size=5, padding=2, act='relu')
        self.p4_1 = Pool2D(pool_size=3, 
                           pool_stride=1,  pool_padding=1, 
                           pool_type='max')
        self.p4_2 = Conv2D(num_filters=c4, 
                           filter_size=1, act='relu')

    def forward(self, x):
        # 支路1只包含一个1x1卷积
        p1 = self.p1_1(x)
        # 支路2包含 1x1卷积 + 3x3卷积
        p2 = self.p2_2(self.p2_1(x))
        # 支路3包含 1x1卷积 + 5x5卷积
        p3 = self.p3_2(self.p3_1(x))
        # 支路4包含 最大池化和1x1卷积
        p4 = self.p4_2(self.p4_1(x))
        # 将每个支路的输出特征图拼接在一起作为最终的输出结果
        return fluid.layers.concat([p1, p2, p3, p4], axis=1)  


# In[ ]:


# -*- coding:utf-8 -*-

# GoogLeNet模型代码
import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, BatchNorm, Linear
from paddle.fluid.dygraph.base import to_variable

# 定义Inception块
class Inception(fluid.dygraph.Layer):
    def __init__(self, c0,c1, c2, c3, c4, **kwargs):
        '''
        Inception模块的实现代码，
        
        c1,  图(b)中第一条支路1x1卷积的输出通道数，数据类型是整数
        c2，图(b)中第二条支路卷积的输出通道数，数据类型是tuple或list, 
               其中c2[0]是1x1卷积的输出通道数，c2[1]是3x3
        c3，图(b)中第三条支路卷积的输出通道数，数据类型是tuple或list, 
               其中c3[0]是1x1卷积的输出通道数，c3[1]是3x3
        c4,  图(b)中第一条支路1x1卷积的输出通道数，数据类型是整数
        '''
        super(Inception, self).__init__()
        # 依次创建Inception块每条支路上使用到的操作
        self.p1_1 = Conv2D(num_channels=c0, num_filters=c1, 
                           filter_size=1, act='relu')
        self.p2_1 = Conv2D(num_channels=c0, num_filters=c2[0], 
                           filter_size=1, act='relu')
        self.p2_2 = Conv2D(num_channels=c2[0], num_filters=c2[1], 
                           filter_size=3, padding=1, act='relu')
        self.p3_1 = Conv2D(num_channels=c0, num_filters=c3[0], 
                           filter_size=1, act='relu')
        self.p3_2 = Conv2D(num_channels=c3[0], num_filters=c3[1], 
                           filter_size=5, padding=2, act='relu')
        self.p4_1 = Pool2D(pool_size=3, 
                           pool_stride=1,  pool_padding=1, 
                           pool_type='max')
        self.p4_2 = Conv2D(num_channels=c0, num_filters=c4, 
                           filter_size=1, act='relu')

    def forward(self, x):
        # 支路1只包含一个1x1卷积
        p1 = self.p1_1(x)
        # 支路2包含 1x1卷积 + 3x3卷积
        p2 = self.p2_2(self.p2_1(x))
        # 支路3包含 1x1卷积 + 5x5卷积
        p3 = self.p3_2(self.p3_1(x))
        # 支路4包含 最大池化和1x1卷积
        p4 = self.p4_2(self.p4_1(x))
        # 将每个支路的输出特征图拼接在一起作为最终的输出结果
        return fluid.layers.concat([p1, p2, p3, p4], axis=1)  
    
class GoogLeNet(fluid.dygraph.Layer):
    def __init__(self):
        super(GoogLeNet, self).__init__()
        # GoogLeNet包含五个模块，每个模块后面紧跟一个池化层
        # 第一个模块包含1个卷积层
        self.conv1 = Conv2D(num_channels=3, num_filters=64, filter_size=7, 
                            padding=3, act='relu')
        # 3x3最大池化
        self.pool1 = Pool2D(pool_size=3, pool_stride=2,  
                            pool_padding=1, pool_type='max')
        # 第二个模块包含2个卷积层
        self.conv2_1 = Conv2D(num_channels=64, num_filters=64, 
                              filter_size=1, act='relu')
        self.conv2_2 = Conv2D(num_channels=64, num_filters=192, 
                              filter_size=3, padding=1, act='relu')
        # 3x3最大池化
        self.pool2 = Pool2D(pool_size=3, pool_stride=2,  
                            pool_padding=1, pool_type='max')
        # 第三个模块包含2个Inception块
        self.block3_1 = Inception(192, 64, (96, 128), (16, 32), 32)
        self.block3_2 = Inception(256, 128, (128, 192), (32, 96), 64)
        # 3x3最大池化
        self.pool3 = Pool2D(pool_size=3, pool_stride=2,  
                               pool_padding=1, pool_type='max')
        # 第四个模块包含5个Inception块
        self.block4_1 = Inception(480, 192, (96, 208), (16, 48), 64)
        self.block4_2 = Inception(512, 160, (112, 224), (24, 64), 64)
        self.block4_3 = Inception(512, 128, (128, 256), (24, 64), 64)
        self.block4_4 = Inception(512, 112, (144, 288), (32, 64), 64)
        self.block4_5 = Inception(528, 256, (160, 320), (32, 128), 128)
        # 3x3最大池化
        self.pool4 = Pool2D(pool_size=3, pool_stride=2,  
                               pool_padding=1, pool_type='max')
        # 第五个模块包含2个Inception块
        self.block5_1 = Inception(832, 256, (160, 320), (32, 128), 128)
        self.block5_2 = Inception(832, 384, (192, 384), (48, 128), 128)
        # 全局池化，尺寸用的是global_pooling，pool_stride不起作用
        self.pool5 = Pool2D(pool_stride=1, 
                               global_pooling=True, pool_type='avg')
        self.fc = Linear(input_dim=1024, output_dim=1, act=None)

    def forward(self, x):
        x1 = self.pool1(self.conv1(x))
        x2 = self.pool2(self.conv2_2(self.conv2_1(x1)))
        x3 = self.pool3(self.block3_2(self.block3_1(x2)))
        x4 = self.block4_3(self.block4_2(self.block4_1(x3)))
        x5 = self.pool4(self.block4_5(self.block4_4(x4)))
        x6 = self.pool5(self.block5_2(self.block5_1(x5)))
        x7 = fluid.layers.reshape(x6, [x6.shape[0], -1])
        x8 = self.fc(x7)
        conv=[x,x1,x2,x3,x4,x5,x6,x7,x8]
        return x8,conv


# In[ ]:


# 执行训练
for i in range(4):
    dataset_path = "/home/aistudio/"+data_set_dict[str(i)]
    log_txt_path = "/home/aistudio/"+log_txt_dict[str(i)]
    log_path = "./log/google_"+data_name_dict[str(i)]

    DATADIR = dataset_path # 数据集存放位置
    full_path = log_txt_path  # 新创建的txt文件的存放路径
    file = open(full_path, 'w')
    
    log_writer = LogWriter(log_path) #创建储存vgg结果的日志文件夹
    DATADIR2 = '/home/aistudio/dataset/PALM-Validation400'

    # 定义训练过程
    def train(model):
        with fluid.dygraph.guard():
            print('start training ... ')
            model.train()
            epoch_num = 30
            iter=0
            # 定义优化器
            opt = fluid.optimizer.Momentum(learning_rate=0.001, momentum=0.9, parameter_list=model.parameters())
            # 定义数据读取器，训练数据读取器和验证数据读取器
            train_loader = data_loader(DATADIR, batch_size=10, mode='train')
            valid_loader = valid_data_loader(DATADIR2, CSVFILE)
            for epoch in range(epoch_num):
                for batch_id, data in enumerate(train_loader()):
                    x_data, y_data = data
                    img = fluid.dygraph.to_variable(x_data)
                    label = fluid.dygraph.to_variable(y_data)
                    # 运行模型前向计算，得到预测值
                    logits,conv = model(img)
                    pred = fluid.layers.sigmoid(logits)
                    pred2 = pred * (-1.0) + 1.0
                    pred = fluid.layers.concat([pred2, pred], axis=1)
                    #将每一层输出的图片数据转成numpy array格式并写入日志文件
                    log_writer.add_image(tag='input_googlenet/original', img=conv[0].numpy(), step=batch_id)
                    log_writer.add_image(tag='input_googlenet/pool_1', img=conv[1].numpy(), step=batch_id)
                    log_writer.add_image(tag='input_googlenet/pool_2', img=conv[2].numpy(), step=batch_id)
                    log_writer.add_image(tag='input_googlenet/pool_3', img=conv[3].numpy(), step=batch_id)
                    log_writer.add_image(tag='input_googlenet/block4_3', img=conv[4].numpy(), step=batch_id)
                    log_writer.add_image(tag='input_googlenet/pool_4', img=conv[5].numpy(), step=batch_id)
                    log_writer.add_image(tag='input_googlenet/pool_5', img=conv[6].numpy(), step=batch_id)
                    log_writer.add_image(tag='input_googlenet/reshape', img=conv[7].numpy(), step=batch_id)
                    log_writer.add_image(tag='input_googlenet/fc', img=conv[8].numpy(), step=batch_id)
                    #计算accuracy
                    acc = fluid.layers.accuracy(pred, fluid.layers.cast(label, dtype='int64'))
                    # 进行loss计算
                    loss = fluid.layers.sigmoid_cross_entropy_with_logits(logits, label)
                    avg_loss = fluid.layers.mean(loss)
                    #训练过程中插入作图语句，当每10个batch训练完成后，将当前损失、准确率作为一个新增的数据点储存到记录器中。
                    if batch_id % 10 == 0:
                        log_writer.add_scalar(tag='train/acc', step=iter, value=acc.numpy())
                        log_writer.add_scalar(tag='train/loss', step=iter, value=avg_loss.numpy())
                        iter+=10
                        print("epoch: {}, batch_id: {}, loss is: {}".format(epoch, batch_id, avg_loss.numpy()))
                    # 反向传播，更新权重，清除梯度
                    avg_loss.backward()
                    opt.minimize(avg_loss)
                    model.clear_gradients()

                model.eval()
                accuracies = []
                losses = []
                for batch_id, data in enumerate(valid_loader()):
                    x_data, y_data = data
                    img = fluid.dygraph.to_variable(x_data)
                    label = fluid.dygraph.to_variable(y_data)
                    # 运行模型前向计算，得到预测值
                    logits,conv = model(img)
                    # 二分类，sigmoid计算后的结果以0.5为阈值分两个类别
                    # 计算sigmoid后的预测概率，进行loss计算
                    pred = fluid.layers.sigmoid(logits)
                    loss = fluid.layers.sigmoid_cross_entropy_with_logits(logits, label)
                    avg_loss = fluid.layers.mean(loss)
                    # 计算预测概率小于0.5的类别
                    pred2 = pred * (-1.0) + 1.0
                    # 得到两个类别的预测概率，并沿第一个维度级联
                    pred = fluid.layers.concat([pred2, pred], axis=1)
                    acc = fluid.layers.accuracy(pred, fluid.layers.cast(label, dtype='int64'))
                    accuracies.append(acc.numpy())
                    losses.append(loss.numpy())
                print("[validation] accuracy/loss: {}/{}".format(np.mean(accuracies), np.mean(losses)))
                file.write("[validation] accuracy: {}  loss: {} \n".format(np.mean(accuracies), np.mean(losses)))
                model.train()

                save_path = "./save_all_model/google_"+data_name_dict[str(i)]
                # save params of model
                fluid.save_dygraph(model.state_dict(), save_path)
                # save optimizer state
                fluid.save_dygraph(opt.state_dict(), save_path)

    with fluid.dygraph.guard():
        model = GoogLeNet()

    train(model)
    file.close()


# In[ ]:


# # 测试创建文件
# full_path = "/home/aistudio/vgglog_Training400-color.txt"  # 新创建的txt文件的存放路径
# file = open(full_path, 'w')
# # test
# for i in range(30):
#     file.write("msg   \n") 
# #msg也就是下面的Hello world!
# file.close()
# # 调用函数创建一个名为mytxtfile的.txt文件，并向其写入Hello world!


# In[ ]:


# dataset_path = "/home/aistudio/"+data_set_dict[str(3)]
# log_txt_path = "/home/aistudio/"+log_txt_dict[str(3)]
# log_path = "./log/google_"+data_name_dict[str(3)]

# DATADIR = dataset_path # 数据集存放位置
# full_path = log_txt_path  # 新创建的txt文件的存放路径
# print(full_path)
# file = open(full_path, 'w')
# for i in range(30):
#     file.write("123    \n")
# file.close()


# In[ ]:


dataset_path = "/home/aistudio/"+data_set_dict[str(3)]
log_txt_path = "/home/aistudio/"+log_txt_dict[str(3)]
log_path = "./log/google_"+data_name_dict[str(3)]

DATADIR = dataset_path # 数据集存放位置
full_path = log_txt_path  # 新创建的txt文件的存放路径
file = open(full_path, 'w')

log_writer = LogWriter(log_path) #创建储存vgg结果的日志文件夹

# 定义训练过程
def train(model):
    with fluid.dygraph.guard():
        print('start training ... ')
        model.train()
        epoch_num = 30
        iter=0
        # 定义优化器
        opt = fluid.optimizer.Momentum(learning_rate=0.001, momentum=0.9, parameter_list=model.parameters())
        # 定义数据读取器，训练数据读取器和验证数据读取器
        train_loader = data_loader(DATADIR, batch_size=10, mode='train')
        valid_loader = valid_data_loader(DATADIR2, CSVFILE)
        for epoch in range(epoch_num):
            for batch_id, data in enumerate(train_loader()):
                x_data, y_data = data
                img = fluid.dygraph.to_variable(x_data)
                label = fluid.dygraph.to_variable(y_data)
                # 运行模型前向计算，得到预测值
                logits,conv = model(img)
                pred = fluid.layers.sigmoid(logits)
                pred2 = pred * (-1.0) + 1.0
                pred = fluid.layers.concat([pred2, pred], axis=1)
                #将每一层输出的图片数据转成numpy array格式并写入日志文件
                log_writer.add_image(tag='input_googlenet/original', img=conv[0].numpy(), step=batch_id)
                log_writer.add_image(tag='input_googlenet/pool_1', img=conv[1].numpy(), step=batch_id)
                log_writer.add_image(tag='input_googlenet/pool_2', img=conv[2].numpy(), step=batch_id)
                log_writer.add_image(tag='input_googlenet/pool_3', img=conv[3].numpy(), step=batch_id)
                log_writer.add_image(tag='input_googlenet/block4_3', img=conv[4].numpy(), step=batch_id)
                log_writer.add_image(tag='input_googlenet/pool_4', img=conv[5].numpy(), step=batch_id)
                log_writer.add_image(tag='input_googlenet/pool_5', img=conv[6].numpy(), step=batch_id)
                log_writer.add_image(tag='input_googlenet/reshape', img=conv[7].numpy(), step=batch_id)
                log_writer.add_image(tag='input_googlenet/fc', img=conv[8].numpy(), step=batch_id)
                #计算accuracy
                acc = fluid.layers.accuracy(pred, fluid.layers.cast(label, dtype='int64'))
                # 进行loss计算
                loss = fluid.layers.sigmoid_cross_entropy_with_logits(logits, label)
                avg_loss = fluid.layers.mean(loss)
                #训练过程中插入作图语句，当每10个batch训练完成后，将当前损失、准确率作为一个新增的数据点储存到记录器中。
                if batch_id % 10 == 0:
                    log_writer.add_scalar(tag='train/acc', step=iter, value=acc.numpy())
                    log_writer.add_scalar(tag='train/loss', step=iter, value=avg_loss.numpy())
                    iter+=10
                    print("epoch: {}, batch_id: {}, loss is: {}".format(epoch, batch_id, avg_loss.numpy()))
                # 反向传播，更新权重，清除梯度
                avg_loss.backward()
                opt.minimize(avg_loss)
                model.clear_gradients()

            model.eval()
            accuracies = []
            losses = []
            for batch_id, data in enumerate(valid_loader()):
                x_data, y_data = data
                img = fluid.dygraph.to_variable(x_data)
                label = fluid.dygraph.to_variable(y_data)
                # 运行模型前向计算，得到预测值
                logits,conv = model(img)
                # 二分类，sigmoid计算后的结果以0.5为阈值分两个类别
                # 计算sigmoid后的预测概率，进行loss计算
                pred = fluid.layers.sigmoid(logits)
                loss = fluid.layers.sigmoid_cross_entropy_with_logits(logits, label)
                avg_loss = fluid.layers.mean(loss)
                # 计算预测概率小于0.5的类别
                pred2 = pred * (-1.0) + 1.0
                # 得到两个类别的预测概率，并沿第一个维度级联
                pred = fluid.layers.concat([pred2, pred], axis=1)
                acc = fluid.layers.accuracy(pred, fluid.layers.cast(label, dtype='int64'))
                accuracies.append(acc.numpy())
                losses.append(loss.numpy())
            print("[validation] accuracy/loss: {}/{}".format(np.mean(accuracies), np.mean(losses)))
            file.write("[validation] accuracy: {}  loss: {} \n".format(np.mean(accuracies), np.mean(losses)))
            model.train()

            save_path = "./save_all_model/google_"+data_name_dict[str(3)]
            # save params of model
            fluid.save_dygraph(model.state_dict(), save_path)
            # save optimizer state
            fluid.save_dygraph(opt.state_dict(), save_path)

with fluid.dygraph.guard():
    model = GoogLeNet()

train(model)
file.close()


# ### Resnet数据准备

# In[ ]:


folder_path = "dataset/PALM-Training400"
log_path = "All_model_log_txt"
num_files,file_list = number_of_variation_sample(folder_path)
num_log_files,file_log_list = number_of_variation_sample(log_path)

data_name_dict = {'0': '800-color', 
'1': '1600-overturn-crop-deform', 
'2': '3200-ONCCD',
'3': '1600-imgaug2'}

data_set_dict = {'0': 'dataset/PALM-Training400/PALM-Training800-color', 
'1': 'dataset/PALM-Training400/PALM-Training1600-overturn-crop-deform', 
'2': 'dataset/PALM-Training400/PALM-Training3200-ONCCD',
'3': 'dataset/PALM-Training400/PALM-Training1600-imgaug2'}

log_txt_dict = {'0': 'All_model_log_txt/Res_color.txt', 
'1': 'All_model_log_txt/Res_ov_cr_de.txt', 
'2': 'All_model_log_txt/Res_no_co_cr_fl_di.txt',
'3': 'All_model_log_txt/Res_imgaug2.txt'}

for i in range(4):
    path = "home/aistudio/"+data_set_dict[str(i)]
    log_path = "home/aistudio/"+log_txt_dict[str(i)]
    print(path)
    print(log_path)
    print(data_name_dict[str(i)])
    


# ### ResNet

# In[ ]:


# -*- coding:utf-8 -*-

# ResNet模型代码
import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, BatchNorm, Linear
from paddle.fluid.dygraph.base import to_variable

# ResNet中使用了BatchNorm层，在卷积层的后面加上BatchNorm以提升数值稳定性
# 定义卷积批归一化块
class ConvBNLayer(fluid.dygraph.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 groups=1,
                 act=None):
        """
        
        num_channels, 卷积层的输入通道数
        num_filters, 卷积层的输出通道数
        stride, 卷积层的步幅
        groups, 分组卷积的组数，默认groups=1不使用分组卷积
        act, 激活函数类型，默认act=None不使用激活函数
        """
        super(ConvBNLayer, self).__init__()

        # 创建卷积层
        self._conv = Conv2D(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            act=None,
            bias_attr=False)

        # 创建BatchNorm层
        self._batch_norm = BatchNorm(num_filters, act=act)

    def forward(self, inputs):
        y = self._conv(inputs)
        y = self._batch_norm(y)
        return y

# 定义残差块
# 每个残差块会对输入图片做三次卷积，然后跟输入图片进行短接
# 如果残差块中第三次卷积输出特征图的形状与输入不一致，则对输入图片做1x1卷积，将其输出形状调整成一致
class BottleneckBlock(fluid.dygraph.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 stride,
                 shortcut=True):
        super(BottleneckBlock, self).__init__()
        # 创建第一个卷积层 1x1
        self.conv0 = ConvBNLayer(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=1,
            act='relu')
        # 创建第二个卷积层 3x3
        self.conv1 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            act='relu')
        # 创建第三个卷积 1x1，但输出通道数乘以4
        self.conv2 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters * 4,
            filter_size=1,
            act=None)

        # 如果conv2的输出跟此残差块的输入数据形状一致，则shortcut=True
        # 否则shortcut = False，添加1个1x1的卷积作用在输入数据上，使其形状变成跟conv2一致
        if not shortcut:
            self.short = ConvBNLayer(
                num_channels=num_channels,
                num_filters=num_filters * 4,
                filter_size=1,
                stride=stride)

        self.shortcut = shortcut

        self._num_channels_out = num_filters * 4

    def forward(self, inputs):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)
        conv2 = self.conv2(conv1)

        # 如果shortcut=True，直接将inputs跟conv2的输出相加
        # 否则需要对inputs进行一次卷积，将形状调整成跟conv2输出一致
        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)

        y = fluid.layers.elementwise_add(x=short, y=conv2)
        layer_helper = LayerHelper(self.full_name(), act='relu')
        return layer_helper.append_activation(y)

# 定义ResNet模型
class ResNet(fluid.dygraph.Layer):
    def __init__(self, layers=50, class_dim=1):
        """
        
        layers, 网络层数，可以是50, 101或者152
        class_dim，分类标签的类别数
        """
        super(ResNet, self).__init__()
        self.layers = layers
        supported_layers = [50, 101, 152]
        assert layers in supported_layers,             "supported layers are {} but input layer is {}".format(supported_layers, layers)

        if layers == 50:
            #ResNet50包含多个模块，其中第2到第5个模块分别包含3、4、6、3个残差块
            depth = [3, 4, 6, 3]
        elif layers == 101:
            #ResNet101包含多个模块，其中第2到第5个模块分别包含3、4、23、3个残差块
            depth = [3, 4, 23, 3]
        elif layers == 152:
            #ResNet50包含多个模块，其中第2到第5个模块分别包含3、8、36、3个残差块
            depth = [3, 8, 36, 3]
        
        # 残差块中使用到的卷积的输出通道数
        num_filters = [64, 128, 256, 512]

        # ResNet的第一个模块，包含1个7x7卷积，后面跟着1个最大池化层
        self.conv = ConvBNLayer(
            num_channels=3,
            num_filters=64,
            filter_size=7,
            stride=2,
            act='relu')
        self.pool2d_max = Pool2D(
            pool_size=3,
            pool_stride=2,
            pool_padding=1,
            pool_type='max')

        # ResNet的第二到第五个模块c2、c3、c4、c5
        self.bottleneck_block_list = []
        num_channels = 64
        for block in range(len(depth)):
            shortcut = False
            for i in range(depth[block]):
                bottleneck_block = self.add_sublayer(
                    'bb_%d_%d' % (block, i),
                    BottleneckBlock(
                        num_channels=num_channels,
                        num_filters=num_filters[block],
                        stride=2 if i == 0 and block != 0 else 1, # c3、c4、c5将会在第一个残差块使用stride=2；其余所有残差块stride=1
                        shortcut=shortcut))
                num_channels = bottleneck_block._num_channels_out
                self.bottleneck_block_list.append(bottleneck_block)
                shortcut = True

        # 在c5的输出特征图上使用全局池化
        self.pool2d_avg = Pool2D(pool_size=7, pool_type='avg', global_pooling=True)

        # stdv用来作为全连接层随机初始化参数的方差
        import math
        stdv = 1.0 / math.sqrt(2048 * 1.0)
        
        # 创建全连接层，输出大小为类别数目
        self.out = Linear(input_dim=2048, output_dim=class_dim,
                      param_attr=fluid.param_attr.ParamAttr(
                          initializer=fluid.initializer.Uniform(-stdv, stdv)))

        
    def forward(self, inputs):
        y = self.conv(inputs)
        y = self.pool2d_max(y)
        for bottleneck_block in self.bottleneck_block_list:
            y = bottleneck_block(y)
        y1 = self.pool2d_avg(y)
        y2 = fluid.layers.reshape(y1, [y1.shape[0], -1])
        y3 = self.out(y2)
        conv=[inputs,y1,y2,y3]
        return y3,conv


# In[ ]:


resnet = ResNet()


# In[ ]:


dataset_path = "/home/aistudio/"+data_set_dict[str(0)]
dataset_path = "/home/aistudio/"+data_set_dict[str(0)]
log_txt_path = "/home/aistudio/"+log_txt_dict[str(0)]
log_path = "./log/res_"+data_name_dict[str(0)]

DATADIR = dataset_path # 数据集存放位置
full_path = log_txt_path  # 新创建的txt文件的存放路径
file = open(full_path, 'w')
    
log_writer = LogWriter(log_path) #创建储存vgg结果的日志文件夹  

DATADIR2 = '/home/aistudio/dataset/PALM-Validation400'
CSVFILE = '/home/aistudio/dataset/PALM-Validation-GT/labels.csv'
    
# 定义训练过程

def train(model):
        with fluid.dygraph.guard():
            print('start training ... ')
            model.train()
            epoch_num = 30
            iter=0
            # 定义优化器
            opt = fluid.optimizer.Momentum(learning_rate=0.001, momentum=0.9, parameter_list=model.parameters())
            # 定义数据读取器，训练数据读取器和验证数据读取器
            train_loader = data_loader(DATADIR, batch_size=10, mode='train')
            valid_loader = valid_data_loader(DATADIR2, CSVFILE)
            for epoch in range(epoch_num):
                for batch_id, data in enumerate(train_loader()):
                    x_data, y_data = data
                    img = fluid.dygraph.to_variable(x_data)
                    label = fluid.dygraph.to_variable(y_data)
                    # 运行模型前向计算，得到预测值
                    logits,conv = model(img)
                    pred = fluid.layers.sigmoid(logits)
                    pred2 = pred * (-1.0) + 1.0
                    pred = fluid.layers.concat([pred2, pred], axis=1)
                    #将每一层输出的图片数据转成numpy array格式并写入日志文件
                    log_writer.add_image(tag='input_resnet/original', img=conv[0].numpy(), step=batch_id)
                    log_writer.add_image(tag='input_resnet/pool2d_avg', img=conv[1].numpy(), step=batch_id)
                    log_writer.add_image(tag='input_resnet/reshape', img=conv[2].numpy(), step=batch_id)
                    log_writer.add_image(tag='input_resnet/output', img=conv[3].numpy(), step=batch_id)
                    #计算accuracy
                    acc = fluid.layers.accuracy(pred, fluid.layers.cast(label, dtype='int64'))
                    # 进行loss计算
                    loss = fluid.layers.sigmoid_cross_entropy_with_logits(logits, label)
                    avg_loss = fluid.layers.mean(loss)
                    #训练过程中插入作图语句，当每10个batch训练完成后，将当前损失、准确率作为一个新增的数据点储存到记录器中。
                    if batch_id % 10 == 0:
                        log_writer.add_scalar(tag='train/acc', step=iter, value=acc.numpy())
                        log_writer.add_scalar(tag='train/loss', step=iter, value=avg_loss.numpy())
                        iter+=10
                        print("epoch: {}, batch_id: {}, loss is: {}".format(epoch, batch_id, avg_loss.numpy()))
                    # 反向传播，更新权重，清除梯度
                    avg_loss.backward()
                    opt.minimize(avg_loss)
                    model.clear_gradients()

                model.eval()
                accuracies = []
                losses = []
                for batch_id, data in enumerate(valid_loader()):
                    x_data, y_data = data
                    img = fluid.dygraph.to_variable(x_data)
                    label = fluid.dygraph.to_variable(y_data)
                    # 运行模型前向计算，得到预测值
                    logits,conv = model(img)
                    # 二分类，sigmoid计算后的结果以0.5为阈值分两个类别
                    # 计算sigmoid后的预测概率，进行loss计算
                    pred = fluid.layers.sigmoid(logits)
                    loss = fluid.layers.sigmoid_cross_entropy_with_logits(logits, label)
                    avg_loss = fluid.layers.mean(loss)
                    # 计算预测概率小于0.5的类别
                    pred2 = pred * (-1.0) + 1.0
                    # 得到两个类别的预测概率，并沿第一个维度级联
                    pred = fluid.layers.concat([pred2, pred], axis=1)
                    acc = fluid.layers.accuracy(pred, fluid.layers.cast(label, dtype='int64'))
                    accuracies.append(acc.numpy())
                    losses.append(loss.numpy())
                print("[validation] accuracy/loss: {}/{}".format(np.mean(accuracies), np.mean(losses)))
                file.write("[validation] accuracy: {}  loss: {} \n".format(np.mean(accuracies), np.mean(losses)))
                model.train()

                save_path = "./save_all_model/res_"+data_name_dict[str(0)]

                model.train()
                # 保存模型
                fluid.save_dygraph(model.state_dict(), save_path)
                # 保存优化器
                fluid.save_dygraph(opt.state_dict(), save_path)

with fluid.dygraph.guard():
    model = ResNet()

train(model)
file.close()


# In[81]:


# for i in range(4):
dataset_path = "/home/aistudio/dataset/PALM-Training400/PALM-Training400"
# log_txt_path = "/home/aistudio/"+log_txt_dict[str(3)]
# log_path = "./log/res_"+data_name_dict[str(3)]

DATADIR = dataset_path # 数据集存放位置
# full_path = log_txt_path  # 新创建的txt文件的存放路径
# file = open(full_path, 'w')

# log_writer = LogWriter(log_path) #创建储存vgg结果的日志文件夹

# 定义训练过程
DATADIR2 = '/home/aistudio/dataset/PALM-Validation400'
CSVFILE = '/home/aistudio/dataset/PALM-Validation-GT/labels.csv'
def train(model):
    with fluid.dygraph.guard():
        print('start training ... ')
        model.train()
        epoch_num = 1
        iter=0
        # 定义优化器
        opt = fluid.optimizer.Momentum(learning_rate=0.001, momentum=0.9, parameter_list=model.parameters())
        # 定义数据读取器，训练数据读取器和验证数据读取器
        train_loader = data_loader(DATADIR, batch_size=10, mode='train')
        valid_loader = valid_data_loader(DATADIR2, CSVFILE)
        for epoch in range(epoch_num):
            for batch_id, data in enumerate(train_loader()):
                x_data, y_data = data
                img = fluid.dygraph.to_variable(x_data)
                label = fluid.dygraph.to_variable(y_data)
                # 运行模型前向计算，得到预测值
                logits,conv = model(img)
                pred = fluid.layers.sigmoid(logits)
                pred2 = pred * (-1.0) + 1.0
                pred = fluid.layers.concat([pred2, pred], axis=1)
                #将每一层输出的图片数据转成numpy array格式并写入日志文件
                # log_writer.add_image(tag='input_resnet/original', img=conv[0].numpy(), step=batch_id)
                # log_writer.add_image(tag='input_resnet/pool2d_avg', img=conv[1].numpy(), step=batch_id)
                # log_writer.add_image(tag='input_resnet/reshape', img=conv[2].numpy(), step=batch_id)
                # log_writer.add_image(tag='input_resnet/output', img=conv[3].numpy(), step=batch_id)
                #计算accuracy
                acc = fluid.layers.accuracy(pred, fluid.layers.cast(label, dtype='int64'))
                # 进行loss计算
                loss = fluid.layers.sigmoid_cross_entropy_with_logits(logits, label)
                avg_loss = fluid.layers.mean(loss)
                #训练过程中插入作图语句，当每10个batch训练完成后，将当前损失、准确率作为一个新增的数据点储存到记录器中。
                if batch_id % 10 == 0:
                    log_writer.add_scalar(tag='train/acc', step=iter, value=acc.numpy())
                    log_writer.add_scalar(tag='train/loss', step=iter, value=avg_loss.numpy())
                    iter+=10
                    print("epoch: {}, batch_id: {}, loss is: {}".format(epoch, batch_id, avg_loss.numpy()))
                # 反向传播，更新权重，清除梯度
                avg_loss.backward()
                opt.minimize(avg_loss)
                model.clear_gradients()

            model.eval()
            accuracies = []
            losses = []
            for batch_id, data in enumerate(valid_loader()):
                x_data, y_data = data
                img = fluid.dygraph.to_variable(x_data)
                label = fluid.dygraph.to_variable(y_data)
                # 运行模型前向计算，得到预测值
                logits,conv = model(img)
                print("logits",logits)
                print("label",label)
                # 二分类，sigmoid计算后的结果以0.5为阈值分两个类别
                # 计算sigmoid后的预测概率，进行loss计算
                pred = fluid.layers.sigmoid(logits)
                loss = fluid.layers.sigmoid_cross_entropy_with_logits(logits, label)
                avg_loss = fluid.layers.mean(loss)
                # 计算预测概率小于0.5的类别
                pred2 = pred * (-1.0) + 1.0
                print("pred2",pred2)
                # 得到两个类别的预测概率，并沿第一个维度级联
                pred = fluid.layers.concat([pred2, pred], axis=1)
                print("pred",pred)
                acc = fluid.layers.accuracy(pred, fluid.layers.cast(label, dtype='int64'))
                accuracies.append(acc.numpy())
                losses.append(loss.numpy())
            print("[validation] accuracy/loss: {}/{}".format(np.mean(accuracies), np.mean(losses)))
            # file.write("[validation] accuracy: {}  loss: {} \n".format(np.mean(accuracies), np.mean(losses)))
            model.train()

            # save_path = "./save_all_model/res_"+data_name_dict[str(3)]

            model.train()
            # save params of model
            fluid.save_dygraph(model.state_dict(), "123")
            # save optimizer state
            fluid.save_dygraph(opt.state_dict(), '123')
with fluid.dygraph.guard():
    model = ResNet()
    
train(model)
# file.close


# ### 模型融合
# #### 加载模型

# In[ ]:


from PIL import Image
import numpy as np
import paddle.fluid as fluid
import os
import shutil


# In[101]:


img = cv2.imread(('/home/aistudio/work/imgs/V0002.jpg')
img = transform_img(img)


# In[89]:





# In[90]:


place=fluid.CPUPlace()
with fluid.dygraph.guard(place):
    model = ResNet()
inference(model,'res_3200-ONCCD','/home/aistudio/work/imgs/V0002.jpg')


# 请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
# Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions. 
