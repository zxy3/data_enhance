#!/usr/bin/env python
# coding: utf-8

# 同时添加如下代码, 这样每次环境(kernel)启动的时候只要运行下方代码即可:
# Also add the following code, so that every time the environment (kernel) starts, just run the following code:
# import sys
# sys.path.append('/home/aistudio/external-libraries')

# ### 查看表头

# In[17]:


import pandas as pd
# 读取前5行看下效果
data1 = pd.read_csv('C://Users//18030//Desktop//Bladder_calculi.csv',nrows=5) #读入数据
# 读取全部数据
data = pd.read_csv('C://Users//18030//Desktop//Bladder_calculi.csv') #读入数据
# print(data1.columns)
# print(data1)


# ### 提取第二列和第四列

# In[18]:


data_list = pd.DataFrame(data,columns=(['波数','相对强度']))#将csv文件中flow列中的数据保存到列表中


# In[19]:


# print(data_list)


# In[129]:


import matplotlib.pyplot as plt

x = pd.DataFrame(data,columns=(['波数']))#将csv文件中flow列中的数据保存到列表中
y = pd.DataFrame(data,columns=(['相对强度']))#将csv文件中flow列中的数据保存到列表中
plt.plot(x,y)

plt.savefig("original_spectral_image.png")
plt.show()
# new_data_list = data.loc[(data_list['波数']>-180)&(data_list['波数']<600)]
# new_data_list = pd.DataFrame(new_data_list,columns=(['波数','相对强度']))
# # print(new_data_list)
# x_o = pd.DataFrame(new_data_list,columns=(['波数']))#将csv文件中flow列中的数据保存到列表中
# y_o = pd.DataFrame(new_data_list,columns=(['相对强度']))#将csv文件中flow列中的数据保存到列表中
# plt.plot(x_o,y_o)
# plt.show()
#
#
# # In[91]:
#
#
# print(new_data_list.shape)
# # print(new_data_list)
# import os
# file = open("123.txt",'w')
# for i in range (116):
#     num = str(new_data_list.iat[i,0])+'  '+str(new_data_list.iat[i,1])
#     file.write(num)
#     # file.write(str(new_data_list.iat[i,1]))
#     file.write('\n')
#
# file.close()
#
#
# # In[56]:
#
#
# print(new_data_list.iat[0,1])
#
#
# # In[100]:
#
#
# min_num_x = []
# min_num_y = []
# for i in range (115):
#     if i > 0:
#         if ((new_data_list.iat[i,1]<new_data_list.iat[i-1,1]) & (new_data_list.iat[i,1]<new_data_list.iat[i+1,1])):
#             min_num_x.append(new_data_list.iat[i,0])
#             min_num_y.append(new_data_list.iat[i,1])
#
#
# # In[101]:
#
#
# print(min_num_x,min_num_y)
#
#
# # In[130]:
#
#
# plt.plot(x_o,y_o)
# plt.plot(min_num_x,min_num_y)
# plt.show()
#
#
# # In[134]:
#
#
# print(len(min_num_x) ,len(min_num_y))
#
#
# # In[140]:
#
#
# import numpy as np
# import math
# import matplotlib.pyplot as plt
#
# M=9#多项式阶数
#
# x = np.array(min_num_x)
# y_noise=np.array(min_num_y)
#
# #绿色曲线显示x - y，散点显示x - y_noise
# plt.figure(figsize=(10,8))
# plt.title("")
# plt.plot(x,y_noise,'bo')
#
# X=x
# for i in range(2,M+1):
#          X = np.column_stack((X, pow(x,i)))
#
# #add 1 on the first column of X, now X's shape is (SAMPLE_NUM*(M+1))
# X = np.insert(X,0,[1],1)
#
# #calculate W, W's shape is ((M+1)*1)#
# W=np.linalg.inv((X.T.dot(X))).dot(X.T).dot(y_noise)#have no regularization
# # W=np.linalg.inv((X.T.dot(X))+np.exp(-8) * np.eye(M+1)).dot(X.T).dot(y_noise)#introduce regularization
# y_estimate=X.dot(W)
#
# #红色曲线显示x - y_estimate
# # plt.plot(x_o,y_o,label = '12')
# plt.plot(x_o,y_o,color="orange")
# plt.plot(min_num_x)
# plt.plot(x,y_estimate,linewidth=2)
# plt.show()
