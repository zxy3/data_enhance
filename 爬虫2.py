#-*- coding: utf-8 -*-
import json
import re
import requests
import datetime
from bs4 import BeautifulSoup
import os
import time
def crawl_pic_urls():
    '''
    爬取每个选手的百度百科图片，并保存
    '''
    # with open('work/' + today + '.json', 'r', encoding='UTF-8') as file:
    #     json_array = json.loads(file.read())

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.99 Safari/537.36'
    }

    # for star in json_array:

        # name = star['name']
        # link = star['link']

        # ！！！请在以下完成对每个选手图片的爬取，将所有图片url存储在一个列表pic_urls中！！！
    # url = "http://image.baidu.com/search/index?tn=baiduimage&ps=1&ct=201326592&lm=-1&cl=2&nc=1&ie=utf-8&word=%E8%99%9E%E4%B9%A6%E6%AC%A3"
    pic_urls = []
    for i in range(1,12):
        print(i)
        url = 'https://www.kuyv.cn/star/ningjing/photo/' + str(i) + '/'
        print(url)


        session = requests.Session()
        try:
            response = session.get(url, headers=headers)
            if response.status_code == 200:
                responseTxt = response.text
                # print(response.text)
                # print("ok")
                pass
            # 将一段文档传入BeautifulSoup的构造方法,就能得到一个文档的对象, 可以传入一段字符串

            # soup = BeautifulSoup(response.text, 'lxml')

            # print(responseJson)
            print('hi')
            # print(type(responseTxt))
            str_pat = re.compile(r'bigimg="(.*g)')
            li = str_pat.findall(responseTxt)
            pic_urls = pic_urls + li

        except Exception as e:
            print(e)
            print("cuole")
    print(pic_urls)
    # ！！！根据图片链接列表pic_urls, 下载所有图片，保存在以name命名的文件夹中！！！
    down_pic(pic_urls)

    # if name == '艾依依':
    #     break
def down_pic(pic_urls):
    '''
    根据图片链接列表pic_urls, 下载所有图片，保存在以name命名的文件夹中,
    '''
    path = 'work/'+'pics/'

    if not os.path.exists(path):
      os.makedirs(path)

    for i, pic_url in enumerate(pic_urls):
        try:
            pic = requests.get(pic_url, timeout=15)
            string = str(i + 1) + '.jpg'
            with open(path+string, 'wb') as f:
                f.write(pic.content)
                print('成功下载第%s张图片: %s' % (str(i + 1), str(pic_url)))
        except Exception as e:
            print('下载第%s张图片时失败: %s' % (str(i + 1), str(pic_url)))
            print(e)
            continue

if __name__ == '__main__':
    crawl_pic_urls()