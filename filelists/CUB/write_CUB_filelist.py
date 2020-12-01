import numpy as np
from os import listdir
from os.path import isfile, isdir, join
import os
import json
import random

cwd = os.getcwd() 
data_path = join(cwd,'CUB_200_2011/images')
savedir = './'
dataset_list = ['base','val','novel']

#if not os.path.exists(savedir):
#    os.makedirs(savedir)

folder_list = [f for f in listdir(data_path) if isdir(join(data_path, f))]
# 排序，然后按照先后顺序给定类标放到　label_dict 中。这里排序可以保证类别对应的类标始终恒定。
folder_list.sort()
label_dict = dict(zip(folder_list,range(0,len(folder_list))))
# 该list中的第 i 个元素均存放 第 i 个类别的全部图片名
classfile_list_all = []

for i, folder in enumerate(folder_list):
    folder_path = join(data_path, folder)
    classfile_list_all.append( [ join(folder_path, cf) for cf in listdir(folder_path) if (isfile(join(folder_path,cf)) and cf[0] != '.')])
    # 打乱第 i 类的全部样本
    random.shuffle(classfile_list_all[i])


for dataset in dataset_list:
    file_list = []
    label_list = []
    #　对于每一类数据
    for i, classfile_list in enumerate(classfile_list_all):
        # 划分规则为，对于第 0～9 类，1、5、9为val，3、7为noval，偶数的全部为base；
        # 对于 第10～19类，13、17为val，11、15、19为novel，偶数的全部为base；
        # 其余依次按照上面两个规则划分，最终100个base类，50个val，50个novel类
        if 'base' in dataset:
            if (i%2 == 0):
                file_list = file_list + classfile_list
                label_list = label_list + np.repeat(i, len(classfile_list)).tolist()
        if 'val' in dataset:
            if (i%4 == 1):
                file_list = file_list + classfile_list
                label_list = label_list + np.repeat(i, len(classfile_list)).tolist()
        if 'novel' in dataset:
            if (i%4 == 3):
                file_list = file_list + classfile_list
                label_list = label_list + np.repeat(i, len(classfile_list)).tolist()
    # 对于三个json文件，label_names 存放的都是全部的类标名
    fo = open(savedir + dataset + ".json", "w")
    fo.write('{"label_names": [')
    fo.writelines(['"%s",' % item  for item in folder_list])
    # 将文件的指针置到 文件尾部+0 的位置
    fo.seek(0, os.SEEK_END) 
    # 文件的相对开始位置，
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write('],')

    fo.write('"image_names": [')
    fo.writelines(['"%s",' % item  for item in file_list])
    fo.seek(0, os.SEEK_END) 
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write('],')

    fo.write('"image_labels": [')
    fo.writelines(['%d,' % item  for item in label_list])
    fo.seek(0, os.SEEK_END) 
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write(']}')

    fo.close()
    print("%s -OK" %dataset)
