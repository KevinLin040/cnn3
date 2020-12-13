# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 11:55:03 2020

@author: kevin
"""
import os
import shutil
import random

#設定三個目錄比率
train_rate = 0.7
val_rate = 0.2
test_rate = 0.1
seed = 10

#來源目錄及目的目錄設定
source_dataset_dir = './data_set/chest_xray'
target_dataset_dir = './data_set/chest_xray721'
#建立目的目錄
if not os.path.exists('{}'.format(target_dataset_dir)):
        os.makedirs('{}'.format(target_dataset_dir))
#設定三目錄相關數值
dir_lists = ['train','val','test']
rate_lists = [train_rate,val_rate,test_rate]
#建立遞增加總比率，用於取出檔名區間
rate_sum = []
s = 0
for r in rate_lists:
    s = s + r
    rate_sum.append(s)
#建立三目錄
i = 1    
for dir_list, rate, sums in zip(dir_lists, rate_lists, rate_sum):
    if not os.path.exists('./{}/{}'.format(target_dataset_dir, dir_list)):
            os.makedirs('./{}/{}'.format(target_dataset_dir, dir_list))
    #用迴圈建立所有目錄
    for root,dirs,files in os.walk(source_dataset_dir):
        root = root.replace('\\','/')
        if dirs:
            for d in dirs:
                target_root = root.replace(source_dataset_dir,'{}/{}'.format(target_dataset_dir, dir_list))
                target_dir = '{}/{}'.format(target_root,d)
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)
        if files:
            #按比率算出抓取的區間
            files_start = int(len(files) * sums) - int(len(files) * rate)
            files_end = int(len(files) * sums)
            #打亂順序
            random.seed(seed)
            random.shuffle(files)
            files_part = files[files_start:files_end]
            #復制檔案
            for f in files_part:
                source_file = '{}/{}'.format(root,f)
                target_file = source_file.replace(source_dataset_dir,'{}/{}'.format(target_dataset_dir, dir_list))
                shutil.copy(source_file,target_file)
                print(target_file)
                i = i + 1
print('done. copyed {} files'.format(i))


