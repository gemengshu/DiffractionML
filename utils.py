"""
Created on Wed Feb 7 2018
Load training and testing dataset
@author: mengshu
"""

from __future__ import print_function
import numpy as np
from os.path import exists, join
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, ToTensor
from os import listdir
import os
from PIL import Image
import sys
import random
import xlsxwriter
import config

import cv2 as cv


class DatasetFromFolder(data.Dataset):

    def __init__(self, image_dir, colordim, size, _input_transform=False, _target_transform=False, suffix='.bmp'):

        super(DatasetFromFolder, self).__init__()
        self.suffix = suffix
        self.mask_filenames = [x for x in listdir(image_dir) if any(
            x.endswith(extension) for extension in ['_mask'+self.suffix])]
        self._input_transform = _input_transform
        self._target_transform = _target_transform
        self.image_dir = image_dir

        self.colordim = colordim
        self.size = size

    def load_img(self, filepath, gray_scale = False):

        if gray_scale:
            img = Image.open(filepath).convert('L')
        else:
            img = Image.open(filepath).convert('RGB')
        return img

    def __getitem__(self, index):

        data_suffix = self.suffix
        mask_suffix = '_mask' + self.suffix
        mask_name = join(self.image_dir, self.mask_filenames[index])
        image_name = mask_name.replace(mask_suffix, data_suffix)
        input = self.load_img(image_name)
        target = self.load_img(mask_name, gray_scale = True)
        
        transform = ToTensor()
        input = transform(input)
        target = transform(target)

        if self._input_transform:
            transform = CenterCrop(self.size)
            input = transform(input)
        if self._target_transform:
            transform = CenterCrop(self.size)
            target = transform(target)
        return input, target

    def __len__(self):

        return len(self.mask_filenames)


#
# https://gist.github.com/adewes/5884820
#
def get_random_color(pastel_factor= 0.5):
    return [(x+pastel_factor)/(1.0+pastel_factor) for x in [random.uniform(0, 1.0) for i in [1, 2, 3]]]


def color_distance(c1, c2):
    return sum([abs(x[0]-x[1]) for x in zip(c1, c2)])


def generate_new_color(existing_colors, pastel_factor=0.5):
    max_distance = None
    best_color = None
    for i in range(0, 100):
        color = get_random_color(pastel_factor=pastel_factor)
        if not existing_colors:
            return color
        best_distance = min([color_distance(color, c)
                             for c in existing_colors])
        if not max_distance or best_distance > max_distance:
            max_distance = best_distance
            best_color = color
    return best_color

def tupletostring(data):
    string_data = str(data[0])
    for i in range(1,len(data)):
        string_data = string_data +', '+ str(data[i])
    return string_data
        

def update_excel_results(sheet, frame_num, cell_pattern, frame_pattern):
    # writing header for showing frame number
    sheet.write(0, frame_num + 1, str(frame_num), frame_pattern)
    for bbox in config.bboxes_config:
        y,x,y_len,x_len,_,_,_,_,index = bbox
        index = int(index) + 1
        sheet.write(index, frame_num + 1, tupletostring((y,x,y_len,x_len)), cell_pattern)
    return

def video_crop_to_images(video_path, rect, save_path):
    cap = cv.VideoCapture(video_path)
    h, w, height, width = rect
    num = 0
    while(cap.isOpened()):
        flag, frame_ori = cap.read()
        print(flag)
        if flag is not True:
            break
        frame_new = frame_ori[h:h+height,w:w+width,:]
        cv.imshow("original", frame_ori)
        cv.imshow("new", frame_new)
        cv.imwrite(save_path + '{}.bmp'.format(num), frame_new)
        num = num + 1

    cap.release()
    cv.destroyAllWindows()
    return
