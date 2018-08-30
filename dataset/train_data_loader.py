#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 8/26/18 1:58 AM
# @Author  : NYY
import torch.utils.data as data
import cv2
import logging
from utils.image_aug import aug_img_func
import torch
import numpy as np


class TrainDataLoader(data.Dataset):

    def __init__(self, image_list, label_list, config):
        super(TrainDataLoader, self).__init__()
        self.image_list = image_list
        self.label_list = label_list
        self.config = config
        self.input_resolution = self.config.dataset.input_resolution
        self.size = len(image_list)
        logging.info('using %d for train' % (len(self.image_list)))

    def __getitem__(self, index):
        image_path = self.image_list[index]
        image = cv2.imread(image_path)
        image = aug_img_func(image, self.config.train.aug_strategy)
        image = cv2.resize(image, tuple(self.input_resolution), interpolation=cv2.INTER_LINEAR)
        return torch.from_numpy(image.transpose(2, 0, 1).astype(np.float32)), \
               np.array(self.label[index][0]), np.array(self.label[index][1])


    def __len__(self):
        return self.size