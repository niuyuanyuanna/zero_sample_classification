#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 8/27/18 10:25 PM
# @Author  : NYY
# @Site    : www.niuyuanyuanna@github.io
import torch.utils.data as data
import cv2
import logging
import numpy as np
import torch
from utils.image_aug import aug_img_func


class TestDataLoader(data.Dataset):
    def __init__(self, image_list, class_symbol, config):
        super(TestDataLoader, self).__init__()
        self.image_list = image_list
        self.class_symbol = class_symbol
        self.config = config
        self.input_resolution = self.config.dateset.input_resolution
        self.size = len(image_list)

    def __getitem__(self, index):
        image_path = self.image_list[index]
        image = cv2.imread(image_path)
        image = aug_img_func(image, self.config.test.aug_strategy, self.config)
        image = cv2.resize(image, tuple(self.input_resolution), interpolation=cv2.INTER_LINEAR)
        if self.class_symbol is not None:
            return torch.from_numpy(image.transpose(2, 0, 1).astype(np.float32)), \
                   self.image_list[index], self.class_symbol[index]
        else:
            return torch.from_numpy(image.transpose(2, 0, 1).astype(np.float32)), self.image_list[index]

    def __len__(self):
        return self.size