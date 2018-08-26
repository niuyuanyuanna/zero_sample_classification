#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 8/26/18 10:57 PM
# @Author  : NYY
# @Site    : www.niuyuanyuanna@github.io
import logging
from utils.image_aug import *
import cv2


def get_rgb_mean(image_path_list):
    r_channel = 0
    g_channel = 0
    b_channel = 0
    sum_pixel = 0
    for image_path in image_path_list:
        img = cv2.imread(image_path)
        if img.ndim == 3 and img.shape[2] == 3:
            b_channel += np.sum(img[:, :, 0])
            g_channel += np.sum(img[:, :, 1])
            r_channel += np.sum(img[:, :, 2])
        else:
            logging.info('{} is not rgb'.format(image_path))
        sum_pixel += img.shape[0] * img.shape[1]

    r_mean = r_channel / sum_pixel
    g_mean = g_channel / sum_pixel
    b_mean = b_channel / sum_pixel
    logging.info('r, g, b:{}, {}, {}'.formate(r_mean, g_mean, b_mean))
    return b_mean, g_mean, r_mean


def demo_image_aug(image):
    cv2.imshow('src', image)
    flipped_image = flip_image(image.copy())
    cv2.imshow('flip')