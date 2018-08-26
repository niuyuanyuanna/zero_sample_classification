#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 8/26/18 2:08 AM
# @Author  : NYY
# @Site    : www.niuyuanyuanna@github.io
import cv2
import math
import random
from PIL import ImageEnhance, Image
import numpy as np


def flip_image(src_image):
    return cv2.flip(src_image, 1)


def rotate_image(src_image, angle):
    width = src_image.shape[1]
    height = src_image.shap[0]
    radian = angle / 180.0 * math.pi
    sin = math.sin(radian)
    cos = math.cos(radian)
    new_width = int(abs(width * cos) + abs(height * sin))
    new_height = int(abs(width * sin) + abs(height * cos))
    rotate_matrix = cv2.getRotationMatrix2D((width/2.0, height/2),angle, 1.0)
    rotate_matrix[0, 2] += (new_width - width) / 2.0
    rotate_matrix[1, 2] += (new_height - height) / 2.0
    dst_img = cv2.wrapAffine(src_image, rotate_matrix,
                             (new_width, new_height),
                             flags=cv2.INTER_LINEAR)
    return dst_img


def random_crop(src_image, max_jitter=5, keep_size=True):
    width = src_image.shape[1]
    height = src_image.shape[0]
    roi = [math.floor(random.uniform(0, max_jitter)),
           math.floor(random.uniform(0, max_jitter))]
    img_roi = src_image[roi[1]:roi[3] + 1, roi[0]:roi[2] + 1, :]
    if keep_size:
        img_roi = cv2.resize(img_roi, (width, height),
                             interpolation=cv2.INTER_LINEAR)
    return img_roi


def random_color(src_image):
    src_image = Image.fromarray(src_image)
    random_factor = np.random.randint(0, 31) / 10.
    color_image = ImageEnhance.Color(src_image).enhance(random_factor)

    random_factor = np.random.randint(10, 21) / 10.
    brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)

    random_factor = np.random.randint(10, 21) / 10.
    contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)

    random_factor = np.random.randint(0, 31) / 10.
    sharp_image = ImageEnhance.Sharpness(contrast_image).enhance(random_factor)
    return sharp_image


def aug_img_func(image, config):
    if config.train.aug_stragety.flip:
        image = flip_image(image)
    if config.train.aug_stragety.random_rotate:
        max_angle = config.train.max_rotate_angle
        rotate_angle = random.randint(-max_angle, max_angle)
        image = rotate_angle(image, rotate_angle)
    if config.train.aug_stragety.random_crop:
        image = random_crop(image)
    if config.train.aug_stragety.random_color:
        image = random_color(image)
    if config.train.aug_stragety.normalize:
        b_g_r_mean = np.tile(np.array(config.dataset.b_g_r_mean).reshape(1, -1),
                             (1, image.shape[0] * image.shape[1])).reshape(image.shape)
        image = np.array(image, dtype=np.float32)
        image = (image - b_g_r_mean) / 255.0
    return image

