#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 8/26/18 10:57 PM
# @Author  : NYY
# @Site    : www.niuyuanyuanna@github.io
import cv2

import logging
import time
import os
import pprint

from config.config import config
from utils.image_aug import *
from utils.extra_utils import create_logger
from dataset.load_data import load_image_and_class_symbol


def get_rgb_statistics(image_path_list):
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

    r_channel = 0
    g_channel = 0
    b_channel = 0
    for image_path in image_path_list:
        img = cv2.imread(image_path)
        if img.ndim == 3 and img.shape[2] == 3:
            b_channel += np.sum(img[:, :, 0]) - b_mean
            g_channel += np.sum(img[:, :, 1]) - g_mean
            r_channel += np.sum(img[:, :, 2]) - r_mean
        else:
            logging.info('{} is not rgb'.format(image_path))

    b_std = math.sqrt(b_channel / sum_pixel)
    g_std = math.sqrt(g_channel / sum_pixel)
    r_std = math.sqrt(r_channel / sum_pixel)

    logging.info('b, g, r mean:{}, {}, {}'.formate(b_mean, g_mean, r_mean))
    logging.info('b, g, r:{}, {}, {}'.formate(b_std, g_mean, r_std))
    return [b_mean, g_mean, r_mean], [b_std, g_std, r_std]


def demo_image_aug(image):
    cv2.imshow('src', image)

    flipped_image = flip_image(image.copy())
    cv2.imshow('flip', flipped_image)

    rotated_image = rotate_image(image.copy(), 20)
    cv2.imshow('rotate', rotated_image)

    cropped_image = random_crop(image.copy())
    cv2.imshow('crop', cropped_image)

    colored_image = random_color(image.copy())
    cv2.imshow('color', colored_image)
    cv2.waitKey(0)


if __name__ == '__main__':
    now_date = time.strftime('%Y-%m-%d', time.localtime(time.time()))
    now_time = time.strftime('%H-%M-%S', time.localtime(time.time()))
    log_path = os.path.join(config.exp, now_date, now_time + '_compute_rgb_mean.log')
    logger = create_logger(log_path)
    logging.info('config:\n{}'.format(pprint.pformat(config)))
    train_image_name, train_label = load_image_and_class_symbol(config.dataset.train_image_path_list,
                                                                config.dataset.train_image_name_and_label_list)
    assert len(train_image_name) == len(train_label)
    logger.info('load %d train data' % len(train_label))

    test_image_name, test_label = load_image_and_class_symbol(config.dataset.test_image_path_list,
                                                              config.dataset.test_image_name_list)
    assert len(test_image_name) == len(test_label)
    logger.info('load  %d test data' % len(test_label))