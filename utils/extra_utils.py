#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 8/28/18 2:45 PM
# @Author  : NYY
# @Site    : www.niuyuanyuanna@github.io
import os
import logging


def create_logger(log_path=None, log_format='%(asctime)-15s %(message)s'):
    logger = logging.getLogger()
    if log_path is not None:
        while os.path.exists(log_path):
            log_path = log_path[: -4] + '_1.log'
        log_dir = os.path.dirname(log_path)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        logger.handlers = []
        formatter = logging.Formatter(log_format)
        handler = logging.FileHandler(log_path)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logging.basicConfig(level=logging.INFO, format=log_format)
    return logger
