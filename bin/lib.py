# -*- coding: utf-8 -*-
# @Author: rlin
# @Date:   2018-07-31 13:13:19
# @Last Modified by:   rlin
# @Last Modified time: 2018-08-02 16:17:48

import logging
import os

import yaml

CONFS = None

def load_confs(confs_path='../confs/config.yaml'):
    # TODO Docstring
    global CONFS

    if CONFS is None:
        try:
            CONFS = yaml.load(open(confs_path))
        except IOError:
            confs_template_path = confs_path + '.template'
            logging.warn(
                'Confs path: {} does not exist. Attempting to load confs template, '
                'from path: {}'.format(confs_path, confs_template_path))
            CONFS = yaml.load(open(confs_template_path))
    return CONFS


def get_conf(conf_name):
    return load_confs()[conf_name]