# -*- coding: utf-8 -*-
"""
create on 2018-12-27 下午6:29

author @heyao
"""
import numpy as np


def chars2ids(chars, char2index):
    return np.array([[char2index[i] for i in s] for s in chars])
