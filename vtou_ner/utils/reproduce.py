# -*- coding: utf-8 -*-
"""
create on 2018-12-28 上午2:09

author @heyao
"""
import os
import random

import numpy as np
import torch


def set_seed(seed=1, use_cuda=False):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
