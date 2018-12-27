# -*- coding: utf-8 -*-
"""
create on 2018-12-25 下午10:29

author @heyao
"""
import numpy as np


def find_index(content, start_index, finder={'。', '！', '？', '……', '\n'}, ignore_quotation=True):
    """找到分句的索引
    :param content: 
    :param start_index: 
    :param finder: 
    :param ignore_quotation: 
    :return: 
    """
    left_quotation = u'“'
    right_quotation = u'”'
    indexes = []
    left_quotation_index = content.find(left_quotation, start_index)
    right_quotation_index = content.find(right_quotation, start_index)
    max_len = max(map(len, finder))

    for sx in finder:
        indexes.append(content.find(sx, start_index))

    while -1 in indexes:
        indexes.remove(-1)

    if not indexes:
        return len(content)

    if not ignore_quotation:
        return min(indexes)
    indexes = np.array(indexes)
    if sum(indexes < left_quotation_index) > 0 or sum(indexes > right_quotation_index) == len(indexes):
        min_index = min(indexes)
        quotation = content[min_index:min_index + max_len]
        return min_index + (quotation in finder) + 1
    return right_quotation_index + 1


def _content2sentence(content, start_index):
    try:
        index = find_index(content, start_index)
    except ValueError as e:
        raise e
    return content[start_index:index], index


def text2sentences(text):
    index = 0
    while index != len(text):
        try:
            s, index = _content2sentence(text, index)
        except ValueError:
            break
        yield s
