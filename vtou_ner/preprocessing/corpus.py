# -*- coding: utf-8 -*-
"""
create on 2018-12-27 下午6:05

author @heyao
"""
from collections import Counter
from collections import OrderedDict
from itertools import chain


def build_vocab(char_lists, top_k=None):
    """build vocab from words list
    :param char_lists: list. list of list
    :param top_k: int. default None. top n word to choose.
    :return: `collections.OrderedDict`
    """
    chars = chain.from_iterable(char_lists)
    chars_counter = Counter(chars)
    chars_counter_sorted = list(sorted(chars_counter.items(), key=lambda x: x[1], reverse=True))[:top_k]
    vocab = OrderedDict(chars_counter_sorted)
    return vocab


def build_char2index(vocab):
    return {char: i for i, char in enumerate(vocab.keys(), 1)}


def build_index2char(vocab):
    return {i: char for i, char in enumerate(vocab.keys(), 1)}


if __name__ == '__main__':
    print(build_vocab(["今", "天", "天", "天", "我", "我", "天"]))
