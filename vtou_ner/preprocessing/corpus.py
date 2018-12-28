# -*- coding: utf-8 -*-
"""
create on 2018-12-27 下午6:05

author @heyao
"""
from collections import Counter
from collections import OrderedDict
from itertools import chain

from vtou_ner.preprocessing.sequence import pad_sequence
from vtou_ner.preprocessing.text import chars2ids


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


def content2id(data, vocab=None):
    if vocab is None:
        vocab = build_vocab(data, top_k=None)
    char2index = build_char2index(vocab)
    data = chars2ids(data, char2index)
    return data


def prepare_model_input(text, vocab, maxlen=60, left_pad=False, padding_val=0):
    chars = list(text)
    char_ids = [vocab.get(i, 0) for i in chars]
    return pad_sequence(char_ids, maxlen=maxlen, left_pad=left_pad, padding_val=padding_val)


if __name__ == '__main__':
    print(build_vocab(["今", "天", "天", "天", "我", "我", "天"]))
