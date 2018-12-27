# -*- coding: utf-8 -*-
"""
create on 2018-12-27 下午4:33

author @heyao
"""
import numpy as np


def pad_sequence(sequence, maxlen=None, left_pad=True, padding_val=0):
    if len(sequence) >= maxlen:
        return sequence[:maxlen]
    if not left_pad:
        return sequence + [padding_val] * (maxlen - len(sequence))
    return [padding_val] * (maxlen - len(sequence)) + sequence


def pad_sequences(sequences, maxlen=None, left_pad=True, padding_val=0):
    max_seq_len = maxlen if maxlen is not None else max(map(len, sequences))
    seqs = []

    for seq in sequences:
        seqs.append(pad_sequence(seq, max_seq_len, left_pad, padding_val))
    return np.array(seqs)


if __name__ == '__main__':
    print(pad_sequences(np.array([[2, 3], [4], [1, 2, 3]]), maxlen=2, left_pad=False, padding_val=0))
