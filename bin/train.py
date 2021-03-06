# -*- coding: utf-8 -*-
"""
create on 2018-12-25 下午8:34

author @heyao
"""
import json
import os

import numpy as np
import pickle
import torch.optim as optim
from vtou_ner import base_path
from vtou_ner.core.model import BiLSTMCRF
from vtou_ner.core.train import ModelTrainer
from vtou_ner.preprocessing.corpus import build_char2index
from vtou_ner.preprocessing.corpus import build_vocab
from vtou_ner.preprocessing.corpus import content2id
from vtou_ner.preprocessing.sequence import pad_sequences
from vtou_ner.utils.reproduce import set_seed

set_seed(1)
# model hyper parameters
EMBEDDING_DIM = 100
HIDDEN_DIM = 200
EPOCHS = 15
SEQ_LEN = 60

with open(os.path.join(base_path, 'datasets/boson', 'fmt_boson_data_split.json'), 'r') as f:
    data = json.load(f)
X_text = []
X = []
y = []
for i in data:
    X_text.append(''.join(i['texts']))
    X.append(i['texts'])
    y.append(i['tags'])
X_text = np.array(X_text)

vocab = build_vocab(X, top_k=None)
vocab_y = build_vocab(y, top_k=None)
tag_size = len(build_char2index(vocab_y)) + 2
tag2id = build_char2index(vocab_y)
id2tag = dict(zip(tag2id.values(), tag2id.keys()))
# save id2tag mapping and vocab
with open(os.path.join(base_path, "bin/models/id2tag.pkl"), 'wb') as f:
    pickle.dump(id2tag, f, protocol=2)
with open(os.path.join(base_path, "bin/models/vocab.pkl"), 'wb') as f:
    pickle.dump(vocab, f, protocol=2)

X = pad_sequences(content2id(X, vocab=vocab), maxlen=SEQ_LEN, left_pad=False)
y = pad_sequences(content2id(y, vocab=vocab_y), maxlen=SEQ_LEN, left_pad=False)

idx = np.random.permutation(range(X.shape[0]))
train_idx = idx[:int(len(idx) * 0.8)]
valid_idx = idx[int(len(idx) * 0.8): int(len(idx) * 0.9)]
test_idx = idx[int(len(idx) * 0.9):]

x_train, y_train = X[train_idx], y[train_idx]
x_valid, y_valid = X[valid_idx], y[valid_idx]
x_test, y_test = X[test_idx], y[test_idx]

# START_TAG = "<START>"
# STOP_TAG = "<STOP>"
print("vocabulary size:", len(vocab), "tag size:", tag_size)
print("train on", x_train.shape[0],
      "samples, validate on", x_valid.shape[0],
      "samples, test on", x_test.shape[0], "samples.")
model = BiLSTMCRF(SEQ_LEN, len(vocab) + 1, tag_size, dropout_prob=0.5)

optimizer = optim.Adam(model.parameters(), lr=0.005, amsgrad=True)
model_trainer = ModelTrainer(model, optimizer, loss_fn=model.loss, validation_data=(x_valid, y_valid), verbose=1,
                             id2tag_mapping=id2tag)
try:
    model_trainer.fit(x_train, y_train, epochs=EPOCHS, batch_size=32, shuffle=True, use_gpu=False)
except KeyboardInterrupt:
    print("killed by user")
print("save model...")
model_trainer.save(os.path.join(base_path, "bin/models/boson-ner.pkl"))
print("model save successfully.")
test_pred = model_trainer.predict(x_test, batch_size=16)
# print(test_pred)
recall, precision, f1 = model_trainer.evaluate(y_test, test_pred)
print("recall:", recall)
print("precision:", precision)
print("f1 score:", f1)
