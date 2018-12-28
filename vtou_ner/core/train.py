# -*- coding: utf-8 -*-
"""
create on 2018-12-25 下午10:55

author @heyao
"""
from collections import defaultdict

import numpy as np
import torch
import torch.utils.data

from vtou_ner.utils.predict import tag_ids2entities


class ModelTrainer(object):
    """train validate test the model"""

    def __init__(self, model, optimizer=None, loss_fn=None, validation_data=None,
                 scheduler=None, verbose=1, ordinal_tag=1, id2tag_mapping=None, **kwargs):
        self.model = model
        self.optimizer = optimizer
        self.validation_data = validation_data
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.verbose = verbose
        self.ordinal_tag = ordinal_tag
        self.id2tag_mapping = id2tag_mapping
        self.kwargs = kwargs
        self.history = defaultdict(list)
        if verbose == 1 and id2tag_mapping is None:
            raise ValueError("id2tag_mapping should not be None if verbose is 1.")

    @classmethod
    def from_cfg(cls, model_cfg, optimizer=None, loss_fn=None, validation_data=None,
                 scheduler=None, verbose=1, ordinal_tag=1, id2tag_mapping=None, **kwargs):
        with open(model_cfg, 'rb') as f:
            model = torch.load(f)
        return cls(model, optimizer, loss_fn, validation_data=validation_data,
                   scheduler=scheduler, verbose=verbose, ordinal_tag=ordinal_tag, id2tag_mapping=id2tag_mapping,
                   **kwargs)

    def _train_one_epoch(self, model, X, y, batch_size=32, shuffle=True, epoch=1, use_gpu=False):
        train_loader = self._make_data_loader(X, y, batch_size, shuffle, use_gpu)
        avg_loss = 0.
        for x_batch, y_batch in train_loader:
            if self.scheduler:
                self.scheduler.batch_step()
            # y_pred = model(x_batch)
            loss = self.loss_fn(x_batch, y_batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            avg_loss += loss.item() / len(train_loader)
        self.history['train_loss'].append(avg_loss)
        if not self.verbose:
            return model
        # only verbose
        if self.validation_data is not None:
            X, y = self.validation_data
            self.predict(X, y, batch_size=train_loader.batch_size,
                         shuffle=False, use_gpu=use_gpu, val=True)
            print("Epoch:", epoch, "train loss:", avg_loss, "val loss:", self.history['val_loss'][-1], "val f1:",
                  self.history['val_f1'][-1])
        else:
            print("Epoch:", epoch, "train loss:", avg_loss)
        return model

    @staticmethod
    def _make_data_loader(X, y=None, batch_size=32, shuffle=False, use_gpu=False):
        x_tensor = torch.tensor(X, dtype=torch.long)
        if y is not None:
            y_tensor = torch.tensor(y, dtype=torch.long)
        if use_gpu:
            x_tensor = x_tensor.cuda()
            if y is not None:
                y_tensor = y_tensor.cuda()
        if y is None:
            data = torch.utils.data.TensorDataset(x_tensor)
        else:
            data = torch.utils.data.TensorDataset(x_tensor, y_tensor)
        data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle)
        return data_loader

    def fit(self, X, y, epochs=1, batch_size=32, shuffle=False, use_gpu=False):
        if self.optimizer is None or self.loss_fn is None:
            raise RuntimeError("optimizer, loss_fn must not be None.")
        self.model.train()
        for epoch in range(1, epochs + 1):
            self.model = self._train_one_epoch(self.model, X, y, batch_size=batch_size, shuffle=shuffle, epoch=epoch,
                                               use_gpu=use_gpu)

    def predict(self, X, y=None, batch_size=32, shuffle=False, use_gpu=False, val=False):
        avg_loss = 0.
        self.model.eval()
        preds = []
        data_loader = self._make_data_loader(X, y, batch_size=batch_size, shuffle=shuffle, use_gpu=use_gpu)
        for batch_data in data_loader:
            y_batch = None
            if val:
                x_batch, y_batch = batch_data
            else:
                x_batch = batch_data[0]
            y_pred = self.model(x_batch)
            preds.extend(list(y_pred.cpu().numpy()))
            loss = self.loss_fn(x_batch, y_batch).item() if val else 0.
            avg_loss += loss / len(data_loader)
        if val:
            _, _, val_f1 = self.evaluate(self.validation_data[1], preds)
            self.history['val_loss'].append(avg_loss)
            self.history['val_f1'].append(val_f1)
        return np.array(preds)

    def save(self, filename):
        with open(filename, 'wb') as f:
            torch.save(self.model, f)

    def evaluate(self, y_true, y_pred):
        tp = 0
        fp = 0
        fn = 0
        for y_t, y_p in zip(y_true, y_pred):
            tag_true = set(tag_ids2entities(y_t, self.id2tag_mapping))
            tag_pred = set(tag_ids2entities(y_p, self.id2tag_mapping))
            tp += len(tag_true & tag_pred)
            fp += len(tag_pred - tag_true)
            fn += len(tag_true - tag_pred)
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        f1 = 2 * recall * precision / (recall + precision)
        return recall, precision, f1

    def plot(self):
        pass
