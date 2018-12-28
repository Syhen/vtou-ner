# -*- coding: utf-8 -*-
"""
create on 2018-12-28 下午12:33

author @heyao
"""
from collections import defaultdict
import os
import pickle

from vtou_ner import base_path
from vtou_ner.core.train import ModelTrainer
from vtou_ner.preprocessing.corpus import prepare_model_input, build_char2index
from vtou_ner.utils.predict import tag_ids2entities_names
from vtou_ner.utils.preprocessing import text2sentences


def predict_paragraph(text, trainer, char2id_mapping, id2tag_mapping):
    sentences = list(text2sentences(text))
    x = []
    for sentence in sentences:
        x.append(prepare_model_input(sentence, char2id_mapping))
    predict_ids = trainer.predict(x, batch_size=1)
    entities = defaultdict(set)
    for predict_id, sentence in zip(predict_ids, sentences):
        entity_name = tag_ids2entities_names(sentence, predict_id, id2tag_mapping)
        for key in entity_name:
            entities[key] = entities[key] ^ set(entity_name[key])
    return entities


if __name__ == '__main__':
    from argparse import ArgumentParser

    with open(os.path.join(base_path, "bin/models/id2tag.pkl"), "rb") as f:
        id2tag = pickle.load(f)
    with open(os.path.join(base_path, "bin/models/vocab.pkl"), "rb") as f:
        vocab = pickle.load(f)
    char2id_mapping = build_char2index(vocab)
    trainer = ModelTrainer.from_cfg(os.path.join(base_path, "bin/models/boson-ner.pkl"), id2tag_mapping=id2tag)

    parser = ArgumentParser()
    parser.add_argument("-t", "--text", nargs="+", help="text to ")

    args = parser.parse_args()

    texts = args.text
    for text in texts:
        print(text)
        print(predict_paragraph(text, trainer, char2id_mapping, id2tag))
        print("=" * 60)
