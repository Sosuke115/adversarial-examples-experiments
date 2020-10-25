from sklearn.utils import shuffle
import pickle
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import numpy as np
import copy
import os
import shutil
import importlib
import random
import logging
import os
import random
import re
from collections import Counter
import numpy as np
from tqdm import tqdm
import torch

PAD_TOKEN = "<PAD>"


class Dataset(object):
    def __init__(self, name, instances, label_names):
        self.name = name
        self.instances = instances
        self.label_names = label_names

    def __iter__(self):
        for instance in self.instances:
            yield instance

    def __len__(self):
        return len(self.instances)

    def get_instances(self, fold=None):
        if fold is None:
            return self.instances
        else:
            return [ins for ins in self.instances if ins.fold == fold]


class DatasetInstance(object):
    def __init__(self, text, label, fold):
        self.text = text
        self.label = label
        self.fold = fold


def generate_features(dataset, tokenizer, min_count, max_word_length):
    def create_numpy_sequence(source_sequence, length, dtype):
        ret = np.zeros(length, dtype=dtype)
        source_sequence = source_sequence[:length]
        ret[: len(source_sequence)] = source_sequence
        return ret

    #     logger.info('Creating vocabulary...')
    word_counter = Counter()
    for instance in tqdm(dataset):
        sentence = instance.text.lower()
        tokenized = tokenizer.tokenize(sentence, return_str=True)
        tokenized_list = tokenized.split()
        word_counter.update(token for token in tokenized_list)

    words = [word for word, count in word_counter.items() if count >= min_count]
    word_vocab = {word: index for index, word in enumerate(words, 1)}
    word_vocab[PAD_TOKEN] = 0

    ret = dict(train=[], dev=[], test=[], word_vocab=word_vocab)

    for fold in ("train", "dev", "test"):
        for instance in dataset.get_instances(fold):
            sentence = instance.text.lower()
            tokenized = tokenizer.tokenize(sentence, return_str=True)
            tokenized_list = tokenized.split()
            word_ids = [
                word_vocab[token] for token in tokenized_list if token in word_vocab
            ]
            ret[fold].append(
                dict(
                    word_ids=create_numpy_sequence(word_ids, max_word_length, np.int),
                    label=instance.label,
                )
            )

    return ret


def load_amazon_dataset(lang, options):
    data = {}

    def read(mode):
        x, y = [], []
        porns = ["positive", "negative"]
        for porn in porns:
            with open(
                "data/amazon/" + lang + "/" + mode + "/" + porn + ".txt",
                "r",
                encoding="utf-8",
            ) as f:
                for line in f:
                    if line[-1] == "\n":
                        line = line[:-1]
                    sentence = line.split()
                    x.append(sentence)
                    y.append(porn)

        x, y = shuffle(x, y, random_state=options.random_state)

        if mode == "train":
            dev_idx = len(x) // 8
            data["dev_x"], data["dev_y"] = x[:dev_idx], y[:dev_idx]
            data["train_x"], data["train_y"] = x[dev_idx:], y[dev_idx:]
        else:
            data["test_x"], data["test_y"] = x, y

    read("train")
    read("test")


def load_mldoc_dataset(dataset_path, lang, dev_size=0.05, seed=1):
    data = {}
    instances = []
    categories = ["CCAT", "MCAT", "ECAT", "GCAT"]
    categories_index = {t: i for i, t in enumerate(categories)}

    def read(mode, lang, instances):
        x, y = [], []
        for category in categories:
            path = (
                dataset_path + "/mldoc/" + lang + "/" + mode + "/" + category + ".txt"
            )

            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line[-1] == "\n":
                        line = line[:-1]

                    x.append(line)
                    y.append(categories_index[category])
        x, y = shuffle(x, y, random_state=seed)
        if mode == "train.10000":
            x, y = x[:3000], y[:3000]
            dev_idx = int(len(x) * dev_size)
            instances += [
                DatasetInstance(text, label, "dev")
                for (text, label) in zip(x[:dev_idx], y[:dev_idx])
            ]
            instances += [
                DatasetInstance(text, label, "train")
                for (text, label) in zip(x[dev_idx:], y[dev_idx:])
            ]
        else:
            x, y = x[:3000], y[:3000]
            instances += [
                DatasetInstance(text, label, "test") for (text, label) in zip(x, y)
            ]

    read("train.10000", lang, instances)
    read("test", lang, instances)
    return Dataset("mldoc", instances, categories)


# 複数配列に対応
def shuffle_samples(seed=1, *args):
    np.random.seed(seed)
    zipped = list(zip(*args))
    np.random.shuffle(zipped)
    shuffled = list(zip(*zipped))
    result = []
    for ar in shuffled:
        result.append(ar)
    return result
