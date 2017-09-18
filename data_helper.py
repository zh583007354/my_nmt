#-*- coding:utf-8 -*-

import os
import numpy as np
import tensorflow as tf
from collections import Counter

class DataSet(object):
    """docstring for DataSet"""
    def __init__(self, data, data_type):
        self.data = data
        self.data_type = data_type
        self.num_examples = len(self.data[0])
        self.in_x = None
        self.in_y = None
        self.ou_y = None

    def vectorize(self, word_dict_en, word_dict_zh, sort_by_len=True, verbose=True):
        
        in_x, in_y, ou_y = [], [], []

        for idx, (en, zh) in enumerate(zip(self.data[0], self.data[1])):
            en_words = en.split(' ')
            zh_words = zh.split(' ')
            seq_en = [word_dict_en[w] if w in word_dict_en else 0 for w in en_words]
            seq_zh_i = [word_dict_zh[w] if w in word_dict_zh else 0 for w in zh_words]
            seq_zh_o = [word_dict_zh[w] if w in word_dict_zh else 0 for w in zh_words]
            seq_zh_i.insert(0, 1)
            seq_zh_o.append(2)
            if (len(seq_en) > 0) and (len(seq_zh_i) > 0):
                in_x.append(seq_en)
                in_y.append(seq_zh_i)
                ou_y.append(seq_zh_o)
            if verbose and (idx % 1000000 == 0):
                print('Vectorization: processed %d / %d' % (idx, self.num_examples))

        def len_argsort(seq):
            return sorted(range(len(seq)), key=lambda x: len(seq[x]), reverse=True)

        if sort_by_len:
            sorted_index = len_argsort(in_x)
            in_x = [in_x[i] for i in sorted_index]
            in_y = [in_y[i] for i in sorted_index]
            ou_y = [ou_y[i] for i in sorted_index]
        self.in_x = in_x
        self.in_y = in_y
        self.ou_y = ou_y

    def gen_minbatches(self, batch_size, start_examples=None, end_examples=None, shuffle=False):
        m = 0
        n = 0
        if start_examples is None:
            m = 0
            n = self.num_examples
        else:
            m = start_examples
            n = end_examples

        idx_list = np.arange(m, n, batch_size)
        if shuffle:
            np.random.shuffle(idx_list)
        minibatches = []
        for idx in idx_list:
            minibatches.append(np.arange(idx, min(idx + batch_size, n)))

        for minibatch in minibatches:
            mb_x = [self.in_x[t] for t in minibatch]
            mb_yi = [self.in_y[t] for t in minibatch]
            mb_yo = [self.ou_y[t] for t in minibatch]
            mb_x, mb_x_mask = prepare_data(mb_x)
            mb_yi, mb_y_mask = prepare_data(mb_yi)
            mb_yo, _ = prepare_data(mb_yo)

            yield (mb_x, mb_yi, mb_yo, mb_x_mask, mb_y_mask)

def load_data(args, data_type=None, max_example=None):

    data_path_en = os.path.join(args.data_dir, "{}.en".format(data_type))
    data_path_zh = os.path.join(args.data_dir, "{}.zhc".format(data_type))

    Englishs = []
    Chineses = []

    num_examples_en = 0
    num_examples_zh = 0

    f1 = open(data_path_en, 'r', encoding='utf-8')
    while True:
        line = f1.readline()
        if not line:
            break
        English = line.strip().lower()
        Englishs.append(English)
        num_examples_en += 1

        if (max_example is not None) and (num_examples_en >= max_example):
            break
    f1.close()
    print('#English examples: %d' % len(Englishs))

    f2 = open(data_path_zh, 'r', encoding='utf-8')
    while True:
        line = f2.readline()
        if not line:
            break
        Chinese = line.strip()
        Chineses.append(Chinese)
        num_examples_zh += 1

        if (max_example is not None) and (num_examples_zh >= max_example):
            break
        
    f2.close()
    if not num_examples_en == num_examples_zh:
        raise ValueError("")
    print('#Chinese examples: %d' % len(Chineses))
    dataset = DataSet((Englishs, Chineses), data_type)
    return dataset

def build_dict(sentences, max_words=50000):

    word_count = Counter()
    for sent in sentences:
        for w in sent.split():
            word_count[w] += 1

    ls = word_count.most_common(max_words)

    print('#Words: %d -> %d' % (len(word_count), len(ls)))
    for key in ls[:5]:
        print(key)
    print('...')
    for key in ls[-5:]:
        print(key)

    # leave 0 to UNK
    # leave 1 to delimiter <s>
    # leave 2 to delimiter </s> 
    word_dict = {w[0]: index + 3 for (index, w) in enumerate(ls)}
    word_dict['<unk>'] = 0
    word_dict['<s>'] = 1
    word_dict['</s>'] = 2
    return word_dict
    # 查查most_common就知道为什么w[0]了.

def gen_embeddings(word_dict, dim, in_file=None):
    """
        Generate an initial embedding matrix for `word_dict`.
        If an embedding file is not given or a word is not in the embedding file,
        a randomly initialized vector will be used.
        50000 * 100
        以词在词典中的序号为索引。
    """

    num_words = max(word_dict.values()) + 1
    embeddings = np.random.normal(0, 1, size=[num_words, dim])
    print('Embeddings: %d x %d' % (num_words, dim))

    if in_file is not None:
        print('Loading embedding file: %s' % in_file)
        pre_trained = 0
        for line in open(in_file, 'r', encoding='utf-8').readlines():
            sp = line.split()
            if len(sp) == 2:
                continue
            assert len(sp) == dim + 1
            if sp[0] in word_dict:
                pre_trained += 1
                embeddings[word_dict[sp[0]]] = [float(x) for x in sp[1:]]
        print('Pre-trained: %d (%.2f%%)' %
                     (pre_trained, pre_trained * 100.0 / num_words))
    return embeddings

def prepare_data(seqs):
    lengths = [len(seq) for seq in seqs]
    n_samples = len(seqs)
    max_len = np.max(lengths)
    x = np.zeros((n_samples, max_len)).astype('int32')
    x_mask = np.zeros((n_samples, max_len)).astype('float')
    for idx, seq in enumerate(seqs):
        x[idx, :lengths[idx]] = seq
        x_mask[idx, :lengths[idx]] = 1.0
    # print x, x_mask
    return x, x_mask