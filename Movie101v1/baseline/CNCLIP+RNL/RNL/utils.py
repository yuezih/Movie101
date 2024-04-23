import copy

import nltk
import json
from gensim.models import KeyedVectors
import h5py
import numpy as np
from torch import nn
import math
import random


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def video_aug(v_feat, ent, length):
        seed1= random.random()
        _, dim_ft = v_feat.shape
        if seed1 < 0.5:
            fast_forward_ratio = random.uniform(0.5,2)
            target_span_frame = math.ceil((ent[1] - ent[0]) * fast_forward_ratio)
            new_length = length - (ent[1] - ent[0]) + target_span_frame
            new_ent = [ent[0], ent[0]+ target_span_frame]
            ft_new = np.zeros((new_length, dim_ft), np.float32)
            ft_new[:ent[0]] = v_feat[:ent[0]]
            indices = np.round(np.linspace(0, ent[1]-ent[0]-1, target_span_frame)).astype(np.int32)
            ft_new[new_ent[0]:new_ent[1]] = v_feat[ent[0]:ent[1]][indices]
            ft_new[new_ent[1]:] = v_feat[ent[1]:]
            # pdb.set_trace()
            v_feat, ent, length = ft_new, new_ent, new_length

        seed2= random.random()
        if seed2 < 0.5:
            fast_forward_ratio = random.uniform(0.5,2)
            target_span_frame = math.ceil(ent[0] * fast_forward_ratio)
            new_length = length - ent[0] + target_span_frame
            ft_new = np.zeros((new_length, dim_ft), np.float32)
            indices = np.round(np.linspace(0, ent[0]-1, target_span_frame)).astype(np.int32)
            ft_new[:target_span_frame] = v_feat[:ent[0]][indices]
            new_ent = [target_span_frame, target_span_frame+(ent[1] - ent[0])]
            ft_new[target_span_frame:] = v_feat[ent[0]:]
            v_feat, ent, length = ft_new, new_ent, new_length
        seed3= random.random()
        if seed3 < 0.5:
            fast_forward_ratio = random.uniform(0.5,2)
            target_span_frame = math.ceil((length- ent[1]) * fast_forward_ratio)
            new_length = length - (length-ent[1]) + target_span_frame
            ft_new = np.zeros((new_length, dim_ft), np.float32)
            indices = np.round(np.linspace(0, (length- ent[1])-1, target_span_frame)).astype(np.int32)
            new_ent = ent
            ft_new[ent[1]:] = v_feat[ent[1]:][indices]
            ft_new[:ent[1]] = v_feat[:ent[1]]
            v_feat, ent, length = ft_new, new_ent, new_length

        return v_feat, np.asarray(ent).astype(np.int32), length


def load_feature(filename, dataset='ActivityNet'):
    if dataset == 'ActivityNet':
        with h5py.File(filename, 'r') as fr:
            return np.asarray(fr['feature']).astype(np.float32)
    elif dataset == 'TACOS':
        return np.load(filename).astype(np.float32)
    elif dataset == 'Charades':
        return np.load(filename).astype(np.float32)
    elif dataset == 'Didemo':
        with h5py.File(filename, 'r') as fr:
            return np.asarray(fr['feature']).astype(np.float32)
    return None


def load_json(filename):
    with open(filename, encoding='utf8') as fr:
        return json.load(fr)


def load_word2vec(filename, binary=False):
    word2vec = KeyedVectors.load_word2vec_format(filename, binary=binary)
    return word2vec


def tokenize(sentence, word2vec):
    punctuations = ['.', '?', ',', '', '(', ')']
    raw_text = sentence.lower()
    words = nltk.word_tokenize(raw_text)
    words = [word for word in words if word not in punctuations]
    return [word for word in words if word in word2vec]


def generate_anchors(dataset='ActivityNet'):
    if dataset == 'ActivityNet':
        widths = np.array([8, 16, 32, 64])
        center = 7.5
        start = center - 0.5 * (widths - 1)
        end = center + 0.5 * (widths - 1)
    elif dataset == 'TACOS':
        widths = np.array([8, 16, 32, 64])#np.array([6, 18, 32])
        center = 7.5
        start = center - 0.125 * (widths - 1)
        end = center + 0.125 * (widths - 1)
    elif dataset == 'Didemo':
        widths = np.array([8, 16, 32, 64])#np.array([6, 18, 32])
        center = 7.5
        start = center - 0.125 * (widths - 1)
        end = center + 0.125 * (widths - 1)
    elif dataset == 'Charades':
        widths = np.array([16, 24, 32, 40])#np.array([6, 18, 32])
        center = 7.5
        start = center - 0.125 * (widths - 1)
        end = center + 0.125 * (widths - 1)
    else:
        return None
    return np.stack([start, end], -1)


import time


class CountMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = np.zeros([2, 4],dtype=np.float32)
        self.count = 0

    def update(self, val, n=1):
        self.val += val
        self.count += n

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class TimeMeter(object):
    """Computes the average occurrence of some event per second"""

    def __init__(self, init=0):
        self.reset(init)

    def reset(self, init=0):
        self.init = init
        self.start = time.time()
        self.n = 0

    def update(self, val=1):
        self.n += val

    @property
    def avg(self):
        return self.n / self.elapsed_time

    @property
    def elapsed_time(self):
        return self.init + (time.time() - self.start)


class StopwatchMeter(object):
    """Computes the sum/avg duration of some event in seconds"""

    def __init__(self):
        self.reset()

    def start(self):
        self.start_time = time.time()

    def stop(self, n=1):
        if self.start_time is not None:
            delta = time.time() - self.start_time
            self.sum += delta
            self.n += n
            self.start_time = None

    def reset(self):
        self.sum = 0
        self.n = 0
        self.start_time = None

    @property
    def avg(self):
        return self.sum / self.n
