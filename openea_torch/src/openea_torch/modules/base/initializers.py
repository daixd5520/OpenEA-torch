import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import preprocessing


def init_embeddings(shape, name, init, is_l2_norm, dtype=torch.float32):
    embeds = None
    if init == 'xavier':
        embeds = xavier_init(shape, name, is_l2_norm, dtype=dtype)
    elif init == 'normal':
        embeds = truncated_normal_init(shape, name, is_l2_norm, dtype=dtype)
    elif init == 'uniform':
        embeds = random_uniform_init(shape, name, is_l2_norm, dtype=dtype)
    elif init == 'unit':
        embeds = random_unit_init(shape, name, is_l2_norm, dtype=dtype)
    return embeds


def xavier_init(shape, name, is_l2_norm, dtype=None):
    embeddings = nn.Parameter(nn.init.xavier_uniform_(torch.empty(shape, dtype=dtype)), requires_grad=True)
    return F.normalize(embeddings, p=2, dim=1) if is_l2_norm else embeddings


def truncated_normal_init(shape, name, is_l2_norm, dtype=None):
    std = 1.0 / math.sqrt(shape[1])
    embeddings = nn.Parameter(torch.empty(shape, dtype=dtype).normal_(mean=0, std=std), requires_grad=True)
    return F.normalize(embeddings, p=2, dim=1) if is_l2_norm else embeddings


def random_uniform_init(shape, name, is_l2_norm, minval=0, maxval=None, dtype=None):
    embeddings = nn.Parameter(torch.empty(shape, dtype=dtype).uniform_(minval, maxval), requires_grad=True)
    return F.normalize(embeddings, p=2, dim=1) if is_l2_norm else embeddings


def random_unit_init(shape, name, is_l2_norm, dtype=None):
    vectors = list()
    for i in range(shape[0]):
        vectors.append([random.gauss(0, 1) for j in range(shape[1])])
    embeddings = nn.Parameter(torch.tensor(preprocessing.normalize(np.matrix(vectors)), dtype=dtype), requires_grad=True)
    return F.normalize(embeddings, p=2, dim=1) if is_l2_norm else embeddings


def orthogonal_init(shape, name, dtype=None):
    embeddings = nn.Parameter(nn.init.orthogonal_(torch.empty(shape, dtype=dtype)), requires_grad=True)
    return embeddings