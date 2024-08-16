import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from openea_torch.modules.base.initializers import orthogonal_init
from openea_torch.modules.base.losses import mapping_loss


def add_mapping_module(model):
    model.seed_entities1 = None
    model.seed_entities2 = None

    def set_seed_entities(seed_entities1, seed_entities2):
        model.seed_entities1 = seed_entities1
        model.seed_entities2 = seed_entities2
        tes1 = model.ent_embeds[seed_entities1]
        tes2 = model.ent_embeds[seed_entities2]
        model.mapping_loss = model.args.alpha * mapping_loss(tes1, tes2, model.mapping_mat, model.eye_mat)
        model.mapping_optimizer = generate_optimizer(model.mapping_loss, model.args.learning_rate, model.args.optimizer)
    
    model.set_seed_entities = set_seed_entities


def add_mapping_variables(model):
    model.mapping_mat = orthogonal_init([model.args.dim, model.args.dim], 'mapping_matrix')
    model.eye_mat = torch.eye(model.args.dim, dtype=torch.float32)


# def generate_optimizer(loss, learning_rate, optimizer_type):
#     if optimizer_type == 'adam':
#         return optim.Adam([loss], lr=learning_rate)
#     elif optimizer_type == 'sgd':
#         return optim.SGD([loss], lr=learning_rate)
#     # Add other optimizers as needed
#     else:
#         raise ValueError(f"Unsupported optimizer type: {optimizer_type}")


# # Mockup of the mapping_loss function
# def mapping_loss(tes1, tes2, mapping_mat, eye_mat):
#     mapped_tes2 = tes1 @ mapping_mat
#     map_loss = torch.sum((tes2 - mapped_tes2) ** 2)
#     orthogonal_loss = torch.sum((mapping_mat @ mapping_mat.T - eye_mat) ** 2)
#     return map_loss + orthogonal_loss


# # Example usage:
# class Args:
#     def __init__(self):
#         self.dim = 100
#         self.alpha = 0.5
#         self.learning_rate = 0.001
#         self.optimizer = 'adam'
#         self.num_entities = 1000

# args = Args()
# model = Model(args)
# seed_entities1 = torch.tensor([0, 1, 2], dtype=torch.int64)
# seed_entities2 = torch.tensor([3, 4, 5], dtype=torch.int64)
# model.set_seed_entities(seed_entities1, seed_entities2)