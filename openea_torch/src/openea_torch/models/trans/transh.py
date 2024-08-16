import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F

from ...models.trans.transe import TransE
from ...modules.base.initializers import init_embeddings
from ...modules.base.losses import margin_loss
from ...modules.base.optimizers import generate_optimizer

class TransH(TransE):

    def __init__(self):
        super().__init__()
        self._define_variables()

    def _define_variables(self):
        self.ent_embeds = init_embeddings((self.kgs.entities_num, self.args.dim),
                                          'ent_embeds', self.args.init, self.args.ent_l2_norm)
        self.rel_embeds = init_embeddings((self.kgs.relations_num, self.args.dim),
                                          'rel_embeds', self.args.init, self.args.rel_l2_norm)
        self.normal_vector = init_embeddings((self.kgs.relations_num, self.args.dim),
                                             'normal_vector', self.args.init, True)

        # Wrap the embeddings in PyTorch's Parameter to enable gradient computation.
        self.ent_embeds = Parameter(self.ent_embeds)
        self.rel_embeds = Parameter(self.rel_embeds)
        self.normal_vector = Parameter(self.normal_vector)

    def _define_embed_graph(self):
        # In PyTorch, placeholders are not needed. We use forward method inputs instead.

        # Define the forward method for the loss calculation.
        def forward(self, pos_hs, pos_rs, pos_ts, neg_hs, neg_rs, neg_ts):
            # Embedding lookup is done using indexing.
            phs = self.ent_embeds[pos_hs]
            prs = self.rel_embeds[pos_rs]
            pts = self.ent_embeds[pos_ts]
            nhs = self.ent_embeds[neg_hs]
            nrs = self.rel_embeds[neg_rs]
            nts = self.ent_embeds[neg_ts]
            pos_norm_vec = self.normal_vector[pos_rs]
            neg_norm_vec = self.normal_vector[neg_rs]

            phs = self._calc(phs, pos_norm_vec)
            pts = self._calc(pts, pos_norm_vec)
            nhs = self._calc(nhs, neg_norm_vec)
            nts = self._calc(nts, neg_norm_vec)

            triple_loss = margin_loss(phs, prs, pts, nhs, nrs, nts, self.args.margin, self.args.loss_norm)
            return triple_loss

        self.forward = forward.__get__(self, self.__class__)

    @staticmethod
    def _calc(e, n):
        norm = F.normalize(n, p=2, dim=1)
        return e - (e * norm).sum(dim=1, keepdim=True) * norm
