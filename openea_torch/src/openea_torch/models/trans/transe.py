import torch
import torch.nn as nn
import torch.optim as optim
from openea_torch.modules.utils.util import load_session
from openea_torch.models.basic_model import BasicModel
from openea_torch.modules.base.losses import get_loss_func


class TransE(BasicModel):

    def __init__(self):
        super().__init__()
        self.init()

    def init(self):
        self._define_variables()
        self.session = load_session()

        assert self.args.init == 'normal'
        assert self.args.alignment_module == 'sharing'
        assert self.args.loss == 'margin-based'
        assert self.args.neg_sampling == 'uniform'
        assert self.args.optimizer == 'Adagrad'
        assert self.args.eval_metric == 'inner'
        assert self.args.loss_norm == 'L2'
        assert self.args.ent_l2_norm is True
        assert self.args.rel_l2_norm is True
        assert self.args.neg_triple_num == 1

    def _define_variables(self):
        # Define the entity and relation embeddings
        self.ent_embeds = nn.Embedding(self.args.entity_count, self.args.embedding_dim)
        self.rel_embeds = nn.Embedding(self.args.relation_count, self.args.embedding_dim)
        
        # Initialize weights
        if self.args.init == 'normal':
            nn.init.normal_(self.ent_embeds.weight, mean=0.0, std=0.1)
            nn.init.normal_(self.rel_embeds.weight, mean=0.0, std=0.1)

    def forward(self, pos_hs, pos_rs, pos_ts, neg_hs, neg_rs, neg_ts):
        phs = self.ent_embeds(pos_hs)
        prs = self.rel_embeds(pos_rs)
        pts = self.ent_embeds(pos_ts)
        nhs = self.ent_embeds(neg_hs)
        nrs = self.rel_embeds(neg_rs)
        nts = self.ent_embeds(neg_ts)

        self.triple_loss = get_loss_func(phs, prs, pts, nhs, nrs, nts, self.args)
        return self.triple_loss

    def optimize(self, learning_rate):
        self.triple_optimizer = optim.Adagrad(self.parameters(), lr=learning_rate)

    def step(self, pos_hs, pos_rs, pos_ts, neg_hs, neg_rs, neg_ts):
        self.triple_optimizer.zero_grad()
        loss = self.forward(pos_hs, pos_rs, pos_ts, neg_hs, neg_rs, neg_ts)
        loss.backward()
        self.triple_optimizer.step()
        return loss.item()