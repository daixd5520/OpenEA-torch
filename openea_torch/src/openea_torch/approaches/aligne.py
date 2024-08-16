import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from ..modules.utils.util import load_session  # 如果这是你自定义的函数，还需修改
from ..modules.base.initializers import init_embeddings  # 需要将该函数转换为适用于 PyTorch
from ..modules.base.losses import limited_loss  # 需要将该函数转换为适用于 PyTorch
from ..models.basic_model import BasicModel


class AlignE(BasicModel):

    def __init__(self):
        super().__init__()
        self.session = load_session()  # 假设这是需要的，可能需要进一步修改
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def init(self):
        self._define_variables()
        self._define_embed_graph()
        
        # 使用 PyTorch 的 Xavier 正态初始化器（类似于 TensorFlow 的 'normal' 初始化器）
        nn.init.normal_(self.ent_embeds.weight)
        nn.init.normal_(self.rel_embeds.weight)

        # customize parameters
        assert self.args.init == 'normal'
        assert self.args.alignment_module == 'swapping'
        assert self.args.loss == 'limited'
        assert self.args.neg_sampling == 'truncated'
        assert self.args.optimizer == 'Adagrad'
        assert self.args.eval_metric == 'inner'
        assert self.args.loss_norm == 'L2'

        assert self.args.ent_l2_norm is True
        assert self.args.rel_l2_norm is True

        assert self.args.pos_margin >= 0.0
        assert self.args.neg_margin > self.args.pos_margin

        assert self.args.neg_triple_num > 1
        assert self.args.truncated_epsilon > 0.0
        assert self.args.learning_rate >= 0.01

    def _define_variables(self):
        self.ent_embeds = nn.Embedding(self.kgs.entities_num, self.args.dim).to(self.device)
        self.rel_embeds = nn.Embedding(self.kgs.relations_num, self.args.dim).to(self.device)

    def _define_embed_graph(self):
        # 这里使用 PyTorch 的 Variable 来模拟 TensorFlow 的 placeholder
        self.pos_hs = Variable(torch.LongTensor(), requires_grad=False).to(self.device)
        self.pos_rs = Variable(torch.LongTensor(), requires_grad=False).to(self.device)
        self.pos_ts = Variable(torch.LongTensor(), requires_grad=False).to(self.device)
        self.neg_hs = Variable(torch.LongTensor(), requires_grad=False).to(self.device)
        self.neg_rs = Variable(torch.LongTensor(), requires_grad=False).to(self.device)
        self.neg_ts = Variable(torch.LongTensor(), requires_grad=False).to(self.device)

        # 在 PyTorch 中使用 embedding_lookup 等效的操作是 nn.Embedding 的调用
        phs = self.ent_embeds(self.pos_hs)
        prs = self.rel_embeds(self.pos_rs)
        pts = self.ent_embeds(self.pos_ts)
        nhs = self.ent_embeds(self.neg_hs)
        nrs = self.rel_embeds(self.neg_rs)
        nts = self.ent_embeds(self.neg_ts)

        # 计算三元组损失
        self.triple_loss = limited_loss(phs, prs, pts, nhs, nrs, nts,
                                        self.args.pos_margin, self.args.neg_margin,
                                        self.args.loss_norm, balance=self.args.neg_margin_balance).to(self.device)

        # 生成优化器
        if self.args.optimizer == 'Adagrad':
            self.triple_optimizer = optim.Adagrad(self.parameters(), lr=self.args.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {self.args.optimizer}")

    def forward(self, pos_hs, pos_rs, pos_ts, neg_hs, neg_rs, neg_ts):
        # 前向传播，类似于 _define_embed_graph 中的操作
        phs = self.ent_embeds(pos_hs)
        prs = self.rel_embeds(pos_rs)
        pts = self.ent_embeds(pos_ts)
        nhs = self.ent_embeds(neg_hs)
        nrs = self.rel_embeds(neg_rs)
        nts = self.ent_embeds(neg_ts)

        # 计算损失
        loss = limited_loss(phs, prs, pts, nhs, nrs, nts,
                            self.args.pos_margin, self.args.neg_margin,
                            self.args.loss_norm, balance=self.args.neg_margin_balance)
        return loss

    def train_step(self, pos_hs, pos_rs, pos_ts, neg_hs, neg_rs, neg_ts):
        # 执行单步训练
        self.triple_optimizer.zero_grad()
        loss = self.forward(pos_hs, pos_rs, pos_ts, neg_hs, neg_rs, neg_ts)
        loss.backward()
        self.triple_optimizer.step()
        return loss.item()
