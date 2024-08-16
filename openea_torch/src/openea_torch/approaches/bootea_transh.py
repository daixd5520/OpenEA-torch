import gc
import math
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
import numpy as np

from ..modules.finding.evaluation import early_stop
from ..modules.train import batch as bat
# from ..approaches.aligne import AlignE
from ..modules.utils.util import task_divide # type: ignore
from ..modules.load.kg import KG
from ..modules.utils.util import load_session
from ..modules.base.losses import limited_loss
from ..approaches.bootea import generate_supervised_triples, generate_pos_batch, bootstrapping, \
    calculate_likelihood_mat
from ..models.basic_model import BasicModel

class BootEA_TransH(BasicModel):

    def __init__(self):
        super().__init__()
        self.ref_ent1 = None
        self.ref_ent2 = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def init(self):
        self._define_variables()
        self._define_embed_graph()
        self._define_alignment_graph()
        self._define_likelihood_graph()
        self.to(self.device)
        self.ref_ent1 = self.kgs.valid_entities1 + self.kgs.test_entities1
        self.ref_ent2 = self.kgs.valid_entities2 + self.kgs.test_entities2

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

    def _calc(self, e, n):
        norm = torch.nn.functional.normalize(n, p=2, dim=1)
        return e - torch.sum(e * norm, dim=1, keepdim=True) * norm

    def _define_variables(self):
        self.ent_embeds = self._init_embeddings(self.kgs.entities_num, self.args.dim, self.args.ent_l2_norm)
        self.rel_embeds = self._init_embeddings(self.kgs.relations_num, self.args.dim, self.args.rel_l2_norm)
        self.normal_vector = self._init_embeddings(self.kgs.relations_num, self.args.dim, True)

    def _init_embeddings(self, num_embeddings, embedding_dim, requires_grad):
        embeddings = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.normal_(embeddings.weight, mean=0, std=1)
        embeddings.weight.requires_grad = requires_grad
        return embeddings

    def _define_embed_graph(self):
        self.pos_hs = torch.LongTensor().to(self.device)
        self.pos_rs = torch.LongTensor().to(self.device)
        self.pos_ts = torch.LongTensor().to(self.device)
        self.neg_hs = torch.LongTensor().to(self.device)
        self.neg_rs = torch.LongTensor().to(self.device)
        self.neg_ts = torch.LongTensor().to(self.device)

    def forward(self, hs, rs, ts):
        h_embeds = self.ent_embeds(hs)
        r_embeds = self.rel_embeds(rs)
        t_embeds = self.ent_embeds(ts)
        norm_vec = self.normal_vector(rs)
        h_embeds = self._calc(h_embeds, norm_vec)
        t_embeds = self._calc(t_embeds, norm_vec)
        return h_embeds, r_embeds, t_embeds

    def _define_alignment_graph(self):
        self.new_h = torch.LongTensor().to(self.device)
        self.new_r = torch.LongTensor().to(self.device)
        self.new_t = torch.LongTensor().to(self.device)

    def _define_likelihood_graph(self):
        self.entities1 = torch.LongTensor().to(self.device)
        self.entities2 = torch.LongTensor().to(self.device)
        dim = len(self.kgs.valid_links) + len(self.kgs.test_entities1)
        dim1 = self.args.likelihood_slice
        self.likelihood_mat = torch.FloatTensor(dim1, dim).to(self.device)

    def eval_ref_sim_mat(self):
        refs1_embeddings = torch.nn.functional.normalize(self.ent_embeds(torch.LongTensor(self.ref_ent1).to(self.device)), p=2, dim=1)
        refs2_embeddings = torch.nn.functional.normalize(self.ent_embeds(torch.LongTensor(self.ref_ent2).to(self.device)), p=2, dim=1)
        return torch.matmul(refs1_embeddings, refs2_embeddings.t())

    def launch_training_k_epo(self, iter, iter_nums, triple_steps, steps_tasks, training_batch_queue, neighbors1,
                              neighbors2):
        for i in range(1, iter_nums + 1):
            epoch = (iter - 1) * iter_nums + i
            self.launch_triple_training_1epo(epoch, triple_steps, steps_tasks, training_batch_queue, neighbors1,
                                             neighbors2)

    def train_alignment(self, kg1: KG, kg2: KG, entities1, entities2, training_epochs):
        if entities1 is None or len(entities1) == 0:
            return
        newly_tris1, newly_tris2 = generate_supervised_triples(kg1.rt_dict, kg1.hr_dict, kg2.rt_dict, kg2.hr_dict,
                                                               entities1, entities2)
        steps = math.ceil(((len(newly_tris1) + len(newly_tris2)) / self.args.batch_size))
        if steps == 0:
            steps = 1
        optimizer = optim.Adagrad(self.parameters(), lr=self.args.learning_rate)
        for i in range(training_epochs):
            t1 = time.time()
            alignment_loss = 0
            for step in range(steps):
                newly_batch1, newly_batch2 = generate_pos_batch(newly_tris1, newly_tris2, step, self.args.batch_size)
                newly_batch1.extend(newly_batch2)
                self.train()
                optimizer.zero_grad()
                phs, prs, pts = self.forward(
                    torch.LongTensor([tr[0] for tr in newly_batch1]).to(self.device),
                    torch.LongTensor([tr[1] for tr in newly_batch1]).to(self.device),
                    torch.LongTensor([tr[2] for tr in newly_batch1]).to(self.device)
                )
                loss = - torch.sum(torch.log(torch.sigmoid(-torch.sum(torch.pow(phs + prs - pts, 2), 1))))
                loss.backward()
                optimizer.step()
                alignment_loss += loss.item()
            alignment_loss /= (len(newly_tris1) + len(newly_tris2))
            print("alignment_loss = {:.3f}, time = {:.3f} s".format(alignment_loss, time.time() - t1))

    def likelihood(self, labeled_alignment):
        t = time.time()
        likelihood_mat = calculate_likelihood_mat(self.ref_ent1, self.ref_ent2, labeled_alignment)
        optimizer = optim.Adagrad(self.parameters(), lr=self.args.learning_rate)
        likelihood_loss = 0.0
        steps = len(self.ref_ent1) // self.args.likelihood_slice
        ref_ent1_array = np.array(self.ref_ent1)
        ll = list(range(len(self.ref_ent1)))
        for i in range(steps):
            idx = random.sample(ll, self.args.likelihood_slice)
            self.train()
            optimizer.zero_grad()
            ent1_embed = self.ent_embeds(torch.LongTensor(ref_ent1_array[idx]).to(self.device))
            ent2_embed = self.ent_embeds(torch.LongTensor(self.ref_ent2).to(self.device))
            mat = torch.log(torch.sigmoid(torch.matmul(ent1_embed, ent2_embed.t())))
            loss = -torch.sum(mat * torch.FloatTensor(likelihood_mat[idx, :]).to(self.device))
            loss.backward()
            optimizer.step()
            likelihood_loss += loss.item()
        print("likelihood_loss = {:.3f}, time = {:.3f} s".format(likelihood_loss, time.time() - t))

    def run(self):
        t = time.time()
        triples_num = self.kgs.kg1.relation_triples_num + self.kgs.kg2.relation_triples_num
        triple_steps = int(math.ceil(triples_num / self.args.batch_size))
        steps_tasks = task_divide(list(range(triple_steps)), self.args.batch_threads_num)
        manager = mp.Manager()
        training_batch_queue = manager.Queue()
        neighbors1, neighbors2 = None, None
        labeled_align = set()
        sub_num = self.args.sub_epoch
        iter_nums = self.args.max_epoch // sub_num
        for i in range(1, iter_nums + 1):
            print("\niteration", i)
            self.launch_training_k_epo(i, sub_num, triple_steps, steps_tasks, training_batch_queue, neighbors1, neighbors2)
            if i * sub_num >= self.args.start_valid:
                flag = self.valid(self.args.stop_metric)
                self.flag1, self.flag2, self.early_stop = early_stop(self.flag1, self.flag2, flag)
                if self.early_stop or i == iter_nums:
                    break
            labeled_align, entities1, entities2 = bootstrapping(self.eval_ref_sim_mat(),
                                                                self.ref_ent1, self.ref_ent2, labeled_align,
                                                                self.args.sim_th, self.args.k)
            self.train_alignment(self.kgs.kg1, self.kgs.kg2, entities1, entities2, 1)
            if i * sub_num >= self.args.start_valid:
                self.valid(self.args.stop_metric)
            t1 = time.time()
            assert 0.0 < self.args.truncated_epsilon < 1.0
            neighbors_num1 = int((1 - self.args.truncated_epsilon) * self.kgs.kg1.entities_num)
            neighbors_num2 = int((1 - self.args.truncated_epsilon) * self.kgs.kg2.entities_num)
            if neighbors1 is not None:
                del neighbors1, neighbors2
            gc.collect()
            neighbors1 = bat.generate_neighbours(self.eval_kg1_useful_ent_embeddings(),
                                                 self.kgs.useful_entities_list1,
                                                 neighbors_num1, self.args.batch_threads_num)
            neighbors2 = bat.generate_neighbours(self.eval_kg2_useful_ent_embeddings(),
                                                 self.kgs.useful_entities_list2,
                                                 neighbors_num2, self.args.batch_threads_num)
            ent_num = len(self.kgs.kg1.entities_list) + len(self.kgs.kg2.entities_list)
            print("generating neighbors of {} entities costs {:.3f} s.".format(ent_num, time.time() - t1))
        print("Training ends. Total time = {:.3f} s.".format(time.time() - t))
