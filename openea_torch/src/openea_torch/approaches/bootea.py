import gc
import math
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import time
from ..modules.finding.evaluation import early_stop
from ..modules.train import batch as bat
from ..approaches.aligne import AlignE
from ..modules.utils.util import task_divide
from ..modules.bootstrapping.alignment_finder import find_potential_alignment_mwgm, check_new_alignment
from ..modules.load.kg import KG

def bootstrapping(sim_mat, unaligned_entities1, unaligned_entities2, labeled_alignment, sim_th, k):
    curr_labeled_alignment = find_potential_alignment_mwgm(sim_mat, sim_th, k)
    if curr_labeled_alignment is not None:
        labeled_alignment = update_labeled_alignment_x(labeled_alignment, curr_labeled_alignment, sim_mat)
        labeled_alignment = update_labeled_alignment_y(labeled_alignment, sim_mat)
        del curr_labeled_alignment
    if labeled_alignment is not None:
        newly_aligned_entities1 = [unaligned_entities1[pair[0]] for pair in labeled_alignment]
        newly_aligned_entities2 = [unaligned_entities2[pair[1]] for pair in labeled_alignment]
    else:
        newly_aligned_entities1, newly_aligned_entities2 = None, None
    del sim_mat
    gc.collect()
    return labeled_alignment, newly_aligned_entities1, newly_aligned_entities2


def update_labeled_alignment_x(pre_labeled_alignment, curr_labeled_alignment, sim_mat):
    labeled_alignment_dict = dict(pre_labeled_alignment)
    n1, n2 = 0, 0
    for i, j in curr_labeled_alignment:
        if labeled_alignment_dict.get(i, -1) == i and j != i:
            n2 += 1
        if i in labeled_alignment_dict.keys():
            pre_j = labeled_alignment_dict.get(i)
            pre_sim = sim_mat[i, pre_j]
            new_sim = sim_mat[i, j]
            if new_sim >= pre_sim:
                if pre_j == i and j != i:
                    n1 += 1
                labeled_alignment_dict[i] = j
        else:
            labeled_alignment_dict[i] = j
    print("update wrongly: ", n1, "greedy update wrongly: ", n2)
    pre_labeled_alignment = set(zip(labeled_alignment_dict.keys(), labeled_alignment_dict.values()))
    check_new_alignment(pre_labeled_alignment, context="after editing (<-)")
    return pre_labeled_alignment


def update_labeled_alignment_y(labeled_alignment, sim_mat):
    labeled_alignment_dict = dict()
    updated_alignment = set()
    for i, j in labeled_alignment:
        i_set = labeled_alignment_dict.get(j, set())
        i_set.add(i)
        labeled_alignment_dict[j] = i_set
    for j, i_set in labeled_alignment_dict.items():
        if len(i_set) == 1:
            for i in i_set:
                updated_alignment.add((i, j))
        else:
            max_i = -1
            max_sim = -10
            for i in i_set:
                if sim_mat[i, j] > max_sim:
                    max_sim = sim_mat[i, j]
                    max_i = i
            updated_alignment.add((max_i, j))
    check_new_alignment(updated_alignment, context="after editing (->)")
    return updated_alignment


def calculate_likelihood_mat(ref_ent1, ref_ent2, labeled_alignment):
    def set2dic(alignment):
        if alignment is None:
            return None
        dic = dict()
        for i, j in alignment:
            dic[i] = j
        assert len(dic) == len(alignment)
        return dic

    t = time.time()
    ref_mat = np.zeros((len(ref_ent1), len(ref_ent2)), dtype=np.float32)
    if labeled_alignment is not None:
        alignment_dic = set2dic(labeled_alignment)
        n = 1 / len(ref_ent1)
        for ii in range(len(ref_ent1)):
            if ii in alignment_dic.keys():
                ref_mat[ii, alignment_dic.get(ii)] = 1
            else:
                for jj in range(len(ref_ent1)):
                    ref_mat[ii, jj] = n
    print("calculate likelihood matrix costs {:.2f} s".format(time.time() - t))
    return ref_mat


def generate_supervised_triples(rt_dict1, hr_dict1, rt_dict2, hr_dict2, ents1, ents2):
    assert len(ents1) == len(ents2)
    newly_triples1, newly_triples2 = list(), list()
    for i in range(len(ents1)):
        newly_triples1.extend(generate_newly_triples(ents1[i], ents2[i], rt_dict1, hr_dict1))
        newly_triples2.extend(generate_newly_triples(ents2[i], ents1[i], rt_dict2, hr_dict2))
    print("newly triples: {}, {}".format(len(newly_triples1), len(newly_triples2)))
    return newly_triples1, newly_triples2


def generate_newly_triples(ent1, ent2, rt_dict1, hr_dict1):
    newly_triples = list()
    for r, t in rt_dict1.get(ent1, set()):
        newly_triples.append((ent2, r, t))
    for h, r in hr_dict1.get(ent1, set()):
        newly_triples.append((h, r, ent2))
    return newly_triples


def generate_pos_batch(triples1, triples2, step, batch_size):
    num1 = int(len(triples1) / (len(triples1) + len(triples2)) * batch_size)
    num2 = batch_size - num1
    start1 = step * num1
    start2 = step * num2
    end1 = start1 + num1
    end2 = start2 + num2
    if end1 > len(triples1):
        end1 = len(triples1)
    if end2 > len(triples2):
        end2 = len(triples2)
    pos_triples1 = triples1[start1: end1]
    pos_triples2 = triples2[start2: end2]
    return pos_triples1, pos_triples2


def matmul(tensor1, tensor2, num, sigmoid):
    t = time.time()
    if num < 20000:
        sim_mat = torch.matmul(tensor1, tensor2.T)
        if sigmoid:
            res = torch.sigmoid(sim_mat)
        else:
            res = sim_mat
    else:
        res = torch.matmul(tensor1, tensor2.T)
    print("mat mul costs: {:.3f}".format(time.time() - t))
    return res


class BootEA(AlignE):

    def __init__(self):
        super().__init__()
        self.ref_ent1 = None
        self.ref_ent2 = None

    def init(self):
        self._define_variables()
        self._define_embed_graph()
        self._define_alignment_graph()
        self._define_likelihood_graph()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

    def _define_alignment_graph(self):
        self.new_h = None
        self.new_r = None
        self.new_t = None
        self.alignment_loss = None
        self.alignment_optimizer = None

    def _define_likelihood_graph(self):
        self.entities1 = None
        self.entities2 = None
        self.likelihood_mat = None
        self.likelihood_loss = None
        self.likelihood_optimizer = None

    def eval_ref_sim_mat(self):
        refs1_embeddings = F.normalize(self.ent_embeds[self.ref_ent1], p=2, dim=1).to(self.device)
        refs2_embeddings = F.normalize(self.ent_embeds[self.ref_ent2], p=2, dim=1).to(self.device)
        ref_sim_mat = matmul(refs1_embeddings, refs2_embeddings, len(self.ref_ent1), False).cpu().numpy()
        del refs1_embeddings, refs2_embeddings
        gc.collect()
        return ref_sim_mat

    def eval_sim_mat(self, entities1, entities2):
        ent1_embeddings = F.normalize(self.ent_embeds[entities1], p=2, dim=1).to(self.device)
        ent2_embeddings = F.normalize(self.ent_embeds[entities2], p=2, dim=1).to(self.device)
        sim_mat = matmul(ent1_embeddings, ent2_embeddings, len(entities1), False).cpu().numpy()
        del ent1_embeddings, ent2_embeddings
        gc.collect()
        return sim_mat

    def add_supervised_alignment(self, new_alignment):
        self.train_sup_pairs1 = new_alignment[:, 0]
        self.train_sup_pairs2 = new_alignment[:, 1]

    def add_triples(self, newly_triples1, newly_triples2):
        if len(newly_triples1) > 0:
            triples1 = np.vstack((self.kgs.kg1.local_relation_triples, newly_triples1))
            triples2 = np.vstack((self.kgs.kg2.local_relation_triples, newly_triples2))
            print("update triples for KG1 and KG2, respectively: {}, {} -> {}, {}".
                  format(len(self.kgs.kg1.local_relation_triples), len(self.kgs.kg2.local_relation_triples),
                         len(triples1), len(triples2)))
            self.kgs.kg1.local_relation_triples = triples1
            self.kgs.kg2.local_relation_triples = triples2

    def _define_variables(self):
        super()._define_variables()
        self.neg_margin = self.args.neg_margin
        self.truncated_epsilon = self.args.truncated_epsilon
        self.neg_triple_num = self.args.neg_triple_num

        # Create optimizer
        self.alignment_optimizer = optim.Adagrad(self.parameters(), lr=self.args.learning_rate)
        self.likelihood_optimizer = optim.Adagrad(self.parameters(), lr=self.args.learning_rate)

    def find_potential_alignment(self):
        ref_sim_mat = self.eval_ref_sim_mat()
        potential_alignment = find_potential_alignment_mwgm(ref_sim_mat, self.args.sim_th, self.args.k)
        del ref_sim_mat
        gc.collect()
        return potential_alignment

    def update_supervised_triples(self, aligned_entities1, aligned_entities2):
        if aligned_entities1 is None:
            return
        new_triples1, new_triples2 = generate_supervised_triples(self.rt_dict1, self.hr_dict1, self.rt_dict2, self.hr_dict2,
                                                                 aligned_entities1, aligned_entities2)
        if len(new_triples1) == 0:
            print("no newly triples")
            return
        self.add_triples(np.array(new_triples1), np.array(new_triples2))

    def optimize(self):
        if self.alignment_loss is not None:
            self.alignment_optimizer.zero_grad()
            self.alignment_loss.backward()
            self.alignment_optimizer.step()
        if self.likelihood_loss is not None:
            self.likelihood_optimizer.zero_grad()
            self.likelihood_loss.backward()
            self.likelihood_optimizer.step()

    def train(self):
        training_steps = self.args.max_epoch
        for step in range(training_steps):
            self.step = step
            start = time.time()

            pos_triples1, pos_triples2 = generate_pos_batch(self.pos_triples1, self.pos_triples2, step, self.args.batch_size)
            pos_triples1 = torch.tensor(pos_triples1).to(self.device)
            pos_triples2 = torch.tensor(pos_triples2).to(self.device)

            self.alignment_loss = self._calc_alignment_loss(pos_triples1, pos_triples2)
            self.optimize()

            likelihood_mat = calculate_likelihood_mat(self.ref_ent1, self.ref_ent2, self.labeled_alignment)
            self.likelihood_loss = self._calc_likelihood_loss(likelihood_mat)
            self.optimize()

            if early_stop(self):
                break

            print(f"Step {step}/{training_steps} finished in {time.time() - start:.2f}s")
