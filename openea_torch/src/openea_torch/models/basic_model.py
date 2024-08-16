import math
import multiprocessing as mp
import random
import time
import gc
import torch
import numpy as np
import os

import openea_torch.modules.load.read as rd
import openea_torch.modules.train.batch as bat
from openea_torch.modules.finding.evaluation import valid, test, early_stop
from openea_torch.modules.finding.similarity import sim
from openea_torch.modules.utils.util import generate_out_folder
from openea_torch.modules.utils.util import load_session
from openea_torch.modules.utils.util import task_divide
from openea_torch.modules.base.initializers import init_embeddings
from openea_torch.modules.base.losses import get_loss_func
from openea_torch.modules.base.optimizers import generate_optimizer
from openea_torch.modules.base.mapping import add_mapping_variables, add_mapping_module

from openea_torch.modules.finding.alignment import stable_alignment

class BasicModel:

    def set_kgs(self, kgs):
        self.kgs = kgs

    def set_args(self, args):
        self.args = args
        self.out_folder = generate_out_folder(self.args.output, self.args.training_data, self.args.dataset_division,
                                              self.__class__.__name__)

    def init(self):
        # need to be overwritten
        pass

    def __init__(self):
        self.out_folder = None
        self.args = None
        self.kgs = None

        self.seed_entities1 = None
        self.seed_entities2 = None
        self.neg_ts = None
        self.neg_rs = None
        self.neg_hs = None
        self.pos_ts = None
        self.pos_rs = None
        self.pos_hs = None

        self.rel_embeds = None
        self.ent_embeds = None
        self.mapping_mat = None

        self.triple_optimizer = None
        self.triple_loss = None
        self.mapping_optimizer = None
        self.mapping_loss = None

        self.flag1 = -1
        self.flag2 = -1
        self.early_stop = False

    def _define_variables(self):
        self.ent_embeds = init_embeddings([self.kgs.entities_num, self.args.dim], 'ent_embeds',
                                          self.args.init, self.args.ent_l2_norm)
        self.rel_embeds = init_embeddings([self.kgs.relations_num, self.args.dim], 'rel_embeds',
                                          self.args.init, self.args.rel_l2_norm)

    def _define_embed_graph(self):
        self.pos_hs = torch.tensor([], dtype=torch.int32)
        self.pos_rs = torch.tensor([], dtype=torch.int32)
        self.pos_ts = torch.tensor([], dtype=torch.int32)
        self.neg_hs = torch.tensor([], dtype=torch.int32)
        self.neg_rs = torch.tensor([], dtype=torch.int32)
        self.neg_ts = torch.tensor([], dtype=torch.int32)
        
        phs = self.ent_embeds[self.pos_hs]
        prs = self.rel_embeds[self.pos_rs]
        pts = self.ent_embeds[self.pos_ts]
        nhs = self.ent_embeds[self.neg_hs]
        nrs = self.rel_embeds[self.neg_rs]
        nts = self.ent_embeds[self.neg_ts]

        self.triple_loss = get_loss_func(phs, prs, pts, nhs, nrs, nts, self.args)
        self.triple_optimizer = generate_optimizer(self.triple_loss, self.args.learning_rate, opt=self.args.optimizer)

    def _define_mapping_variables(self):
        add_mapping_variables(self)

    def _define_mapping_graph(self):
        add_mapping_module(self)

    def _eval_valid_embeddings(self):
        if len(self.kgs.valid_links) > 0:
            embeds1 = self.ent_embeds[self.kgs.valid_entities1].detach().cpu().numpy()
            embeds2 = self.ent_embeds[self.kgs.valid_entities2 + self.kgs.test_entities2].detach().cpu().numpy()
        else:
            embeds1 = self.ent_embeds[self.kgs.test_entities1].detach().cpu().numpy()
            embeds2 = self.ent_embeds[self.kgs.test_entities2].detach().cpu().numpy()
        mapping = self.mapping_mat.detach().cpu().numpy() if self.mapping_mat is not None else None
        return embeds1, embeds2, mapping

    def _eval_test_embeddings(self):
        embeds1 = self.ent_embeds[self.kgs.test_entities1].detach().cpu().numpy()
        embeds2 = self.ent_embeds[self.kgs.test_entities2].detach().cpu().numpy()
        mapping = self.mapping_mat.detach().cpu().numpy() if self.mapping_mat is not None else None
        return embeds1, embeds2, mapping

    def valid(self, stop_metric):
        embeds1, embeds2, mapping = self._eval_valid_embeddings()
        hits1_12, mrr_12 = valid(embeds1, embeds2, mapping, self.args.top_k,
                                 self.args.test_threads_num, metric=self.args.eval_metric,
                                 normalize=self.args.eval_norm, csls_k=0, accurate=False)
        return hits1_12 if stop_metric == 'hits1' else mrr_12

    def test(self, save=True):
        embeds1, embeds2, mapping = self._eval_test_embeddings()
        rest_12, _, _ = test(embeds1, embeds2, mapping, self.args.top_k, self.args.test_threads_num,
                             metric=self.args.eval_metric, normalize=self.args.eval_norm, csls_k=0, accurate=True)
        test(embeds1, embeds2, mapping, self.args.top_k, self.args.test_threads_num,
             metric=self.args.eval_metric, normalize=self.args.eval_norm, csls_k=self.args.csls, accurate=True)
        if save:
            ent_ids_rest_12 = [(self.kgs.test_entities1[i], self.kgs.test_entities2[j]) for i, j in rest_12]
            rd.save_results(self.out_folder, ent_ids_rest_12)

    def retest(self):
        dir = self.out_folder.split("/")
        new_dir = ""
        for i in range(len(dir) - 2):
            new_dir += (dir[i] + "/")
        exist_file = os.listdir(new_dir)
        new_dir = new_dir + exist_file[0] + "/"
        embeds = np.load(new_dir + "ent_embeds.npy")
        embeds1 = embeds[self.kgs.test_entities1]
        embeds2 = embeds[self.kgs.test_entities2]
        mapping = None

        print(self.__class__.__name__, type(self.__class__.__name__))
        if self.__class__.__name__ == "GCN_Align":
            print(self.__class__.__name__, "loads attr embeds")
            attr_embeds = np.load(new_dir + "attr_embeds.npy")
            attr_embeds1 = attr_embeds[self.kgs.test_entities1]
            attr_embeds2 = attr_embeds[self.kgs.test_entities2]
            embeds1 = np.concatenate([embeds1 * self.args.beta, attr_embeds1 * (1.0 - self.args.beta)], axis=1)
            embeds2 = np.concatenate([embeds2 * self.args.beta, attr_embeds2 * (1.0 - self.args.beta)], axis=1)

        # if self.__class__.__name__ == "MTransE" or self.__class__.__name__ == "SEA" or self.__class__.__name__ == "KDCoE":
        if os.path.exists(new_dir + "mapping_mat.npy"):
            print(self.__class__.__name__, "loads mapping mat")
            mapping = np.load(new_dir + "mapping_mat.npy")

        print("conventional test:")
        test(embeds1, embeds2, mapping, self.args.top_k, self.args.test_threads_num,
             metric=self.args.eval_metric, normalize=self.args.eval_norm, csls_k=0, accurate=True)
        print("conventional reversed test:")
        if mapping is not None:
            embeds1 = np.matmul(embeds1, mapping)
            test(embeds2, embeds1, None, self.args.top_k, self.args.test_threads_num,
                 metric=self.args.eval_metric, normalize=self.args.eval_norm, csls_k=0, accurate=True)
        else:
            test(embeds2, embeds1, mapping, self.args.top_k, self.args.test_threads_num,
                 metric=self.args.eval_metric, normalize=self.args.eval_norm, csls_k=0, accurate=True)
        print("stable test:")
        stable_alignment(embeds1, embeds2, self.args.eval_metric, self.args.eval_norm, csls_k=0,
                         nums_threads=self.args.test_threads_num)
        print("stable test with csls:")
        stable_alignment(embeds1, embeds2, self.args.eval_metric, self.args.eval_norm, csls_k=self.args.csls,
                         nums_threads=self.args.test_threads_num)

    def save(self):
        ent_embeds = self.ent_embeds.weight.data.cpu().numpy()
        rel_embeds = self.rel_embeds.weight.data.cpu().numpy()
        mapping_mat = self.mapping_mat.data.cpu().numpy() if self.mapping_mat is not None else None
        rd.save_embeddings(self.out_folder, self.kgs, ent_embeds, rel_embeds, None, mapping_mat=mapping_mat)

    def eval_kg1_ent_embeddings(self):
        embeds = self.ent_embeds(torch.tensor(self.kgs.kg1.entities_list, device=self.device))
        return embeds.cpu().detach().numpy()

    def eval_kg2_ent_embeddings(self):
        embeds = self.ent_embeds(torch.tensor(self.kgs.kg2.entities_list, device=self.device))
        return embeds.cpu().detach().numpy()

    def eval_kg1_useful_ent_embeddings(self):
        embeds = self.ent_embeds(torch.tensor(self.kgs.useful_entities_list1, device=self.device))
        return embeds.cpu().detach().numpy()

    def eval_kg2_useful_ent_embeddings(self):
        embeds = self.ent_embeds(torch.tensor(self.kgs.useful_entities_list2, device=self.device))
        return embeds.cpu().detach().numpy()

    def launch_training_1epo(self, epoch, triple_steps, steps_tasks, training_batch_queue, neighbors1, neighbors2):
        self.launch_triple_training_1epo(epoch, triple_steps, steps_tasks, training_batch_queue, neighbors1, neighbors2)
        if self.args.alignment_module == 'mapping':
            self.launch_mapping_training_1epo(epoch, triple_steps)

    def launch_triple_training_1epo(self, epoch, triple_steps, steps_tasks, batch_queue, neighbors1, neighbors2):
        start = time.time()
        for steps_task in steps_tasks:
            mp.Process(target=bat.generate_relation_triple_batch_queue,
                    args=(self.kgs.kg1.relation_triples_list, self.kgs.kg2.relation_triples_list,
                          self.kgs.kg1.relation_triples_set, self.kgs.kg2.relation_triples_set,
                          self.kgs.kg1.entities_list, self.kgs.kg2.entities_list,
                          self.args.batch_size, steps_task,
                          batch_queue, neighbors1, neighbors2, self.args.neg_triple_num)).start()
        epoch_loss = 0
        trained_samples_num = 0
        for i in range(triple_steps):
            batch_pos, batch_neg = batch_queue.get()
            pos_hs = torch.tensor([x[0] for x in batch_pos], device=self.device)
            pos_rs = torch.tensor([x[1] for x in batch_pos], device=self.device)
            pos_ts = torch.tensor([x[2] for x in batch_pos], device=self.device)
            neg_hs = torch.tensor([x[0] for x in batch_neg], device=self.device)
            neg_rs = torch.tensor([x[1] for x in batch_neg], device=self.device)
            neg_ts = torch.tensor([x[2] for x in batch_neg], device=self.device)
            self.triple_optimizer.zero_grad()
            pos_scores = self.compute_scores(pos_hs, pos_rs, pos_ts)
            neg_scores = self.compute_scores(neg_hs, neg_rs, neg_ts)
            batch_loss = self.triple_loss(pos_scores, neg_scores, torch.tensor([-1], device=self.device))
            batch_loss.backward()
            self.triple_optimizer.step()
            trained_samples_num += len(batch_pos)
            epoch_loss += batch_loss.item()
        epoch_loss /= trained_samples_num
        random.shuffle(self.kgs.kg1.relation_triples_list)
        random.shuffle(self.kgs.kg2.relation_triples_list)
        print('epoch {}, avg. triple loss: {:.4f}, cost time: {:.4f}s'.format(epoch, epoch_loss, time.time() - start))

    def launch_mapping_training_1epo(self, epoch, triple_steps):
        start = time.time()
        epoch_loss = 0
        trained_samples_num = 0
        for i in range(triple_steps):
            links_batch = random.sample(self.kgs.train_links, len(self.kgs.train_links) // triple_steps)
            seed_entities1 = torch.tensor([x[0] for x in links_batch], device=self.device)
            seed_entities2 = torch.tensor([x[1] for x in links_batch], device=self.device)
            self.mapping_optimizer.zero_grad()
            embeds1 = self.ent_embeds(seed_entities1)
            embeds2 = self.ent_embeds(seed_entities2)
            if self.mapping_mat is not None:
                embeds1 = torch.matmul(embeds1, self.mapping_mat)
            batch_loss = self.mapping_loss(embeds1, embeds2)
            batch_loss.backward()
            self.mapping_optimizer.step()
            epoch_loss += batch_loss.item()
            trained_samples_num += len(links_batch)
        epoch_loss /= trained_samples_num
        print('epoch {}, avg. mapping loss: {:.4f}, cost time: {:.4f}s'.format(epoch, epoch_loss, time.time() - start))

    def run(self):
        t = time.time()
        triples_num = self.kgs.kg1.relation_triples_num + self.kgs.kg2.relation_triples_num
        triple_steps = int(math.ceil(triples_num / self.args.batch_size))
        steps_tasks = task_divide(list(range(triple_steps)), self.args.batch_threads_num)
        manager = mp.Manager()
        training_batch_queue = manager.Queue()
        neighbors1, neighbors2 = None, None
        for i in range(1, self.args.max_epoch + 1):
            self.launch_training_1epo(i, triple_steps, steps_tasks, training_batch_queue, neighbors1, neighbors2)
            if i >= self.args.start_valid and i % self.args.eval_freq == 0:
                flag = self.valid(self.args.stop_metric)
                self.flag1, self.flag2, self.early_stop = early_stop(self.flag1, self.flag2, flag)
                if self.early_stop or i == self.args.max_epoch:
                    break
            if self.args.neg_sampling == 'truncated' and i % self.args.truncated_freq == 0:
                t1 = time.time()
                assert 0.0 < self.args.truncated_epsilon < 1.0
                neighbors_num1 = int((1 - self.args.truncated_epsilon) * self.kgs.kg1.entities_num)
                neighbors_num2 = int((1 - self.args.truncated_epsilon) * self.kgs.kg2.entities_num)
                if neighbors1 is not None:
                    del neighbors1, neighbors2
                gc.collect()
                neighbors1 = bat.generate_neighbours_single_thread(self.eval_kg1_useful_ent_embeddings(),
                                                                   self.kgs.useful_entities_list1,
                                                                   neighbors_num1, self.args.test_threads_num)
                neighbors2 = bat.generate_neighbours_single_thread(self.eval_kg2_useful_ent_embeddings(),
                                                                   self.kgs.useful_entities_list2,
                                                                   neighbors_num2, self.args.test_threads_num)
                ent_num = len(self.kgs.kg1.entities_list) + len(self.kgs.kg2.entities_list)
                print("\ngenerating neighbors of {} entities costs {:.3f} s.".format(ent_num, time.time() - t1))
                gc.collect()
        print("Training ends. Total time = {:.3f} s.".format(time.time() - t))

    def predict(self, top_k=1, min_sim_value=None, output_file_name=None):
        embeds1 = self.ent_embeds(torch.tensor(self.kgs.kg1.entities_list, device=self.device)).cpu().detach().numpy()
        embeds2 = self.ent_embeds(torch.tensor(self.kgs.kg2.entities_list, device=self.device)).cpu().detach().numpy()

        if self.mapping_mat is not None:
            embeds1 = np.matmul(embeds1, self.mapping_mat.data.cpu().numpy())

        sim_mat = sim(embeds1, embeds2, metric=self.args.eval_metric, normalize=self.args.eval_norm, csls_k=0)

        matched_entities_indexes = set()
        if top_k:
            assert top_k > 0
            for i in range(sim_mat.shape[0]):
                for rank_index in np.argpartition(-sim_mat[i, :], top_k)[:top_k]:
                    matched_entities_indexes.add((i, rank_index))

            for i in range(sim_mat.shape[1]):
                for rank_index in np.argpartition(-sim_mat[:, i], top_k)[:top_k]:
                    matched_entities_indexes.add((rank_index, i))

            if min_sim_value:
                matched_entities_indexes.intersection(map(tuple, np.argwhere(sim_mat > min_sim_value)))
        elif min_sim_value:
            matched_entities_indexes = set(map(tuple, np.argwhere(sim_mat > min_sim_value)))
        else:
            raise ValueError("Either top_k or min_sim_value should have a value")

        kg1_id_to_uri = {v: k for k, v in self.kgs.kg1.entities_id_dict.items()}
        kg2_id_to_uri = {v: k for k, v in self.kgs.kg2.entities_id_dict.items()}

        topk_neighbors_w_sim = [(kg1_id_to_uri[self.kgs.kg1.entities_list[i]],
                                 kg2_id_to_uri[self.kgs.kg2.entities_list[j]],
                                 sim_mat[i, j]) for i, j in matched_entities_indexes]

        if output_file_name is not None:
            if not os.path.exists(self.out_folder):
                os.makedirs(self.out_folder)
            with open(self.out_folder + output_file_name, 'w', encoding='utf8') as file:
                for entity1, entity2, confidence in topk_neighbors_w_sim:
                    file.write(str(entity1) + "\t" + str(entity2) + "\t" + str(confidence) + "\n")
            print(self.out_folder + output_file_name, "saved")
        return topk_neighbors_w_sim

    def compute_scores(self, hs, rs, ts):
        h_embeds = self.ent_embeds(hs)
        r_embeds = self.rel_embeds(rs)
        t_embeds = self.ent_embeds(ts)
        return torch.norm(h_embeds + r_embeds - t_embeds, p=1, dim=1)

def predict_entities(self, entities_file_path, output_file_name=None):
    """
    Compute the confidence of given entities if they match or not.
    Parameters
    ----------
    entities_file_path : str
        A path pointing to a file formatted as (entity1, entity2) with tab separated (tsv-file).
        If given, the similarity of the entities is retrieved and returned (or also written to file if output_file_name is given).
        The parameters top_k and min_sim_value do not play a role, if this parameter is set.
    output_file_name : str, optional
        The name of the output file. It is formatted as tsv file with entity1, entity2, confidence.
    Returns
    -------
    topk_neighbors_w_sim : A list of tuples of form (entity1, entity2, confidence)
    """

    kg1_entities = []
    kg2_entities = []
    with open(entities_file_path, 'r', encoding='utf-8') as input_file:
        for line in input_file:
            entities = line.strip('\n').split('\t')
            kg1_entities.append(self.kgs.kg1.entities_id_dict[entities[0]])
            kg2_entities.append(self.kgs.kg2.entities_id_dict[entities[1]])
    kg1_distinct_entities = list(set(kg1_entities))  # make distinct
    kg2_distinct_entities = list(set(kg2_entities))

    kg1_mapping = {entity_id: index for index, entity_id in enumerate(kg1_distinct_entities)}
    kg2_mapping = {entity_id: index for index, entity_id in enumerate(kg2_distinct_entities)}

    embeds1 = self.ent_embeds[kg1_distinct_entities]
    embeds2 = self.ent_embeds[kg2_distinct_entities]

    if self.mapping_mat is not None:
        embeds1 = torch.matmul(embeds1, self.mapping_mat)

    sim_mat = self.sim(embeds1, embeds2, metric=self.args.eval_metric, normalize=self.args.eval_norm, csls_k=0)

    # map back with entities_id_dict to be sure that the right uri is chosen
    kg1_id_to_uri = {v: k for k, v in self.kgs.kg1.entities_id_dict.items()}
    kg2_id_to_uri = {v: k for k, v in self.kgs.kg2.entities_id_dict.items()}

    topk_neighbors_w_sim = []
    for entity1_id, entity2_id in zip(kg1_entities, kg2_entities):
        topk_neighbors_w_sim.append((
            kg1_id_to_uri[entity1_id],
            kg2_id_to_uri[entity2_id],
            sim_mat[kg1_mapping[entity1_id], kg2_mapping[entity2_id]].item()
        ))

    if output_file_name is not None:
        # create dir if not existent
        if not os.path.exists(self.out_folder):
            os.makedirs(self.out_folder)
        with open(os.path.join(self.out_folder, output_file_name), 'w', encoding='utf8') as file:
            for entity1, entity2, confidence in topk_neighbors_w_sim:
                file.write(f"{entity1}\t{entity2}\t{confidence:.6f}\n")
        print(os.path.join(self.out_folder, output_file_name), "saved")

    return topk_neighbors_w_sim