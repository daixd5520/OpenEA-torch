
# 2024.07.18备份：
#   加入调另一个窗口开启的大语言模型API服务但是实验室机器太满会OOM
import tensorflow as tf
import re
# import torch
import os
import requests
# from tqdm import tqdm
# from pathlib import Path
# from peft import AutoPeftModelForCausalLM, PeftModelForCausalLM
# from transformers import (
#     AutoModelForCausalLM,
#     AutoTokenizer,
#     PreTrainedModel,
#     PreTrainedTokenizer,
#     PreTrainedTokenizerFast
# )
# from typing import Union, Annotated




def read_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    return [line.strip().split('\t') for line in lines]

def chunk_list(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

def extract_numbers(text):
    # 使用正则表达式匹配文本中的所有数字
    numbers = re.findall(r'\d+', text)
    # 将匹配到的数字字符串转换为整数或浮点数
    numbers = [int(num) if num.isdigit() else float(num) for num in numbers]
    return numbers

def get_score_from_language_model(triples):
    """
    Call the local language model to get the score for given triples.
    :param triples: A list of triples to be scored.
    :return: A list of scores for the triples.
    """
    scores = []
    for triple in triples:
        prompt = f"以下是实体对齐过程中的三元组：{triple}，给出三元组评分。三元组评分介于0和100之间，评分越小，代表你判断实体对齐三元组效果越好。你应该仅输出一个浮点数代表你给出的分数，不要有任何其他解释。"
        response = requests.post('http://localhost:5000/generate', json={'prompt': prompt})
        response=response.json()['response']
        # score = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        #score=extract_numbers(response)
        score = float(response)
        print(score)
        scores.append(float(score))
        #score = model.chat(prompt) 
        #scores.append(float(score))  # 确保 score 是一个可用的浮点数
    return scores


def get_loss_func(phs, prs, pts, nhs, nrs, nts, args):
    triple_loss = None
    if args.loss == 'margin-based':
        triple_loss = margin_loss(phs, prs, pts, nhs, nrs, nts, args.margin, args.loss_norm)
    elif args.loss == 'logistic':
        triple_loss = logistic_loss(phs, prs, pts, nhs, nrs, nts, args.loss_norm)
    elif args.loss == 'limited':
        triple_loss = limited_loss(phs, prs, pts, nhs, nrs, nts, args.pos_margin, args.neg_margin, args.loss_norm)
    return triple_loss


def margin_loss(phs, prs, pts, nhs, nrs, nts, margin, loss_norm):
    with tf.name_scope('margin_loss_distance'):
        pos_distance = phs + prs - pts
        neg_distance = nhs + nrs - nts
    with tf.name_scope('margin_loss'):
        if loss_norm == 'L1':  # L1 normal
            pos_score = tf.reduce_sum(tf.abs(pos_distance), axis=1)
            neg_score = tf.reduce_sum(tf.abs(neg_distance), axis=1)
        else:  # L2 normal
            pos_score = tf.reduce_sum(tf.square(pos_distance), axis=1)
            neg_score = tf.reduce_sum(tf.square(neg_distance), axis=1)
        loss = tf.reduce_sum(tf.nn.relu(tf.constant(margin) + pos_score - neg_score), name='margin_loss')
    return loss


def positive_loss(phs, prs, pts, loss_norm):
    with tf.name_scope('positive_loss_distance'):
        pos_distance = phs + prs - pts
    with tf.name_scope('positive_loss_score'):
        if loss_norm == 'L1':  # L1 score
            pos_score = tf.reduce_sum(tf.abs(pos_distance), axis=1)
        else:  # L2 score
            pos_score = tf.reduce_sum(tf.square(pos_distance), axis=1)
        loss = tf.reduce_sum(pos_score, name='positive_loss')
    return loss


def limited_loss(phs, prs, pts, nhs, nrs, nts, pos_margin, neg_margin, loss_norm, balance=1.0,alpha=0.6):
    with tf.name_scope('limited_loss_distance'):
        pos_distance = phs + prs - pts# positive_head_relation_tail
        neg_distance = nhs + nrs - nts# negative_head_relation_tail
    with tf.name_scope('limited_loss_score'):
        if loss_norm == 'L1':  # L1 score
            pos_score = tf.reduce_sum(tf.abs(pos_distance), axis=1)
            neg_score = tf.reduce_sum(tf.abs(neg_distance), axis=1)
        else:  # L2 score
            pos_score = tf.reduce_sum(tf.square(pos_distance), axis=1)
            neg_score = tf.reduce_sum(tf.square(neg_distance), axis=1)
        
         # Convert embeddings to numpy arrays to be used with the language model
        with tf.Session() as sess:
            phs_np = sess.run(phs)
            prs_np = sess.run(prs)
            pts_np = sess.run(pts)
            nhs_np = sess.run(nhs)
            nrs_np = sess.run(nrs)
            nts_np = sess.run(nts)
        
        # Integrate language model scores
        triples = list(zip(phs_np, prs_np, pts_np, nhs_np, nrs_np, nts_np))
        lm_scores = get_score_from_language_model(triples)
        lm_pos_score = tf.constant(lm_scores[:len(phs_np)], dtype=tf.float32)
        lm_neg_score = tf.constant(lm_scores[len(phs_np):], dtype=tf.float32)
        
        pos_score = alpha*pos_score+(1-alpha)*lm_pos_score
        neg_score = alpha*neg_score+(1-alpha)*lm_neg_score
        
        
        pos_loss = tf.reduce_sum(tf.nn.relu(pos_score - tf.constant(pos_margin)))
        neg_loss = tf.reduce_sum(tf.nn.relu(tf.constant(neg_margin) - neg_score))
        loss = tf.add(pos_loss, balance * neg_loss, name='limited_loss')
    return loss


def logistic_loss(phs, prs, pts, nhs, nrs, nts, loss_norm):
    with tf.name_scope('logistic_loss_distance'):
        pos_distance = phs + prs - pts
        neg_distance = nhs + nrs - nts
    with tf.name_scope('logistic_loss_score'):
        if loss_norm == 'L1':  # L1 score
            pos_score = tf.reduce_sum(tf.abs(pos_distance), axis=1)
            neg_score = tf.reduce_sum(tf.abs(neg_distance), axis=1)
        else:  # L2 score
            pos_score = tf.reduce_sum(tf.square(pos_distance), axis=1)
            neg_score = tf.reduce_sum(tf.square(neg_distance), axis=1)
        pos_loss = tf.reduce_sum(tf.log(1 + tf.exp(pos_score)))
        neg_loss = tf.reduce_sum(tf.log(1 + tf.exp(-neg_score)))
        loss = tf.add(pos_loss, neg_loss, name='logistic_loss')
    return loss


def mapping_loss(tes1, tes2, mapping, eye):
    mapped_tes2 = tf.matmul(tes1, mapping)
    map_loss = tf.reduce_sum(tf.reduce_sum(tf.pow(tes2 - mapped_tes2, 2), 1))
    orthogonal_loss = tf.reduce_sum(tf.reduce_sum(tf.pow(tf.matmul(mapping, mapping, transpose_b=True) - eye, 2), 1))
    return map_loss + orthogonal_loss
