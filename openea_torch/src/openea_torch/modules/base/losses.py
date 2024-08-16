import torch
import re
import random as rd
import os
import requests

def read_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    return [line.strip().split('\t') for line in lines]

def chunk_list(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

def extract_numbers(text):
    numbers = re.findall(r'\d+', text)
    numbers = [int(num) if num.isdigit() else float(num) for num in numbers]
    return numbers

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
    pos_distance = phs + prs - pts
    neg_distance = nhs + nrs - nts
    if loss_norm == 'L1':
        pos_score = torch.sum(torch.abs(pos_distance), dim=1)
        neg_score = torch.sum(torch.abs(neg_distance), dim=1)
    else:
        pos_score = torch.sum(pos_distance ** 2, dim=1)
        neg_score = torch.sum(neg_distance ** 2, dim=1)
    loss = torch.sum(torch.relu(margin + pos_score - neg_score))
    return loss

def positive_loss(phs, prs, pts, loss_norm):
    pos_distance = phs + prs - pts
    if loss_norm == 'L1':
        pos_score = torch.sum(torch.abs(pos_distance), dim=1)
    else:
        pos_score = torch.sum(pos_distance ** 2, dim=1)
    loss = torch.sum(pos_score)
    return loss

def limited_loss(phs, prs, pts, nhs, nrs, nts, pos_margin, neg_margin, loss_norm, balance=1.0, alpha=0.6):
    pos_distance = phs + prs - pts
    neg_distance = nhs + nrs - nts
    if loss_norm == 'L1':
        pos_score = torch.sum(torch.abs(pos_distance), dim=1)
        neg_score = torch.sum(torch.abs(neg_distance), dim=1)
    else:
        pos_score = torch.sum(pos_distance ** 2, dim=1)
        neg_score = torch.sum(neg_distance ** 2, dim=1)
    pos_loss = torch.sum(torch.relu(pos_score - pos_margin))
    neg_loss = torch.sum(torch.relu(neg_margin - neg_score))
    loss = pos_loss + balance * neg_loss
    return loss

def logistic_loss(phs, prs, pts, nhs, nrs, nts, loss_norm):
    pos_distance = phs + prs - pts
    neg_distance = nhs + nrs - nts
    if loss_norm == 'L1':
        pos_score = torch.sum(torch.abs(pos_distance), dim=1)
        neg_score = torch.sum(torch.abs(neg_distance), dim=1)
    else:
        pos_score = torch.sum(pos_distance ** 2, dim=1)
        neg_score = torch.sum(neg_distance ** 2, dim=1)
    pos_loss = torch.sum(torch.log(1 + torch.exp(pos_score)))
    neg_loss = torch.sum(torch.log(1 + torch.exp(-neg_score)))
    loss = pos_loss + neg_loss
    return loss

def mapping_loss(tes1, tes2, mapping, eye):
    mapped_tes2 = torch.matmul(tes1, mapping)
    map_loss = torch.sum((tes2 - mapped_tes2) ** 2)
    orthogonal_loss = torch.sum((torch.matmul(mapping, mapping.T) - eye) ** 2)
    return map_loss + orthogonal_loss