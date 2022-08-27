# -*- coding: utf-8 -*-
from tqdm import tqdm
import os
import random
import torch
import torch.nn as nn
from torch.nn.functional import softmax

from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup

import pdb
import argparse, logging
import glob

from model import MRSModel
from dataset import static_loader

def main():
    """save & log path"""
    model_type = args.model_type
    save_path = os.path.join(model_type)
    print("###Save Path### ", save_path)
    
    os.makedirs(save_path, exist_ok=True)
    log_path = os.path.join(save_path, 'test.log')
    fileHandler = logging.FileHandler(log_path)
    
    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)    
    logger.setLevel(level=logging.DEBUG)    
    
    """Model Loading"""    
    model = MRSModel(model_type).cuda()
    modelfile = os.path.join('model.bin') # save_path
    model.load_state_dict(torch.load(modelfile))    
    model.eval()
    
    """dataset Loading"""
    test_path = '/data/project/rw/rung/source/dataset/MRS/ubuntu_data/test.txt'
        
    test_dataset = static_loader(test_path, model_type)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=test_dataset.collate_fn)    
    
    test_r1, test_r3, test_r5 = _CalR(model, test_loader)
    logger.info('test r@1: {}, r@3: {}, r@5: {}'.format(test_r1, test_r3, test_r5))

def rek(true_probs, batch_labels, k):
    sort_probs=sorted(true_probs, reverse=True)    
    r_inds = [true_probs.index(x) for x in sort_probs[:k+1]]
    r_score = sum([batch_labels[x].item() for x in r_inds])
    return r_score
    
        
def _CalR(model, data_loader):
    candidate_num = 10
    model.eval()
        
    true_index = 1
    true_probs, batch_labels = [], []
    count = 0
    r1_scores, r3_scores, r5_scores = [], [], []
    for i_batch, (input_token, label) in enumerate(tqdm(data_loader, desc='eval_iteration')):
        count += 1
        if count < candidate_num:
            input_token = input_token.cuda()
            logits = model(input_token) # (1, C)
            prob = softmax(logits, 1) # (1, 2)
            true_prob = prob.squeeze(0)[true_index].item()
            true_probs.append(true_prob)
            batch_labels.append(label)
        else: # 10
            r1_score = rek(true_probs, batch_labels, 1)
            r3_score = rek(true_probs, batch_labels, 3)
            r5_score = rek(true_probs, batch_labels, 5)            
            r1_scores.append(r1_score)
            r3_scores.append(r3_score)
            r5_scores.append(r5_score)
            
            true_probs, batch_labels = [], []
            count = 0
            input_token = input_token.cuda()
            logits = model(input_token) # (1, C)
            prob = softmax(logits, 1) # (1, 2)
            true_prob = prob.squeeze(0)[true_index].item()
            true_probs.append(true_prob)
            batch_labels.append(label)
        
        if i_batch == len(data_loader)-1:
            r1_score = rek(true_probs, batch_labels, 1)
            r3_score = rek(true_probs, batch_labels, 3)
            r5_score = rek(true_probs, batch_labels, 5)            
            r1_scores.append(r1_score)
            r3_scores.append(r3_score)
            r5_scores.append(r5_score)            
        
    r1 = sum(r1_scores)/len(r1_scores)
    r3 = sum(r3_scores)/len(r3_scores)
    r5 = sum(r5_scores)/len(r5_scores)
    return r1, r3, r5
    
def clsLoss(pred_outs, labels):
    """
        pred_outs: [batch, clsNum]
        labels: [clsNum]
    """
    loss = nn.CrossEntropyLoss()
    loss_val = loss(pred_outs, labels)
    return loss_val

def _SaveModel(model, path):
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(path, 'model.bin'))    

if __name__ == '__main__':
    torch.cuda.empty_cache()
    
    """Parameters"""
    parser  = argparse.ArgumentParser(description = "Response selection" )    
    parser.add_argument("--model_type", help = "pretrained model", default = 'roberta-large')
            
    args = parser.parse_args()
    
    logger = logging.getLogger(__name__)
    streamHandler = logging.StreamHandler()
    
    main()