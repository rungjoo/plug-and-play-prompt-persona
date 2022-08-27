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
from persona_dataset import peronsa_loader, peronsa_test_loader

from collections import OrderedDict

def main():
    """save & log path"""
    model_type = args.model_type
    data_type = args.data_type
    save_path = os.path.join(model_type)
    print("###Save Path### ", save_path)
    
    os.makedirs(save_path, exist_ok=True)
    log_path = os.path.join(save_path, 'train.log')
    fileHandler = logging.FileHandler(log_path)
    
    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)    
    logger.setLevel(level=logging.DEBUG)    
    
    """Model Loading"""    
    model = MRSModel(model_type).cuda()
    model.train()
        
    """dataset Loading"""
    train_path = "../datset/personachat/train_both_" + data_type + ".json"
    dev_path = "../datset/personachat/valid_both_" + data_type + ".json"
    test_path = "../datset/personachat/test_both_" + data_type + ".json"
    
    batch_size = args.batch
    logger.info("###################")
    print('batch size: ', batch_size)
    train_dataset = peronsa_loader(train_path, model_type)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=train_dataset.collate_fn)
    
    """Training Parameter Setting"""    
    training_epochs = args.epoch
    print('Training Epochs: ', str(training_epochs))
    max_grad_norm = args.norm
    lr = args.lr
    num_training_steps = len(train_dataset)*training_epochs
    num_warmup_steps = len(train_dataset)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr) # , eps=1e-06, weight_decay=0.01    
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    
    """Training"""
    best_dev_p1 = 0
    for epoch in range(training_epochs):
        model.train()
        for i_batch, (batch_input_token, batch_labels, batch_personas, batch_response) in enumerate(tqdm(train_data_loader, desc='train_iteration')):
            """
                batch_input_token: (B, L)
                batch_labels: (B)
            """
            if i_batch % 100000 == 0:
                logger.info("i_batch: {}".format(i_batch))
            
            optimizer.zero_grad()
            batch_input_token, batch_labels = batch_input_token.cuda(), batch_labels.cuda()
            pred_outs = model(batch_input_token) # (B, C)
            loss_val = clsLoss(pred_outs, batch_labels)
            
            loss_val.backward() # noraml
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  # Gradient clipping is not in AdamW anymore (so you can use amp without issue)
            optimizer.step()
            scheduler.step()
        
        model.eval()        
        logger.info('Epoch: {}'.format(epoch))
        dev_p1 = CalPER(model, dev_path, model_type)
        logger.info('적합성 dev p@1: {}'.format(dev_p1))

        if dev_p1 > best_dev_p1:
            best_dev_p1 = dev_p1
            test_p1 = CalPER(model, test_path, model_type)
            logger.info('적합성 test p@1: {}'.format(test_p1))
            _SaveModel(model, save_path)
            
    logger.info('Final test 적합성 p@1: {}'.format(test_p1))
    
def CalPER(model, data_path, model_type):
    """dataset"""
    dataset = peronsa_test_loader(data_path, model_type)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=dataset.collate_fn)    
    
    true_index = 1
    pre1 = []
    for i_batch, (batch_input_token, batch_labels, batch_personas, batch_response) in enumerate(tqdm(data_loader, desc='eval_iteration')):
        true_probs = []
        batch_labels = batch_labels.tolist()        
            
        for input_token, input_label in zip(batch_input_token, batch_labels):                        
            input_token = input_token.unsqueeze(0).cuda()
            logits = model(input_token) # (1, C)
            
            prob = softmax(logits, 1) # (1, 2)
            true_prob = prob.squeeze(0)[true_index].item()
            true_probs.append(true_prob)        
        
        final_scores = true_probs
        max_ind = final_scores.index(max(final_scores))
        
        if batch_labels[max_ind] > 0:
            pre1.append(1)
        else:
            pre1.append(0)
    test_p1 = round(sum(pre1)/len(pre1)*100, 2)
    
    return test_p1
    
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
    
    return

if __name__ == '__main__':
    torch.cuda.empty_cache()
    
    """Parameters"""
    parser  = argparse.ArgumentParser(description = "Response selection" )
    parser.add_argument("--epoch", type=int, help = 'training epohcs', default = 5) # 12 for iemocap
    parser.add_argument("--norm", type=int, help = "max_grad_norm", default = 10)
    parser.add_argument("--lr", type=float, help = "learning rate", default = 1e-6) # 1e-5
    parser.add_argument("--batch", type=int, help = "training batch size", default = 1) # base
    
    parser.add_argument("--model_type", help = "pretrained model", default = 'roberta-large')
    parser.add_argument("--data_type", help = "original or revised", default = 'original')
#     parser.add_argument("--negative_numbers", type=int, help = "how much?", default = 1)
#     parser.add_argument("--negative_context_numbers", type=int, help = "negative context numbers in whole candidates", default = 0)
            
    args = parser.parse_args()
    
    logger = logging.getLogger(__name__)
    streamHandler = logging.StreamHandler()
    
    main()