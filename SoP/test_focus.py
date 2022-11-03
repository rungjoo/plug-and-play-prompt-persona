import pdb
import argparse, logging
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn.functional import softmax
import sys, os

sys.path.append('../NP_focus')
from model import MRSModel

from torch.utils.data import Dataset, DataLoader
from persona_dataset import peronsa_loader
from simfunc import SimCSE, senBERT, BERTScore

def main():
    """settings"""
    model_type = args.model_type
    persona = args.persona
    weight = args.weight
    agg = args.agg
    
    """log"""    
    log_path = os.path.join('test.log'+persona)
    fileHandler = logging.FileHandler(log_path)
    
    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)    
    logger.setLevel(level=logging.DEBUG)       
    
    """model loadings"""
    model = MRSModel(model_type).cuda()
    modelfile = os.path.join('../model/NP_focus', model_type, 'model.bin')
    model.load_state_dict(torch.load(modelfile))    
    model.eval()
    print('Model Loading!!')    
    
    test_p1 = CalPER(model, args)
    logger.info("########################################")
    logger.info('모델: {}, persona: {}, agg: {}, weight: {}, test p@1: {}'.\
                format(model_type, persona, agg, weight, test_p1))
    
def CalPER(model, args):
    model_type, persona, weight, agg = args.model_type, args.persona, args.weight, args.agg
    
    """similarity persona"""
    if persona == "simcse":
        sim_model = SimCSE().cuda()
    elif persona == "nli":
        sim_model = senBERT().cuda()
    elif persona == "bertscore":
        sim_model = BERTScore()
        
    """dataset"""
    data_path = "../dataset/FoCus/valid_focus_persona.json"
    dataset = peronsa_loader(data_path, model_type)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=dataset.collate_fn)    
    
    true_index = 1
    pre1 = []
    for i_batch, (batch_input_token, batch_labels, batch_personas, batch_response) in enumerate(tqdm(data_loader, desc='eval_iteration')):
        true_probs = []
        batch_labels = batch_labels.tolist()
        
        if persona in ["simcse", "nli", "bertscore"]:
            cand_persona_scores = []
            for personas, responses in zip(batch_personas, batch_response): # batch = 1
                for response in responses:                    
                    persona_scores = sim_model(response, personas)
                    if agg == 'mean':
                        persona_score = sum(persona_scores)/len(persona_scores)
                    elif agg == 'max':
                        persona_score = max(persona_scores)
                    cand_persona_scores.append(persona_score)
            
        for input_token, input_label in zip(batch_input_token, batch_labels):                        
            input_token = input_token.unsqueeze(0).cuda()
            logits = model(input_token) # (1, C)
            
            prob = softmax(logits, 1) # (1, 2)
            true_prob = prob.squeeze(0)[true_index].item()
            true_probs.append(true_prob)
        
        if weight < 0:
            max_persona_score = max(cand_persona_scores)
            max_true_prob = max(true_probs)
            weight = max_true_prob/max_persona_score
        
        if persona in ["simcse", "nli", "bertscore"]:
            final_scores = []
            for cand_persona_score, true_prob in zip(cand_persona_scores, true_probs):
                final_scores.append(cand_persona_score*weight + true_prob)
        else:
            final_scores = true_probs
        max_ind = final_scores.index(max(final_scores))
        
        if batch_labels[max_ind] > 0:
            pre1.append(1)
        else:
            pre1.append(0)
    test_p1 = round(sum(pre1)/len(pre1)*100, 2)
    
    return test_p1
    
if __name__ == '__main__':
    torch.cuda.empty_cache()
    """Parameters"""
    parser  = argparse.ArgumentParser(description = "Persona Response selection" )
    
    parser.add_argument("--model_type", help = "pretrained model", default = 'roberta-large')
    parser.add_argument("--persona", help = "how to refelct persona (simcse or nli or bertscore)", default = None)
    parser.add_argument("--weight", type=float, help = "weighted sum", default = 0.5)
    parser.add_argument("--agg", type=str, help = "aggregation to personas", default = 'max')
            
    args = parser.parse_args()
    
    logger = logging.getLogger(__name__)
    streamHandler = logging.StreamHandler()
    
    main()