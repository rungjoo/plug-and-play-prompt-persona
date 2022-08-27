import pdb
import argparse, logging
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn.functional import softmax
import sys, os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from torch.utils.data import Dataset, DataLoader
from persona_dataset import peronsa_loader
from simfunc import SimCSE, senBERT, BERTScore

def main():
    """settings"""
    persona = args.persona # simcse
    num_of_persona = args.num_of_persona

    """log"""
    data_type = args.data_type
    log_path = os.path.join(data_type+'_test.log'+persona)
    fileHandler = logging.FileHandler(log_path)
    
    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)
    logger.setLevel(level=logging.DEBUG)
    
    logger.info("#####################################")
    final_p, final_r, final_f = CalPER(args)

    logger.info('Dataset: focus, persona: {}, test@{} recall: {}'.\
            format(persona, num_of_persona, final_r))
    
def precision_recall_f1(persona_scores, batch_groundings, num_of_persona):
    hits_score = 0

    persona_scores_dict = {}
    for k, v in enumerate(persona_scores):
        persona_scores_dict[k] = v
    sort_persona_scores = sorted(persona_scores_dict.items(), key = lambda item: item[1], reverse = True)

    cand_persona = min(num_of_persona, len(sort_persona_scores))
    for i in range(cand_persona):
        key = sort_persona_scores[i][0]
        if batch_groundings[key]:
            hits_score += 1
    
    recall_num = 0
    for batch_grounding in batch_groundings:
        if batch_grounding:
            recall_num += 1
    precision = hits_score/num_of_persona
    if recall_num == 0:
        recall = 1
    else:
        recall = hits_score/recall_num
        
    if precision+recall == 0:
        f1 = 0
    else:
        f1 = 2*(precision*recall)/(precision+recall)
    return precision, recall, f1
    
def CalPER(args):
    persona= args.persona
    data_type = args.data_type
    num_of_persona = args.num_of_persona
    
    """similarity persona"""
    if persona == "simcse":
        sim_model = SimCSE().cuda()
    elif persona == "nli":
        sim_model = senBERT().cuda()
    elif persona == "bertscore":
        sim_model = BERTScore()
        
    """dataset"""
    data_path = "../datset/FoCus/valid_focus_persona.json"
    dataset = peronsa_loader(data_path, 'roberta-base')
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=dataset.collate_fn)

    precision_scores, recall_scores, f1_scores = [], [], []
    for i_batch, (batch_input_tokens, batch_labels, batch_personas, batch_response, batch_groundings) in enumerate(tqdm(data_loader, desc='eval_iteration')):
        persona_scores = []
        if persona in ["simcse", "nli", "bertscore"]:
            gt_ind = batch_labels.tolist().index(1)
            gt_response = batch_response[gt_ind]
            for persona_utt in batch_personas: # batch = 1
                persona_score = sim_model(gt_response, [persona_utt])[0]
                persona_scores.append(persona_score)
        precision_score, recall_score, f1_score = precision_recall_f1(persona_scores, batch_groundings, num_of_persona)        
        
        precision_scores.append(precision_score)
        recall_scores.append(recall_score)
        f1_scores.append(f1_score)
    
    final_p = round(sum(precision_scores)/len(precision_scores)*100, 2)
    final_r = round(sum(recall_scores)/len(recall_scores)*100, 2)
    final_f = round(sum(f1_scores)/len(f1_scores)*100, 2)
    return final_p, final_r, final_f
    
if __name__ == '__main__':
    torch.cuda.empty_cache()
    """Parameters"""
    parser  = argparse.ArgumentParser(description = "Persona Response selection" )
    
    parser.add_argument("--data_type", help = "focus", default = 'focus')
    parser.add_argument("--persona", type=str, help = "how to refelct persona", choices = ["simcse", "nli", "bertscore"], default = 'simcse')
    parser.add_argument("--num_of_persona", type=int, help = "how to use persona utterance", default = 1)
            
    args = parser.parse_args()
    
    logger = logging.getLogger(__name__)
    streamHandler = logging.StreamHandler()
    
    main()
    