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

def main():
    """settings"""
    model_type = args.model_type
    persona_type = args.persona_type

    """log"""
    data_type = args.data_type
    log_path = os.path.join(f"{model_type}_{persona_type}.log")
    fileHandler = logging.FileHandler(log_path)
    
    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)
    logger.setLevel(level=logging.DEBUG)
    
    """ model loadings """
    sys.path.append('../NP_persona')
    modelfile = os.path.join(f'../model/prompt_finetuning/{model_type}_{args.persona_type}', 'model.bin')
    from model import MRSModel
    model = MRSModel(model_type).cuda()    
    model.load_state_dict(torch.load(modelfile))    
    
    print('Model Loading!!')
    
    """ data loading """
    if data_type == 'personachat':
        train_path = "../dataset/personachat/train_both_" + persona_type + ".json"
        dev_path = "../dataset/personachat/valid_both_" + persona_type + ".json"
        test_path = "../dataset/personachat/test_both_" + persona_type + ".json"
    else:
        print("ERROR data_type")
    
    """similarity persona"""
    prompt_question = None
    sim_model = None
    
    test_p1 = CalPER(model, prompt_question, sim_model, test_path, args)
    logger.info('모델: {}, 데이터: {}, persona: {}, test p@1: {}'.\
            format(model_type, persona_type, persona, test_p1))
    logger.info('test p@1: {}'.format(test_p1))

def CalPER(model, prompt_question, sim_model, data_path, args):
    model.eval()
    model_type, persona_type = args.model_type, args.persona_type
    data_type = args.data_type    
        
    """dataset"""
    dataset = peronsa_loader(data_path, model_type)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=dataset.collate_fn)
    
    true_index = 1
    pre1 = []
    for i_batch, (batch_input_token, batch_labels, batch_personas, batch_response) in enumerate(tqdm(data_loader, desc='eval_iteration')):
        response_true_probs, persona_true_probs = [], []
        batch_labels = batch_labels.tolist()           
            
        for input_token, input_label in zip(batch_input_token, batch_labels):
            """ final tokens [sep; context; cls; sep; response] """
            delete_length = input_token.shape[0]-dataset.tokenizer.model_max_length
            if delete_length > 0:
                concat_token = input_token[delete_length:]
            else:
                concat_token = input_token
            concat_token = concat_token.unsqueeze(0).cuda()            
            
            """ persona MRS 점수 """
            persona_logits = model(concat_token) # (1, C)
            
            persona_prob = softmax(persona_logits, 1) # (1, 2)
            persona_true_prob = persona_prob.squeeze(0)[true_index].item()
            persona_true_probs.append(persona_true_prob)
        
        max_ind = persona_true_probs.index(max(persona_true_probs))
        
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
    
    parser.add_argument("--model_type", help = "pretrained model", default = 'roberta-base')
    
    parser.add_argument("--data_type", help = "personachat", default = 'personachat')
    parser.add_argument("--persona_type", help = "original or revised", default = 'original')
            
    args = parser.parse_args()
    
    logger = logging.getLogger(__name__)
    streamHandler = logging.StreamHandler()
    
    main()
