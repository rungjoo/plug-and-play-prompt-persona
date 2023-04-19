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
    prompt_question = "what is your personality?"
    sim_model = None
    
    test_p1 = CalPER(model, prompt_question, test_path, args)
    logger.info('모델: {}, 데이터: {}, persona: {}, test p@1: {}'.\
            format(model_type, persona_type, persona_type, test_p1))
    logger.info('test p@1: {}'.format(test_p1))
    
def high_persona(persona_scores, personas, k, reverse=False):
    high_persona_utts = personas[:k]
    return high_persona_utts  
    
def CalPER(model, prompt_question, data_path, args):
    model_type, persona_type = args.model_type, args.persona_type
    data_type = args.data_type
    
    """dataset"""
    dataset = peronsa_loader(data_path, model_type)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=dataset.collate_fn)

    prompt_token = dataset.tokenizer.encode(dataset.tokenizer.sep_token + " " + prompt_question, add_special_tokens=False)
    
    true_index = 1
    pre1 = []
    for i_batch, (batch_input_token, batch_labels, batch_personas, batch_response) in enumerate(tqdm(data_loader, desc='eval_iteration')):
        response_true_probs, persona_true_probs = [], []
        batch_labels = batch_labels.tolist()
            
        max_persona_utts = []
        for personas, responses in zip(batch_personas, batch_response): # batch = 1
            for response in responses:
                high_persona_utts = personas
                max_persona_utts.append(high_persona_utts)
            
        for max_persona_utt_list, input_token, input_label in zip(max_persona_utts, batch_input_token, batch_labels):
            """ cls token 위치 찾기 """
            token_id_list = input_token.tolist()
            cls_pos = token_id_list.index(dataset.tokenizer.cls_token_id)
            
            """ sep token 위치 찾기 """
            for i in range(len(token_id_list)):
                if token_id_list[i] == dataset.tokenizer.sep_token_id:
                    sep_pos = i
                    
            """ persona tokens """
            persona_token = []
            persona_token += prompt_token
            persona_string = ""
            for max_persona_utt in max_persona_utt_list:
                persona_string += " " + max_persona_utt
            persona_token += dataset.tokenizer.encode(dataset.tokenizer.sep_token + persona_string, add_special_tokens=False)
#             persona_token += dataset.tokenizer.encode(dataset.tokenizer.sep_token + " " + max_persona_utt_list[0] + " " + max_persona_utt_list[1] + " " + max_persona_utt_list[2], add_special_tokens=False)
            persona_token = torch.tensor(persona_token)
            
            """ context tokens """
            context_token = input_token[:cls_pos]
            
            """ response tokens """
            response_token = input_token[sep_pos+1:]
            
            """ final tokens [persona; context; cls; sep; response] """
            original_token = torch.cat([context_token, input_token[cls_pos:sep_pos+1], response_token], 0)
            delete_length = original_token.shape[0]+persona_token.shape[0]-dataset.tokenizer.model_max_length
            if delete_length > 0:
                concat_token = torch.cat([persona_token, original_token[delete_length:]], 0)
            else:
                concat_token = torch.cat([persona_token, original_token], 0)
            concat_token = concat_token.unsqueeze(0).cuda()
            
            """ persona MRS 점수 """
            persona_logits = model(concat_token) # (1, C)
            
            persona_prob = softmax(persona_logits, 1) # (1, 2)
            persona_true_prob = persona_prob.squeeze(0)[true_index].item()
            persona_true_probs.append(persona_true_prob)
        
        final_scores = persona_true_probs
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
    
    parser.add_argument("--model_type", help = "pretrained model", default = 'roberta-base')
    parser.add_argument("--data_type", help = "personachat or focus", default = 'personachat')
    parser.add_argument("--persona_type", help = "original or revised", default = 'original')
            
    args = parser.parse_args()
    
    logger = logging.getLogger(__name__)
    streamHandler = logging.StreamHandler()
    
    main()