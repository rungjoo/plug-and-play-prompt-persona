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
from simfunc import SimCSE, senBERT, BERTScore, BLEUScore

def main():
    """settings"""
    model_type = args.model_type
    persona = args.persona # simcse
    persona_type = args.persona_type # original
    num_of_persona = args.num_of_persona
    if args.reverse:
        print("Input: [row persona -> high persona -> context]")
    else:
        print("Input: [high persona -> row persona -> context]")
    
    """ prompt """
    prompt_question1 = "what is your personality?"
    prompt_question2 = "tell me your personality."
    prompt_question3 = "tell me more about yourself."
#     prompt_questions = [prompt_question2, prompt_question3]
    prompt_questions = [prompt_question1]

    """log"""
    data_type = args.data_type
    log_path = f"{data_type}_{persona}_test.log"
    fileHandler = logging.FileHandler(log_path)
    
    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)
    logger.setLevel(level=logging.DEBUG)
    
    """model loadings"""
    if args.scratch:
        model_save = f"{model_type}_scratch"
    else:
        model_save = model_type
    if data_type == "personachat":
        sys.path.append('../NP_persona')        
        modelfile = os.path.join('../model/NP_persona', model_save, 'model.bin')
    else:
        sys.path.append('../NP_focus')
        modelfile = os.path.join('../model/NP_focus', model_save, 'model.bin')
    from model import MRSModel
    model = MRSModel(model_type).cuda()    
    model.load_state_dict(torch.load(modelfile))    
    model.eval()
    print('Model Loading!!')    
    
    logger.info("#####################################")
    for prompt_question in prompt_questions:
        test_p1 = CalPER(model, prompt_question, args)

        logger.info("prompt question: "+prompt_question)
        logger.info('모델: {}, 데이터: {}, persona: {}, number of persona: {}+{}, test p@1: {}'.\
                format(model_type, persona_type, persona, num_of_persona, args.reverse, test_p1))
    
def high_persona(persona_scores, personas, k, reverse=False):
    high_persona_utts = []
    sort_persona_scores = sorted(persona_scores, reverse=True)
    cand_nums = min(k,len(sort_persona_scores))
    for i in range(cand_nums):
        persona_score = sort_persona_scores[i]
        persona_ind = persona_scores.index(persona_score)
        high_persona_utts.append(personas[persona_ind])
    if reverse:
        high_persona_utts.reverse()
    return high_persona_utts    
    
def CalPER(model, prompt_question, args):
    model_type, persona_type, persona = args.model_type, args.persona_type, args.persona
    data_type = args.data_type
    
    """similarity persona"""
    if persona == "simcse":
        sim_model = SimCSE().cuda()
        sim_model.eval()
    elif persona == "nli":
        sim_model = senBERT().cuda()
        sim_model.eval()
    elif persona == "bertscore":
        sim_model = BERTScore()
        sim_model.eval()
    elif persona == 'bleuscore':
        sim_model = BLEUScore()    
        
    """dataset"""
    if data_type == 'personachat':
        data_path = "../dataset/personachat/test_both_" + persona_type + ".json"
    elif data_type == 'focus':
        data_path = "../dataset/FoCus/valid_focus_persona.json"
    dataset = peronsa_loader(data_path, model_type)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=dataset.collate_fn)

    prompt_token = dataset.tokenizer.encode(dataset.tokenizer.sep_token + " " + prompt_question, add_special_tokens=False)
    
    true_index = 1
    pre1 = []
    for i_batch, (batch_input_token, batch_labels, batch_personas, batch_response) in enumerate(tqdm(data_loader, desc='eval_iteration')):
        response_true_probs, persona_true_probs = [], []
        batch_labels = batch_labels.tolist()
            
        if persona in ["simcse", "nli", "bertscore", "bleuscore"]:
            cand_persona_scores, max_persona_utts = [], []
            for personas, responses in zip(batch_personas, batch_response): # batch = 1
                for response in responses:                    
                    persona_scores = sim_model(response, personas)
                    high_persona_utts = high_persona(persona_scores, personas, args.num_of_persona, args.reverse)
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
            
            """ response MRS 점수 """
            input_token = input_token.unsqueeze(0).cuda()
            response_logits = model(input_token) # (1, C)
            
            response_prob = softmax(response_logits, 1) # (1, 2)
            response_true_prob = response_prob.squeeze(0)[true_index].item()
            response_true_probs.append(response_true_prob)
        
        
        if persona in ["simcse", "nli", "bertscore"]:
            final_scores = []
            for response_true_prob, persona_true_prob in zip(response_true_probs, persona_true_probs):
                final_scores.append(persona_true_prob)
        else:
            final_scores = response_true_probs
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
    
    parser.add_argument("--data_type", help = "personachat or focus", default = 'personachat')
    parser.add_argument("--persona_type", help = "original or revised", default = 'original')
    parser.add_argument("--persona", type=str, help = "how to refelct persona", choices = ["simcse", "nli", "bertscore", "bleuscore"], default = 'simcse')    
    parser.add_argument("--num_of_persona", type=int, help = "how to use persona utterance", default = 1)
    parser.add_argument('--reverse', help='persona ordering', action="store_true")
    parser.add_argument('--scratch', help='training from scratch', action="store_true")
            
    args = parser.parse_args()
    
    logger = logging.getLogger(__name__)
    streamHandler = logging.StreamHandler()
    
    main()