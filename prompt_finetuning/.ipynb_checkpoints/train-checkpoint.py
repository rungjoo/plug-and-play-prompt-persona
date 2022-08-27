import pdb
import argparse, logging
from tqdm import tqdm

import torch
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup
from torch.nn.functional import softmax
import sys, os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from torch.utils.data import Dataset, DataLoader
from persona_dataset import peronsa_loader
from simfunc import SimCSE, senBERT, BERTScore

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
    prompt_question = "what is your personality?"
#     prompt_question2 = "tell me your personality."
#     prompt_question3 = "tell me more about yourself."

    """log"""
    data_type = args.data_type
    log_path = os.path.join(model_type+'_'+persona_type+'_train.log')
    fileHandler = logging.FileHandler(log_path)
    
    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)
    logger.setLevel(level=logging.DEBUG)
    
    """ model loadings """
    if data_type == "personachat":
        sys.path.append('../NP_persona')
        modelfile = os.path.join('../NP_persona', model_type, 'model.bin')
    else:
        sys.path.append('../NP_focus')
        modelfile = os.path.join('../NP_focus', model_type, 'model.bin')
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
    train_dataset = peronsa_loader(train_path, model_type)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=train_dataset.collate_fn)
    
    """similarity persona"""
    if persona == "simcse":
        sim_model = SimCSE().cuda()
    elif persona == "nli":
        sim_model = senBERT().cuda()
    elif persona == "bertscore":
        sim_model = BERTScore()    
    sim_model.eval()
    
    """ 하이퍼 파라미터들 """
    training_epochs = 5
    max_grad_norm = 10
    lr = 1e-6
    num_training_steps = len(train_dataset)*training_epochs
    num_warmup_steps = len(train_dataset)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr) # , eps=1e-06, weight_decay=0.01
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)    
    
    prompt_token = train_dataset.tokenizer.encode(train_dataset.tokenizer.sep_token + " " + prompt_question, add_special_tokens=False)
    logger.info("prompt question: "+prompt_question)
    best_dev_p1 = 0
    for epoch in tqdm(range(training_epochs)):
        model.train()
        for i_batch, (batch_input_token, batch_labels, batch_personas, batch_response) in enumerate(tqdm(train_loader, desc='train_iteration')):
            batch_labels = batch_labels.cuda()

            if persona in ["simcse", "nli", "bertscore"]:
                cand_persona_scores, max_persona_utts = [], []
                for personas, responses in zip(batch_personas, batch_response): # batch = 1
                    for response in responses:                    
                        persona_scores = sim_model(response, personas)
                        high_persona_utts = high_persona(persona_scores, personas, args.num_of_persona, args.reverse)
                        max_persona_utts.append(high_persona_utts)                    

            batch_persona_logits = []
            for max_persona_utt_list, input_token, input_label in zip(max_persona_utts, batch_input_token, batch_labels):            
                input_label = input_label.item()

                """ persona tokens """
                persona_token = []
                persona_token += prompt_token
                persona_string = ""
                for max_persona_utt in max_persona_utt_list:
                    persona_string += " " + max_persona_utt
                persona_token += train_dataset.tokenizer.encode(train_dataset.tokenizer.sep_token + persona_string, add_special_tokens=False)
#                 persona_token += dataset.tokenizer.encode(dataset.tokenizer.sep_token + " " + max_persona_utt_list[0] + " " + max_persona_utt_list[1] + " " + max_persona_utt_list[2], add_special_tokens=False)
                persona_token = torch.tensor(persona_token)            

                """ final tokens [persona; context; cls; sep; response] """
                delete_length = input_token.shape[0]+persona_token.shape[0]-train_dataset.tokenizer.model_max_length
                if delete_length > 0:
                    concat_token = torch.cat([persona_token, input_token[delete_length:]], 0)
                else:
                    concat_token = torch.cat([persona_token, input_token], 0)
                concat_token = concat_token.unsqueeze(0).cuda()                     

                """ persona MRS 점수 """
                persona_logits = model(concat_token) # (1, C)
                batch_persona_logits.append(persona_logits)
            batch_persona_logits = torch.cat(batch_persona_logits, 0) # (B, C)        

            """Loss calculation & training"""
            loss_val = CELoss(batch_persona_logits, batch_labels)

            loss_val.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        dev_p1 = CalPER(model, prompt_question, sim_model, dev_path, args)
        logger.info('모델: {}, 데이터: {}, persona: {}, number of persona: {}+{}, dev p@1: {}'.\
                format(model_type, persona_type, persona, num_of_persona, args.reverse, dev_p1))

        if dev_p1 > best_dev_p1:
            best_dev_p1 = dev_p1
            test_p1 = CalPER(model, prompt_question, sim_model, test_path, args)
            logger.info('모델: {}, 데이터: {}, persona: {}, number of persona: {}+{}, test p@1: {}'.\
                    format(model_type, persona_type, persona, num_of_persona, args.reverse, test_p1))
            SaveModel(model, model_type+'_'+persona_type)
        logger.info('Best test p@1: {}'.format(test_p1))


def SaveModel(model, path):
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model.state_dict(), os.path.join(path, 'model.bin'))
    
def CELoss(pred_outs, labels):
    """
        pred_outs: [batch, clsNum]
        labels: [batch]
    """
    loss = nn.CrossEntropyLoss()
    loss_val = loss(pred_outs, labels)
    return loss_val

def high_persona(persona_scores, personas, k, reverse=False):
    high_persona_utts = []
    sort_persona_scores = sorted(persona_scores, reverse=True)
    for i in range(k):
        persona_score = sort_persona_scores[i]
        persona_ind = persona_scores.index(persona_score)
        high_persona_utts.append(personas[persona_ind])
    if reverse:
        high_persona_utts.reverse()
    return high_persona_utts    
    
def CalPER(model, prompt_question, sim_model, data_path, args):
    model.eval()
    model_type, persona_type, persona = args.model_type, args.persona_type, args.persona
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
            
        if persona in ["simcse", "nli", "bertscore"]:
            cand_persona_scores, max_persona_utts = [], []
            for personas, responses in zip(batch_personas, batch_response): # batch = 1
                for response in responses:                    
                    persona_scores = sim_model(response, personas)
                    high_persona_utts = high_persona(persona_scores, personas, args.num_of_persona, args.reverse)
                    max_persona_utts.append(high_persona_utts)                    
            
        for max_persona_utt_list, input_token, input_label in zip(max_persona_utts, batch_input_token, batch_labels):
            """ persona tokens """
            persona_token = []
            persona_token += prompt_token
            persona_string = ""
            for max_persona_utt in max_persona_utt_list:
                persona_string += " " + max_persona_utt
            persona_token += dataset.tokenizer.encode(dataset.tokenizer.sep_token + persona_string, add_special_tokens=False)
#             persona_token += dataset.tokenizer.encode(dataset.tokenizer.sep_token + " " + max_persona_utt_list[0] + " " + max_persona_utt_list[1] + " " + max_persona_utt_list[2], add_special_tokens=False)
            persona_token = torch.tensor(persona_token)            
            
            """ final tokens [sep; persona; context; cls; sep; response] """
            delete_length = input_token.shape[0]+persona_token.shape[0]-dataset.tokenizer.model_max_length
            if delete_length > 0:
                concat_token = torch.cat([persona_token, input_token[delete_length:]], 0)
            else:
                concat_token = torch.cat([persona_token, input_token], 0)
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
    
    parser.add_argument("--model_type", help = "pretrained model", default = 'roberta-large')
    
    parser.add_argument("--data_type", help = "personachat or focus", default = 'personachat')
    parser.add_argument("--persona_type", help = "original or revised", default = 'original')
    parser.add_argument("--persona", type=str, help = "how to refelct persona", choices = ["simcse", "nli", "bertscore"], default = 'simcse')
    parser.add_argument("--num_of_persona", type=int, help = "how to use persona utterance", default = 1)
    parser.add_argument('--reverse', help='persona ordering', action="store_true")
            
    args = parser.parse_args()
    
    logger = logging.getLogger(__name__)
    streamHandler = logging.StreamHandler()
    
    main()