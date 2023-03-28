import torch
from torch.utils.data import Dataset, DataLoader
import json
import pdb, os
import random
from transformers import RobertaTokenizer, AutoTokenizer
    
class peronsa_loader(Dataset):
    def __init__(self, data_path, model_type):
        with open(data_path, "r") as json_file:
            self.session_json = json.load(json_file)
                
        model_path = model_type
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)        
        
    def __len__(self):
        return len(self.session_json)

    def __getitem__(self, idx):
        return self.session_json[str(idx)]
    
    def encode_truncated(self, text):
        max_length = self.tokenizer.model_max_length
        tokenized_tokens = self.tokenizer.encode(text, add_special_tokens=False)
        truncated_tokens = tokenized_tokens[-max_length:]    

        return truncated_tokens  
    
    def padding(self, ids_list):
        max_len = 0
        for ids in ids_list:
            if len(ids) > max_len:
                max_len = len(ids)

        pad_ids = []
        for ids in ids_list:
            pad_len = max_len-len(ids)
            add_ids = [self.tokenizer.pad_token_id for _ in range(pad_len)]

            pad_ids.append(ids+add_ids)

        return torch.tensor(pad_ids)     
    
    def collate_fn(self, data):
        '''
            batch = 1
            input:
                data: [batch_session]
            return:
                batch_input_tokens: [(L), (L), (L), ...]
                batch_labels: (20)
        '''
        batch_input_tokens = []
        batch_labels = []
        batch_response = []
        batch_personas = []
            
        for session in data: 
            persona = session['persona']
            context = session['context']
            positive_response = session['postivie_response']
            negative_responses = session['negative_responses']

            input_string = ''
            for turn, utt in enumerate(context):
                if turn > len(session)-6:
                    input_string += ' ' + self.tokenizer.sep_token + ' '
                    input_string += utt

            input_string += ' ' + self.tokenizer.cls_token
            input_string += ' ' + self.tokenizer.sep_token + ' '

            cand_string = input_string + positive_response
            batch_labels.append(1)
            cand_tokens = self.encode_truncated(cand_string.strip())
            batch_input_tokens.append(cand_tokens)

            for negative_response in negative_responses:
                cand_string = input_string + negative_response
                batch_labels.append(0)
                cand_tokens = self.encode_truncated(cand_string.strip())
                batch_input_tokens.append(cand_tokens)
            
            batch_personas.append(persona)
            batch_response.append([positive_response] + negative_responses)
        
        batch_input_tokens = self.padding(batch_input_tokens)
        
        return batch_input_tokens, torch.tensor(batch_labels), batch_personas, batch_response
    
class peronsa_test_loader(Dataset):
    def __init__(self, data_path, model_type):
        with open(data_path, "r") as json_file:
            self.session_json = json.load(json_file)
                
        model_path = model_type
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)        
        
    def __len__(self):
        return len(self.session_json)

    def __getitem__(self, idx):
        return self.session_json[str(idx)]
    
    def encode_truncated(self, text):
        max_length = self.tokenizer.model_max_length
        tokenized_tokens = self.tokenizer.encode(text, add_special_tokens=False)
        truncated_tokens = tokenized_tokens[-max_length:]    

        return truncated_tokens  
    
    def padding(self, ids_list):
        max_len = 0
        for ids in ids_list:
            if len(ids) > max_len:
                max_len = len(ids)

        pad_ids = []
        for ids in ids_list:
            pad_len = max_len-len(ids)
            add_ids = [self.tokenizer.pad_token_id for _ in range(pad_len)]

            pad_ids.append(ids+add_ids)

        return torch.tensor(pad_ids)     
    
    def collate_fn(self, data):
        '''
            batch = 1
            input:
                data: [batch_session]
            return:
                batch_input_tokens: [(L), (L), (L), ...]
                batch_labels: (20)
        '''
        batch_input_tokens = []
        batch_labels = []
        batch_response = []
        batch_personas = []
            
        for session in data: 
            persona = session['persona']
            context = session['context']
            positive_response = session['postivie_response']
            negative_responses = session['negative_responses']

            input_string = ''
            for turn, utt in enumerate(context):
                if turn > len(session)-6:
                    input_string += ' ' + self.tokenizer.sep_token + ' '
                    input_string += utt

            input_string += ' ' + self.tokenizer.cls_token
            input_string += ' ' + self.tokenizer.sep_token + ' '

            cand_string = input_string + positive_response
            batch_labels.append(1)
            cand_tokens = self.encode_truncated(cand_string.strip())
            batch_input_tokens.append(torch.tensor(cand_tokens))            

            for negative_response in negative_responses:
                cand_string = input_string + negative_response
                batch_labels.append(0)
                cand_tokens = self.encode_truncated(cand_string.strip())
                batch_input_tokens.append(torch.tensor(cand_tokens))    
            
            batch_personas.append(persona)
            batch_response.append([positive_response] + negative_responses)
        return batch_input_tokens, torch.tensor(batch_labels), batch_personas, batch_response