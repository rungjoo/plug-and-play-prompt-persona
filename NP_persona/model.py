from transformers import RobertaModel, RobertaTokenizer, RobertaConfig
from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys
import pdb

class MRSModel(nn.Module):
    def __init__(self, model_type, scratch=False):
        super(MRSModel, self).__init__()        
        
        model_path = model_type
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        if scratch:
            print("처음부터 학습")
            configuration = RobertaConfig.from_pretrained(model_path)
            self.model = RobertaModel(configuration)
        else:
            print("PLM 이용")
            self.model = AutoModel.from_pretrained(model_path)
        
        self.Wc = nn.Linear(self.model.config.hidden_size, 2) # for 적합도

    def forward(self, input_tokens):
        """
            input_tokens: (batch, len)
        """
        cls_positions = []
        for input_token in input_tokens:
            try:
                cls_position = input_token.tolist().index(self.tokenizer.cls_token_id)
            except:
                pdb.set_trace()
            cls_positions.append(cls_position) # (B)
            
        hidden_outs = self.model(input_tokens)['last_hidden_state']
        pred_outs = self.Wc(hidden_outs) # (B, L, C)            
            
        cls_outs = []
        for pred_out, cls_position in zip(pred_outs, cls_positions):
            # pred_out: (L, C)
            cls_out = pred_out[cls_position,:].unsqueeze(0) # (1, C)
            cls_outs.append(cls_out)
        
        cls_outs = torch.cat(cls_outs, 0) # (B, C)
        return cls_outs
        

