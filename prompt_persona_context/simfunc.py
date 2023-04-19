import torch
import torch.nn as nn
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer
from datasets import load_metric
from nltk.translate.bleu_score import sentence_bleu

class SimCSE(nn.Module):
    def __init__(self):
        super(SimCSE, self).__init__()  
        model_name = "princeton-nlp/sup-simcse-bert-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
    def forward(self, response, personas):
        """
            texts = [
                "I like book",
                'A woman is reading.', 
                'A man is playing a guitar.'
            ]
        """
        texts = [response] + personas
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        for k, v in inputs.items():
            inputs[k] = v.cuda()

        # Get the embeddings
        with torch.no_grad():
            embeddings = self.model(**inputs, output_hidden_states=True, return_dict=True).pooler_output.cpu()

        res_emb = embeddings[0]
        per_embs = embeddings[1:]
        cosine_sim = []
        for per_emb in per_embs:
            cosine_sim.append(1 - cosine(embeddings[0], per_emb))

        return cosine_sim
    
    
class senBERT(nn.Module):
    def __init__(self):
        super(senBERT, self).__init__()  
        model_path = '/data/project/rw/rung/model/bert-base-nli-mean-tokens'
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)        
        
    def forward(self, response, personas):
        """
            texts = [
                "I like book",
                'A woman is reading.', 
                'A man is playing a guitar.'
            ]
        """
        texts = [response] + personas
        
        # Tokenize sentences
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        for k, v in inputs.items():
            inputs[k] = v.cuda()

        # Get the embeddings
        with torch.no_grad():
            model_output = self.model(**inputs)
            embeddings = self.mean_pooling(model_output, inputs['attention_mask']).cpu()

        res_emb = embeddings[0]
        per_embs = embeddings[1:]
        cosine_sim = []
        for per_emb in per_embs:
            cosine_sim.append(1 - cosine(embeddings[0], per_emb))

        return cosine_sim    
    
class BERTScore(nn.Module):    
    def __init__(self):
        super(BERTScore, self).__init__()
        self.metric = load_metric("bertscore")
    def forward(self, response, personas):
        responses = [response]*len(personas)
        
        results = self.metric.compute(predictions=responses, references=personas, lang="en")
        f1 = results['f1']
        
        return f1
    
class BLEUScore():
    def __init__(self):
        pass
    def __call__(self, response, personas):
        response = response.split(" ")
        
        scores = []
        for persona in personas:
            persona = persona.split(" ")
            score = sentence_bleu([response], persona, weights=(1.0, 0, 0, 0))
            scores.append(score)
        
        return scores