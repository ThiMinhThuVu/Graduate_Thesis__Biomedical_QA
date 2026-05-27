import torch
from transformers import AutoTokenizer, AutoModel
from torch import nn
import yaml
import os
import numpy as np

from classification.medhop_classification import load_trained_model_medhop, inference_question

class BERTClassifier(nn.Module):
    def __init__(self, model_name='bert-base-uncased', num_classes=4, dropout=0.3):
        super(BERTClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name, output_attentions=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits, outputs.attentions

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_model_classification(model_type, model_path, tokenizer_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model_type == 'wiki':
        model = BERTClassifier(model_name='bert-base-uncased', num_classes=4)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        return model, tokenizer, device
    elif model_type == 'medhop':
        return load_trained_model_medhop(model_path, tokenizer_dir)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

def classify_question(model_type, model, tokenizer, device, question, max_length=128):
    inputs = tokenizer(
        question, return_tensors='pt', max_length=max_length, truncation=True, padding='max_length'
    )
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    with torch.no_grad():
        if model_type == 'wiki':
            logits, _ = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)
            pred_label = torch.argmax(probs, dim=1).item()
            label_map = {0: 'inference', 1: 'compositional', 2: 'comparison', 3: 'bridge_comparison'}
        elif model_type == 'medhop':            
            output = inference_question(question, model, tokenizer)
            label_map = {0: 'direct', 1: 'sequence', 2: 'parallel'}
            # probs = output['confidence']
            probs = output['all_probs']
            pred_label = output['predicted_label']
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
    # If probs is already a numpy array, do not call .cpu()
    if hasattr(probs, 'cpu'):
        probs_out = probs.cpu().numpy().flatten()
    else:
        probs_out = np.array(probs).flatten()
    return label_map.get(pred_label, 'unknown'), probs_out

if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    config = load_config(config_path)

    # Lấy loại model từ config (wiki hoặc medhop)
    dataset_name = config['run']['dataset']
    model_type = dataset_name  # 'wiki' hoặc 'medhop'
    model_path = config['classification'][model_type]['model_path']
    tokenizer_dir = config['classification'][model_type]['tokenizer_dir']

    model, tokenizer, device = load_model_classification(model_type, model_path, tokenizer_dir)

    question = "What is a common risk factor for both ectopic pregnancy and endometriosis?"
    # question = "The disease that Pseudo-Hurler polydystrophy resembles is classified as a member of a class of diseases related to a type of polysaccharides. This group of polysaccharides is classified into how many types, based on their core disaccharide structures?"

    predicted_type, probabilities = classify_question(model_type, model, tokenizer, device, question)
    print(f"Predicted type: {predicted_type}")
    print(f"Probabilities: {probabilities}")