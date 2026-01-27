import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class Config:
    # hf_model_name = "michiyasunaga/BioLinkBERT-base"  
    # local_model_path = "classification/model/medhop/cache/biolinkbert/biolinkbert_cache"  
    
    # hf_model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    # local_model_path = "classification/model/medhop/cache/pubmedbert/pubmedbert_cache"
    
    hf_model_name = "dmis-lab/biobert-base-cased-v1.2"
    local_model_path = "classification\model\medhop\cache\biobert_v9\biobert_cache"
    
    max_length = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_safe_model_path():
    if os.path.exists(Config.local_model_path):
        return Config.local_model_path
    else:
        # Nếu không tìm thấy cache, trả về tên model gốc để thư viện tự tải
        return Config.hf_model_name

class BioLinkBERTClassifier(nn.Module):
    def __init__(self, model_path_or_name, freeze_until=4):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_path_or_name)
        
        for param in self.bert.embeddings.parameters(): param.requires_grad = False
        
        if freeze_until > 0:
            for layer in self.bert.encoder.layer[:freeze_until]:
                for param in layer.parameters(): param.requires_grad = False
                
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768, 3) 
        
    def forward(self, ids, mask):
        return self.classifier(self.dropout(self.bert(ids, mask).pooler_output))

def load_trained_model_medhop(model_file_path, tok_path):
    print(f"Loading Tokenizer from: {tok_path}")
    # Convert to absolute path if relative
    if not os.path.isabs(tok_path):
        tok_path = os.path.abspath(tok_path)
    tokenizer = AutoTokenizer.from_pretrained(tok_path, local_files_only=True)

    if not os.path.exists(model_file_path):
        raise FileNotFoundError(f"Không tìm thấy file weights tại: {model_file_path}")
   
    # 1. Xác định nơi lấy config/tokenizer 
    arch_path = get_safe_model_path()
    print(f"Loading architecture from: {arch_path}")
    
    # 2. Khởi tạo khung Model
    model = BioLinkBERTClassifier(arch_path, freeze_until=0)
    
    # 3. Load Weights đã train
    print(f"Loading weights from: {model_file_path}")
    checkpoint = torch.load(model_file_path, map_location=Config.device)
    model.load_state_dict(checkpoint)
    
    model.to(Config.device)
    model.eval()
    return model, tokenizer, Config.device

def inference_question(question, model, tokenizer):
    encoded = tokenizer(
        str(question), 
        truncation=True, 
        padding='max_length', 
        max_length=Config.max_length, 
        return_tensors='pt'
    )
    
    ids = encoded['input_ids'].to(Config.device)
    mask = encoded['attention_mask'].to(Config.device)
    
    with torch.no_grad():
        outputs = model(ids, mask)
        probs = torch.softmax(outputs, dim=1)
        pred_label_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_label_idx].item()
            
    return {
        "predicted_label": pred_label_idx,
        "confidence": round(confidence, 4),
        "all_probs": probs.cpu().numpy()[0]
    }
