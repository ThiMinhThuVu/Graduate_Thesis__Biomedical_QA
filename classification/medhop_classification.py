import os
import json
import time
import csv
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import gc
import shutil
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, classification_report

# ==========================================
# 1. CONFIGURATION
# ==========================================
class Config:
    seed = 42
    hf_model_name = "michiyasunaga/BioLinkBERT-base"
    local_model_path = "./biolinkbert_cache"  # Folder cache cục bộ    
    # Đường dẫn đến folder chứa best_model.pt đã train 
    input_model_folder = "/kaggle/input/medhop-classification-sequential"
    
    max_length = 128
    batch_size = 16
    scheduler_epochs = 15
    run_epochs = 15
    lr_logic = 2e-5
    lr_head = 2e-5
    freeze_bottom_until = 4
    weight_decay = 0.01
    label_smoothing = 0.1
    warmup_ratio = 0.1
    grad_clip = 1.0

    train_path = "/kaggle/input/fixed-dataset/train_all_v4.csv"
    val_path   = "/kaggle/input/fixed-dataset/val_all_v4.csv"
    medhop_path = "/kaggle/input/fixed-dataset/medhop_parallel_v2.csv"

    work_dir = "." 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. HELPER: SAFE PATH LOADING
# ==========================================
def get_safe_model_path():
    """
    Hàm quan trọng để tránh lỗi HFValidationError:
    - Kiểm tra xem folder cache './biolinkbert_cache' có tồn tại không.
    - Nếu CÓ: Trả về đường dẫn folder cache (dùng offline).
    - Nếu KHÔNG: Trả về tên 'michiyasunaga/BioLinkBERT-base' (để tải online).
    """
    if os.path.exists(Config.local_model_path):
        return Config.local_model_path
    else:
        # Nếu không tìm thấy cache, trả về tên model gốc để thư viện tự tải
        return Config.hf_model_name

class BioLinkBERTClassifier(nn.Module):
    def __init__(self, model_path_or_name, freeze_until=4):
        super().__init__()
        # model_path_or_name sẽ được xử lý bởi get_safe_model_path() trước khi truyền vào
        self.bert = AutoModel.from_pretrained(model_path_or_name)
        
        # Freezing logic
        for param in self.bert.embeddings.parameters(): param.requires_grad = False
        
        if freeze_until > 0:
            for layer in self.bert.encoder.layer[:freeze_until]:
                for param in layer.parameters(): param.requires_grad = False
                
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768, 3) 
        
    def forward(self, ids, mask):
        return self.classifier(self.dropout(self.bert(ids, mask).pooler_output))
    
# ==========================================
# 5. INFERENCE UTILITIES 
# ==========================================
def load_trained_model_medhop(model_file_path, tok_path):
    # model_file_path = os.path.join(Config.input_model_folder, "best_model.pt")
    # tok_path = get_safe_model_path()
    print(f"Loading Tokenizer from: {tok_path}")
    tokenizer = AutoTokenizer.from_pretrained(tok_path)

    if not os.path.exists(model_file_path):
        raise FileNotFoundError(f"Không tìm thấy file weights tại: {model_file_path}")

    # 1. Xác định nơi lấy config/tokenizer (Local hay Online)
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
