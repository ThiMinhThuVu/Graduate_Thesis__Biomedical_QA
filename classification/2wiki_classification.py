import torch
from transformers import AutoTokenizer, AutoModel
from torch import nn

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

def load_model(model_path, tokenizer_dir, model_name='bert-base-uncased', num_classes=4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Khởi tạo mô hình
    model = BERTClassifier(model_name, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    
    return model, tokenizer, device

def classify_question(model, tokenizer, device, question, max_length=128):
    # Tokenize input
    inputs = tokenizer(question, return_tensors='pt', max_length=max_length, truncation=True, padding='max_length')
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    # Predict
    with torch.no_grad():
        logits, _ = model(input_ids, attention_mask)  # Ignore attentions
        probs = torch.softmax(logits, dim=1)
        pred_label = torch.argmax(probs, dim=1).item()
    
    # Map label về tên
    label_map = {0: 'inference', 1: 'compositional', 2: 'comparison', 3: 'bridge_comparison'}
    return label_map.get(pred_label, 'unknown'), probs.cpu().numpy().flatten()

# Ví dụ sử dụng
if __name__ == "__main__":
    model_path = 'classification/model/2wiki/bert_classifier.pth'  # Thay bằng đường dẫn thực tế đến .pth file
    tokenizer_dir = 'classification/model/2wiki/tokenizer'  # Thay bằng đường dẫn thực tế đến thư mục tokenizer
    
    model, tokenizer, device = load_model(model_path, tokenizer_dir)
    
    question = "What is the capital of France?"  # Thay bằng câu hỏi cần phân loại
    predicted_type, probabilities = classify_question(model, tokenizer, device, question)
    
    print(f"Predicted type: {predicted_type}")
    print(f"Probabilities: {probabilities}")