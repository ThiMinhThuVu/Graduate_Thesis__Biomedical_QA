import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_model(model_path, tokenizer_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load tokenizer từ thư mục
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    
    # Khởi tạo model từ pretrained name gốc (như trong notebook)
    model = AutoModelForSequenceClassification.from_pretrained('dmis-lab/biobert-base-cased-v1.1', num_labels=2)
    
    # Load state_dict từ file .pth
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model, tokenizer, device

def classify_question(model, tokenizer, device, question, max_length=128):
    # Tokenize input
    inputs = tokenizer(question, return_tensors='pt', max_length=max_length, truncation=True, padding='max_length')
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    probs = torch.softmax(logits, dim=1)
    pred_label = torch.argmax(probs, dim=1).item()
    
    # Map label về tên (dựa trên is_sequence: 0 = non-sequence, 1 = sequence)
    label_map = {0: 'non-sequence', 1: 'sequence'}
    return label_map.get(pred_label, 'unknown'), probs.cpu().numpy().flatten()

# Ví dụ sử dụng
if __name__ == "__main__":
    model_path = 'classification/model/medhop/biobert_classifier.pth'  # Đường dẫn file .pth thực tế
    tokenizer_dir = 'classification/model/medhop/seq_tokenizer'  # Đường dẫn thư mục tokenizer thực tế
    
    model, tokenizer, device = load_model(model_path, tokenizer_dir)
    
    question = "What is a common risk factor for both ectopic pregnancy and endometriosis?"  # Thay bằng câu hỏi cần phân loại
    predicted_type, probabilities = classify_question(model, tokenizer, device, question)
    
    print(f"Predicted type: {predicted_type}")
    print(f"Probabilities: {probabilities}")