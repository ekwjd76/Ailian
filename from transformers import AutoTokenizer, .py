from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_NAME = "roberta-base-openai-detector"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

def model_ai_score(text: str) -> float:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)[0]
    ai_prob = probs[1].item()
    return ai_prob  # 0~1 사이 값
