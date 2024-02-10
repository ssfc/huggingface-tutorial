import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

# 2.5.1 Models expect a batch of inputs
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(device)  # 将模型移动到GPU

sequence = "I've been waiting for a HuggingFace course my whole life."

tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)

input_ids = torch.tensor([ids]).to(device)  # 将输入数据移到GPU
print("Input IDs:", input_ids)

output = model(input_ids)
print("Logits:", output.logits)

tokenized_inputs = tokenizer(sequence, return_tensors="pt")
for key in tokenized_inputs.keys():
    tokenized_inputs[key] = tokenized_inputs[key].to(device)  # 将输入数据移到GPU
print(tokenized_inputs)
print(tokenized_inputs["input_ids"])

# 2.5.2 Padding the inputs
model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(device)

sequence1_ids = [[200, 200, 200]]
sequence2_ids = [[200, 200]]
batched_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id],
]
print("batched_ids: ", batched_ids)

print(model(torch.tensor(sequence1_ids).to(device)).logits)
print(model(torch.tensor(sequence2_ids).to(device)).logits)
print("without attention mask: ", model(torch.tensor(batched_ids).to(device)).logits)

# 2.5.3 Attention masks
batched_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id],
]

# Attention masks
attention_mask = [
    [1, 1, 1],
    [1, 1, 0],
]

outputs = model(torch.tensor(batched_ids).to(device), attention_mask=torch.tensor(attention_mask).to(device))
print("with attention mask: ", outputs.logits)
print("GPU available: ", torch.cuda.is_available())
print("device: ", device)
