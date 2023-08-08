import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 2.5.1 Models expect a batch of inputs
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence = "I've been waiting for a HuggingFace course my whole life."

tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)

input_ids = torch.tensor([ids])
print("Input IDs:", input_ids)

output = model(input_ids)
print("Logits:", output.logits)

tokenized_inputs = tokenizer(sequence, return_tensors="pt")
print(tokenized_inputs)
print(tokenized_inputs["input_ids"])

# 2.5.2 Padding the inputs
# 类似于computer vision中给缺失的部分填充上。也类似于python的broadcast.
# 填充短的句子，截断长的句子，使所有句子表示长度相同。
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence1_ids = [[200, 200, 200]]
sequence2_ids = [[200, 200]]
batched_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id],
]
print("batched_ids: ", batched_ids)

print(model(torch.tensor(sequence1_ids)).logits)
print(model(torch.tensor(sequence2_ids)).logits)
print("without attention mask: ", model(torch.tensor(batched_ids)).logits)

# 2.5.3 Attention masks
batched_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id],
]

attention_mask = [
    [1, 1, 1],
    [1, 1, 0],
]

outputs = model(torch.tensor(batched_ids), attention_mask=torch.tensor(attention_mask))
print("with attention mask: ", outputs.logits)

# 2.5.4 Longer sequences



