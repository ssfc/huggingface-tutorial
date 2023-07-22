import torch
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification

'''
# Same as before
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = [
    "I've been waiting for a HuggingFace course my whole life.",
    "This course is amazing!",
]
batch = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")

# This is new
batch["labels"] = torch.tensor([1, 1])

optimizer = AdamW(model.parameters())
loss = model(**batch).loss
loss.backward()
optimizer.step()
'''

from datasets import load_dataset

raw_datasets = load_dataset("glue", "mrpc")
print(raw_datasets)

raw_train_dataset = raw_datasets["train"]
print(raw_train_dataset[0])
print(raw_train_dataset.features)

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenized_sentences_1 = tokenizer(raw_datasets["train"]["sentence1"])
tokenized_sentences_2 = tokenizer(raw_datasets["train"]["sentence2"])
# print("tokenized_sentences_1", tokenized_sentences_1)
# Comment: 卧槽，太长了，屏幕放不下。
# print("tokenized_sentences_2", tokenized_sentences_2)

inputs = tokenizer("This is the first sentence.", "This is the second one.")
print("inputs: ", inputs)
print(tokenizer.convert_ids_to_tokens(inputs["input_ids"]))

