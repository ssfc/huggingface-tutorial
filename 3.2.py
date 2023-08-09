# 3.2.1 Preprocessing a dataset
import torch
# Q: huggingface的AdamW是干啥的？
# Adam 优化算法是一种常用的梯度下降优化算法。
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding

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
print(raw_train_dataset.features)  # 这是一个鉴别两个句子是否同义的数据集

# 3.2.2 Dynamic padding
checkpoint = "bert-base-uncased"  # 用的是bert呀
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenized_sentences_1 = tokenizer(raw_datasets["train"]["sentence1"])
tokenized_sentences_2 = tokenizer(raw_datasets["train"]["sentence2"])
# print("tokenized_sentences_1", tokenized_sentences_1)
# Comment: 卧槽，太长了，屏幕放不下。
# print("tokenized_sentences_2", tokenized_sentences_2)

inputs = tokenizer("This is the first sentence.", "This is the second one.")
print("inputs: ", inputs)
print(tokenizer.convert_ids_to_tokens(inputs["input_ids"]))


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
print("tokenized datasets: ", tokenized_datasets)

samples = tokenized_datasets["train"][:8]  # 从其训练集中获取前8个样本
samples = {k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]}
print([len(x) for x in samples["input_ids"]])

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
batch = data_collator(samples)
print({k: v.shape for k, v in batch.items()})






