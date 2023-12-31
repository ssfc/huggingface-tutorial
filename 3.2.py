# 3.2.1 Preprocessing a dataset
import torch
# Q: huggingface的AdamW是干啥的？
# Adam 优化算法是一种常用的梯度下降优化算法。
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification
# 在进行文本序列任务的训练时，由于输入序列长度可能不一致，我们需要将输入序列进行填充，以便将它们组织成一个批次（batch），并输入到模型中进行训练。
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

checkpoint = "bert-base-uncased"  # 用的是bert呀
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenized_sentences_1 = tokenizer(raw_datasets["train"]["sentence1"])
tokenized_sentences_2 = tokenizer(raw_datasets["train"]["sentence2"])
# print("tokenized_sentences_1", tokenized_sentences_1)
# Comment: 卧槽，太长了，屏幕放不下。
# print("tokenized_sentences_2", tokenized_sentences_2)

# Q: 结果中的token_type_ids是什么？
# 在 Hugging Face Transformers 库中，`token_type_ids` 是用于处理输入序列中的分段信息的一种编码方式。
# 对于一些预训练模型（例如 BERT、RoBERTa 等），它们在输入时需要同时考虑两个句子或分段的信息，因此需要一种方式来区分不同分段的内容。
# 在处理文本序列时，通常将输入序列切分成多个片段，每个片段对应一个输入序列。
# 例如，当处理句子对任务时，一个输入序列可能包含两个句子，其中一个句子放在前面，另一个句子放在后面，中间可能有一个特殊的分隔符。`token_type_ids` 就是用来区分这些不同片段的标识。
# `token_type_ids` 是一个与输入序列等长的向量，其长度与输入序列中的 token 数目相同。
# 在处理句子对任务时，如果一个 token 属于第一个句子，那么它的 `token_type_ids` 就会被标记为 0；
# 如果它属于第二个句子，那么它的 `token_type_ids` 就会被标记为 1。(如果属于第三个句子，就被标记为2) 这样，模型就可以根据 `token_type_ids` 来区分不同句子的信息。
# Q: 如果有三个分段，token_type_ids显示什么？
# token_type_ids = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2]
# Q: tokenizer.convert_ids_to_tokens是干啥的？
# 将标记 ID 列表 `[101, 2023, 2003, ...]` 转换为对应的文本标记列表
# `['[CLS]', 'what', 'is', 'a', 'good', 'answer', 'to', 'the', 'question', '[SEP]']`。
inputs = tokenizer("This is the first sentence.", "This is the second one.")
print("inputs: ", inputs)
print(tokenizer.convert_ids_to_tokens(inputs["input_ids"]))


def tokenize_function(example):
    # 对句子进行截断以适应模型的输入长度限制
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


# Q: Hugging Face Datasets 库中的 map 函数是干啥的？
# 对数据集中的每个样本应用指定的函数，并返回一个新的数据集
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
print("tokenized datasets: ", tokenized_datasets)

samples = tokenized_datasets["train"][:8]  # 从其训练集中获取前8个样本
# 从字典 samples 中移除键为 "idx"、"sentence1" 和 "sentence2" 的键值对，并创建一个新的字典，只包含其他键值对。新的字典中只保留了不在指定列表中的键值对。
samples = {k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]}
print([len(x) for x in samples["input_ids"]])

# 3.2.2 Dynamic padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
batch = data_collator(samples)
print({k: v.shape for k, v in batch.items()})






