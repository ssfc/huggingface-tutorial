from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding

raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(example):
    # # `truncation`: 是否进行截断，如果为 `True`，会根据最长的序列进行截断。
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# 3.4.1 Prepare for training
tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")
print("tokenized_datasets[train].column_names:", tokenized_datasets["train"].column_names)


train_dataloader = DataLoader(
    tokenized_datasets["train"],  # 要加载的数据集。通常是一个 `torch.utils.data.Dataset` 类的实例。
    shuffle=True,  # 是否在每个 epoch 开始时对数据进行洗牌（随机排序）。
    batch_size=8,  # 每个批次中的样本数量。
    collate_fn=data_collator  # 用于将多个样本组合成一个批次的函数。通常用于处理不同长度的序列数据。
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"],  # 要加载的数据集。通常是一个 `torch.utils.data.Dataset` 类的实例。
    batch_size=8,  # 每个批次中的样本数量。
    collate_fn=data_collator  # # 用于将多个样本组合成一个批次的函数。通常用于处理不同长度的序列数据。
)



