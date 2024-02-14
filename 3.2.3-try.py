# 试试看！在 GLUE SST-2 数据集上复制预处理。它有点不同，因为它是由单个句子而不是成对组成的，但我们所做的其余部分应该看起来相同。

from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding


def tokenize_function(example):
    return tokenizer(example["sentence"], truncation=True)


checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

raw_datasets = load_dataset("glue", "sst2")
print("raw_datasets", raw_datasets)
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
print("tokenized_datasets:", tokenized_datasets)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

samples = tokenized_datasets["train"][:8]
samples = {k: v for k, v in samples.items() if k not in ["idx", "sentence"]}
print([len(x) for x in samples["input_ids"]])

batch = data_collator(samples)
print({k: v.shape for k, v in batch.items()})

