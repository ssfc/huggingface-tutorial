from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

raw_datasets = load_dataset("glue", "mrpc")
print("raw_datasets", raw_datasets)
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
print("tokenized_datasets:", tokenized_datasets)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

samples = tokenized_datasets["train"][:8]
samples = {k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]}
print([len(x) for x in samples["input_ids"]])


