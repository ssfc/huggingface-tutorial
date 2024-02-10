# 奇了怪了, 报错。

from datasets import load_dataset

raw_datasets = load_dataset("glue", "mrpc")
print(raw_datasets)

raw_train_dataset = raw_datasets["train"]
print(raw_train_dataset[0])
print(raw_train_dataset.features)  # 这是一个鉴别两个句子是否同义的数据集


