# 奇了怪了, 报错。

from datasets import load_dataset

raw_datasets = load_dataset("glue", "mrpc")
print(raw_datasets)

raw_train_dataset = raw_datasets["train"]
print(raw_train_dataset[0])
print(raw_train_dataset.features)  # 这是一个鉴别两个句子是否同义的数据集

# ✏️ 试试看！查看训练集的元素 15 和验证集的元素 87。他们的标签是什么？
print("Element 15 in train set", raw_train_dataset[15])

raw_validation_set = raw_datasets["validation"]
print("Element 87 in validation set", raw_train_dataset[15])


from transformers import AutoTokenizer

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenized_sentences_1 = tokenizer(raw_datasets["train"]["sentence1"])
tokenized_sentences_2 = tokenizer(raw_datasets["train"]["sentence2"])

inputs = tokenizer("This is the first sentence.", "This is the second one.")
print("inputs:", inputs)




