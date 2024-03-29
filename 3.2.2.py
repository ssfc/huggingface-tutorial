# 奇了怪了, 报错。

from datasets import load_dataset

raw_datasets = load_dataset("glue", "mrpc")
print(raw_datasets)

raw_train_dataset = raw_datasets["train"]
print(raw_train_dataset[0])
print(raw_train_dataset.features)  # 这是一个鉴别两个句子是否同义的数据集

# ✏️ 试试看！查看训练集的元素 15 和验证集的元素 87。他们的标签是什么？
# print("Element 15 in train set", raw_train_dataset[15])

raw_validation_set = raw_datasets["validation"]
# print("Element 87 in validation set", raw_train_dataset[15])


from transformers import AutoTokenizer

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenized_sentences_1 = tokenizer(raw_datasets["train"]["sentence1"])
tokenized_sentences_2 = tokenizer(raw_datasets["train"]["sentence2"])

tokenized_ids = tokenizer("This is the first sentence.", "This is the second one.")
print("tokenized_inputs:", tokenized_ids)

# ✏️ 试试看！采用训练集的元素 15，将两个句子分别标记化，并成对标记。这两个结果之间有什么区别？
# token_type_ids前面是0后面是1.
tokenized_train15_1 = tokenizer(raw_datasets["train"][15]["sentence1"])
tokenized_train15_2 = tokenizer(raw_datasets["train"][15]["sentence2"])
tokenized_train15_12 = tokenizer(raw_datasets["train"][15]["sentence1"], raw_datasets["train"][15]["sentence2"])
print("tokenized_train15_1:", tokenized_train15_1)
print("tokenized_train15_2:", tokenized_train15_2)
print("tokenized_train15_12:", tokenized_train15_12)

tokenized_words = tokenizer.convert_ids_to_tokens(tokenized_ids["input_ids"])
print("tokenized_words:", tokenized_words)


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
print("tokenized_datasets:", tokenized_datasets)








