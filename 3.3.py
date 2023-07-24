from transformers import AutoTokenizer, DataCollatorWithPadding

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 数据样本列表，每个样本是一个文本对
data_samples = [
    {"text": "Hello, how are you?", "label": 0},
    {"text": "I am fine, thank you!", "label": 1},
    {"text": "Goodbye!", "label": 2},
]

# 将数据样本组织成一个批次，并进行填充
batch = data_collator(data_samples)

# 输出填充后的批次数据
print(batch)
