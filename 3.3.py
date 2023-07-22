from transformers import AutoTokenizer

# 指定模型名称或 checkpoint
checkpoint = "bert-base-uncased"

# 自动选择和加载对应的分词器
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# 使用分词器对输入文本进行分词
text = "Hello, how are you?"
tokens = tokenizer(text)

# 输出分词结果
print(tokens)
# {'input_ids': [101, 7592, 1010, 2129, 2024, 2017, 1029, 102], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0],
# 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1]}



