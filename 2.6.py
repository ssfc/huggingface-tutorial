from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# 标记单个序列：
sequence = "I've been waiting for a HuggingFace course my whole life."
model_inputs = tokenizer(sequence)
print("mark one sentence: ", model_inputs)

# 一次处理多个序列，API 没有变化：
sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]
model_inputs = tokenizer(sequences)
print("mark two sentences: ", model_inputs)

# 根据几个目标进行填充：
# Will pad the sequences up to the maximum sequence length
model_inputs = tokenizer(sequences, padding="longest")
print(model_inputs)

# Will pad the sequences up to the model max length
# (512 for BERT or DistilBERT)
model_inputs = tokenizer(sequences, padding="max_length")
print(model_inputs)

# Will pad the sequences up to the specified max length
model_inputs = tokenizer(sequences, padding="max_length", max_length=8)
print(model_inputs)

# 它还可以截断序列：
sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

# Will truncate the sequences that are longer than the model max length
# (512 for BERT or DistilBERT)
model_inputs = tokenizer(sequences, truncation=True)
print("truncate more than 512: ", model_inputs)

# Will truncate the sequences that are longer than the specified max length
model_inputs = tokenizer(sequences, max_length=8, truncation=True)
print("truncate more than 8: ", model_inputs)
# 这下序列长度还真小于等于8了。

# 处理到特定框架张量的转换，然后可以直接将其发送到模型
sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]
# Returns PyTorch tensors
model_inputs = tokenizer(sequences, padding=True, return_tensors="pt")
print("pytorch tensors: ", model_inputs)
# Returns TensorFlow tensors
# model_inputs = tokenizer(sequences, padding=True, return_tensors="tf")
# print(model_inputs)
# Returns NumPy arrays
model_inputs = tokenizer(sequences, padding=True, return_tensors="np")
print("numpy arrays: ", model_inputs)

# 2.6.1 Special tokens
sequence = "I've been waiting for a HuggingFace course my whole life."

model_inputs = tokenizer(sequence)
print(model_inputs["input_ids"])

tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)

print(tokenizer.decode(model_inputs["input_ids"]))
print(tokenizer.decode(ids))

# 2.6.2 Wrapping up: From tokenizer to model




