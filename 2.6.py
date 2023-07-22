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



