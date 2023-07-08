from transformers import AutoModel
from transformers import AutoTokenizer
from transformers import pipeline


'''
classifier = pipeline("sentiment-analysis")
result = classifier(
    [
        "I've been waiting for a HuggingFace course my whole life.",
        "I hate this so much!",
    ]
)
print(result)
'''

# Section 1: Preprocessing with a tokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
print(inputs)

# 奇怪的是，怎么输出向量的维数和输入单词的数量不一样？

# Section 2: Going through the model
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModel.from_pretrained(checkpoint)
print(model)

outputs = model(**inputs)
print(outputs.last_hidden_state.shape)









