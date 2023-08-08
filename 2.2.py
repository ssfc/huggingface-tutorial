# https://huggingface.co/learn/nlp-course/chapter2/2?fw=pt
# Comment: 将端到端pipeline分解为各个步骤。

import torch
# Q: huggingface AutoModel是干啥的？
# `AutoModel` 是 Hugging Face Transformers 库中的一个类，它是一个方便的工具，用于根据指定的模型名称或 checkpoint 来自动选择和加载相应的预训练模型。
# 在深度学习中，使用预训练模型是一种常见的方式，可以为各种自然语言处理（NLP）任务提供强大的基础。
# 然而，Hugging Face Transformers 库提供了许多不同类型的预训练模型（例如 BERT、GPT、RoBERTa 等），而每个模型可能有多个不同的变体和配置。
# 为了方便用户加载和使用不同模型，`AutoModel` 类允许用户通过一个简单的 API 来加载预训练模型，而无需手动选择特定的模型或配置。
# `AutoModel` 的主要作用是根据给定的模型名称或 checkpoint 来自动选择和加载对应的预训练模型。这样，你只需要指定模型名称或 checkpoint，`AutoModel` 就会自动选择和加载与之对应的预训练模型。
from transformers import AutoModel
# Q: huggingface AutoModelForSequenceClassification是干啥的？
# `AutoModelForSequenceClassification` 是 Hugging Face Transformers 库中的一个类，它是用于序列分类任务的预训练模型的自动加载工具。
# 在自然语言处理（NLP）中，序列分类任务是指将输入的文本序列（例如句子、段落）分类到预定义的类别或标签中的任务。
# `AutoModelForSequenceClassification` 类的作用是根据指定的模型名称或 checkpoint 来自动选择和加载适用于序列分类任务的预训练模型。
# 它是 `AutoModel` 类的一个子类，在加载模型的同时，它会自动设置模型的输出层，以适应序列分类任务的特定需求。
from transformers import AutoModelForSequenceClassification
# Q: huggingface AutoTokenizer是干啥的？
# `AutoTokenizer` 是 Hugging Face Transformers 库中的一个类，它是一个自动加载预训练模型的分词器（Tokenizer）的工具类。
# 在自然语言处理（NLP）中，分词器用于将输入文本（句子、段落等）拆分成单词或子词的序列，以便机器学习模型能够处理和理解文本。
# `AutoTokenizer` 的主要作用是根据给定的模型名称或 checkpoint 来自动选择和加载对应的预训练模型的分词器。
# 这个类是 `AutoModel` 和 `AutoModelForSequenceClassification` 类的分词器版本，它使得在加载预训练模型和分词器时变得非常方便和灵活。
# 使用 `AutoTokenizer` 有以下几个优点：
# 1. 自动选择模型：无需手动指定模型名称，`AutoTokenizer` 会根据提供的模型名称自动选择和加载对应的分词器。
# 2. 多种模型支持：`AutoTokenizer` 支持加载各种不同的预训练模型的分词器，如 BERT、GPT、RoBERTa 等。
# 3. 方便的代码迁移：如果你在代码中使用了 `AutoTokenizer` 来加载分词器，当你改变模型时，只需更改模型名称，而不需要修改其他代码。
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
# padding是自动填充，truncation=true是超过最大长度截断，return_tensors是返回pytorch格式
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
print(inputs)

# 奇怪的是，怎么输出向量的维数和输入单词的数量不一样？这因为是一些单词(比如huggingface)会被分词器分成两个

# Section 2: Going through the model
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModel.from_pretrained(checkpoint)
# print(model)

# Section 3: A high-dimensional vector?
# Q: model(**inputs)为什么inputs前面有**?
# 在 Python 中，双星号（`**`）用于解包字典（或关键字参数）并将其作为关键字参数传递给函数。
# 在 `model(**inputs)` 中，`**inputs` 表示将 `inputs` 这个字典中的键值对解包，并将其中的键作为关键字参数名，将对应的值作为关键字参数值传递给函数 `model`。
# 在 Hugging Face Transformers 库中，`model` 是一个预训练模型的实例，可以接受一组特定的输入参数，例如 `input_ids`、`attention_mask` 等。
# 而 `inputs` 是一个字典，其中包含了这些输入参数，键为参数名，值为相应的张量数据。
# 通过使用 `**inputs`，我们可以将字典 `inputs` 中的键值对解包，并将其传递给 `model` 函数作为关键字参数。
# 这样做的好处是，可以根据需要轻松地添加或修改输入参数，而不需要显式地逐个传递每个参数。
outputs = model(**inputs)
print(outputs.last_hidden_state.shape)

# Section 4: Model heads: Making sense out of numbers
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model(**inputs)
print(outputs.logits.shape)

# Section 5: Postprocessing the output
print(outputs.logits)

predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)

print(model.config.id2label)








