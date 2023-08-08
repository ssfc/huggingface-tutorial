# Q: huggingface中的BertConfig是什么？
# 在 Hugging Face Transformers 库中，`BertConfig` 是用于配置 BERT 模型的类。
# BERT（Bidirectional Encoder Representations from Transformers）是一种流行的预训练模型，在自然语言处理（NLP）任务中取得了很好的效果。
# `BertConfig` 类用于定义和设置 BERT 模型的超参数和配置选项。
# Q: huggingface中的BertModel是什么？
# 在 Hugging Face Transformers 库中，`BertModel` 是 BERT 模型的基本实现类。
# BERT（Bidirectional Encoder Representations from Transformers）是一种预训练的自然语言处理（NLP）模型，
# 由 Google 在 2018 年提出，通过双向编码和自注意力机制来学习文本的上下文表示。
# `BertModel` 是 BERT 模型的基础类，它由多个堆叠的 Transformer 编码器组成，每个编码器由多层自注意力和前馈神经网络组成。
# 这些编码器通过堆叠来形成一个深层的神经网络，可以用于处理不同的 NLP 任务，如文本分类、序列标注、问答等。
# 在使用 `BertModel` 之前，通常需要先加载一个预训练的 BERT 模型，可以使用 `from_pretrained()` 方法从 Hugging Face 模型库中加载预训练的权重。
# 然后，你可以通过 `BertModel` 类来进行文本编码、特征提取等操作。
from transformers import BertConfig, BertModel
import torch

# Section 1: Creating a Transformer
# Building the config
'''
config = BertConfig()

# Building the model from the config
model = BertModel(config)

print(config)
'''
# Section 2: Different loading methods

# Section 3: Saving methods
# Q: bert-base-uncased是什么模型？
# `bert-base-uncased` 是 Hugging Face Transformers 库中预训练的 BERT 模型之一。
# BERT（Bidirectional Encoder Representations from Transformers）是由 Google 在 2018 年提出的一种基于 Transformer 架构的预训练自然语言处理模型，
# 以无监督的方式从大规模文本语料中学习文本的上下文表示。
# 在 Hugging Face Transformers 库中，BERT 模型有多个不同的预训练版本，其中 `bert-base-uncased` 是其中之一。
# `uncased` 表示该模型使用的是小写字母形式的文本，即在预训练时**将所有文本转换为小写形式，不区分大小写**。例如，"Hello" 和 "hello" 在模型的输入中被视为相同的单词。
model = BertModel.from_pretrained("bert-base-cased")
print(model)
# model.save_pretrained("directory_on_my_computer")

# Section 4: Using a Transformer model for inference
sequences = ["Hello!", "Cool.", "Nice!"]
encoded_sequences = [
    [101, 7592, 999, 102],
    [101, 4658, 1012, 102],
    [101, 3835, 999, 102],
]
model_inputs = torch.tensor(encoded_sequences)
print("model input:", model_inputs)
output = model(model_inputs)
print("model output", output)








