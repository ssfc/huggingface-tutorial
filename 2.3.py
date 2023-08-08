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








