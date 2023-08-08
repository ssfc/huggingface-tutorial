# Q: huggingface中的BertConfig是什么？
# 在 Hugging Face Transformers 库中，`BertConfig` 是用于配置 BERT 模型的类。
# BERT（Bidirectional Encoder Representations from Transformers）是一种流行的预训练模型，在自然语言处理（NLP）任务中取得了很好的效果。
# `BertConfig` 类用于定义和设置 BERT 模型的超参数和配置选项。
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








