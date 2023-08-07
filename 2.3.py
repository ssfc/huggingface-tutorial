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








