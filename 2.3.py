from transformers import BertConfig, BertModel

# Section 1: Creating a Transformer
# Building the config
'''
config = BertConfig()

# Building the model from the config
model = BertModel(config)

print(config)
'''
# Section 2: Different loading methods

# Section 3:
model = BertModel.from_pretrained("bert-base-cased")
print(model)
model.save_pretrained("directory_on_my_computer")





