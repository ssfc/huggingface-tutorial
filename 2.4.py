from transformers import AutoTokenizer
from transformers import BertTokenizer

# Section 1: Word-based
'''
tokenized_text = "Jim Henson was a puppeteer".split()
print(tokenized_text)
'''
# Section 2: Character-based

# Section 3: Subword tokenization

# Section 4: And more!

# Section 5: Loading and saving
# tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
# print(tokenizer)

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
print(tokenizer)

result = tokenizer("Using a Transformer network is simple")
print(result)
tokenizer.save_pretrained("directory_on_my_computer")

# Section 6: Encoding

# Section 7: Tokenization
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

# sequence = "Using a Transformer network is simple"
sequence = "*146*171*5*171*1*38*171*23*38*171*23*38*171*23*17*5*"
tokens = tokenizer.tokenize(sequence)

print("tokens: ", tokens)

# Section 8: From tokens to input IDs
ids = tokenizer.convert_tokens_to_ids(tokens)

print(ids)

raw_inputs = "Using a Transformer network is simple"
inputs = tokenizer(raw_inputs)
print(inputs)
# Comment: 输出是不一样的，头部尾部多了俩元素

# 2.4.9 Decoding
'''
decoded_string = tokenizer.decode([7993, 170, 11303, 1200, 2443, 1110, 3014])
print(decoded_string)
'''






