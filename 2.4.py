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
'''
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
print(tokenizer)

result = tokenizer("Using a Transformer network is simple")
print(result)
tokenizer.save_pretrained("directory_on_my_computer")
'''
# Section 6: Encoding

# Section 7: Tokenization
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

sequence = "Using a Transformer network is simple"
tokens = tokenizer.tokenize(sequence)

print(tokens)

# Section 8: From tokens to input IDs
ids = tokenizer.convert_tokens_to_ids(tokens)

print(ids)






