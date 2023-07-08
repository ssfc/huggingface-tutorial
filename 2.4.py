from transformers import BertTokenizer

# Section 1: Word-based
tokenized_text = "Jim Henson was a puppeteer".split()
print(tokenized_text)

# Section 2: Character-based

# Section 3: Subword tokenization

# Section 4: And more!

# Section 5: Loading and saving
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
print(tokenizer)




