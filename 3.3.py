from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
token_ids = [101, 2023, 2003, 1037, 2232, 1999, 1996, 2190, 2226, 102]

tokens = tokenizer.convert_ids_to_tokens(token_ids)
print(tokens)
