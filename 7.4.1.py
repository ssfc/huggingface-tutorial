from datasets import load_dataset


raw_datasets = load_dataset("kde4", lang1="en", lang2="fr")
print(raw_datasets)

split_datasets = raw_datasets["train"].train_test_split(train_size=0.9, seed=20)
print(split_datasets)

split_datasets["validation"] = split_datasets.pop("test")

print(split_datasets["train"][1]["translation"])


from transformers import pipeline


model_checkpoint = "Helsinki-NLP/opus-mt-en-fr"
translator = pipeline("translation", model=model_checkpoint)
result = translator("Default to expanded threads")
print(result)


from transformers import AutoTokenizer


model_checkpoint = "Helsinki-NLP/opus-mt-en-fr"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, return_tensors="pt")

en_sentence = split_datasets["train"][1]["translation"]["en"]
fr_sentence = split_datasets["train"][1]["translation"]["fr"]

inputs = tokenizer(en_sentence, text_target=fr_sentence)
print(inputs)

wrong_targets = tokenizer(fr_sentence)
print(tokenizer.convert_ids_to_tokens(wrong_targets["input_ids"]))
print(tokenizer.convert_ids_to_tokens(inputs["labels"]))






