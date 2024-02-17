from datasets import load_dataset


raw_datasets = load_dataset("kde4", lang1="en", lang2="fr")
print(raw_datasets)

split_datasets = raw_datasets["train"].train_test_split(train_size=0.9, seed=20)
print(split_datasets)

split_datasets["validation"] = split_datasets.pop("test")

print(split_datasets["train"][1]["translation"])




