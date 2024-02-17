from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq


model_checkpoint = "Helsinki-NLP/opus-mt-en-fr"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, return_tensors="pt")
# 吾人认为，从逻辑上讲先分词再模型, 所以tokenizer应该放在model的前面
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)  # 之前还看到过AutoModelForCausalLM, AutoModelForMaskedLM, AutoModelForSequenceClassification

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


def preprocess_function(examples):
    max_length = 128
    inputs = [ex["en"] for ex in examples["translation"]]
    targets = [ex["fr"] for ex in examples["translation"]]
    model_inputs = tokenizer(
        inputs, text_target=targets, max_length=max_length, truncation=True
    )
    return model_inputs


raw_datasets = load_dataset("kde4", lang1="en", lang2="fr")
print(raw_datasets)

split_datasets = raw_datasets["train"].train_test_split(train_size=0.9, seed=20)


tokenized_datasets = split_datasets.map(
    preprocess_function,
    batched=True,
    remove_columns=split_datasets["train"].column_names,
)


batch = data_collator([tokenized_datasets["train"][i] for i in range(1, 3)])
# print(batch)
print(batch.keys())
print(batch["labels"])

for i in range(1, 3):
    print(tokenized_datasets["train"][i]["labels"])

import evaluate

metric = evaluate.load("sacrebleu")

predictions = [
    "This plugin lets you translate web pages between several languages automatically."
]
references = [
    [
        "This plugin allows you to automatically translate web pages between several languages."
    ]
]

print(metric.compute(predictions=predictions, references=references))





