from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainingArguments
from transformers import AutoModelForSeq2SeqLM # 之前还看到过AutoModelForCausalLM, AutoModelForMaskedLM, AutoModelForSequenceClassification
from transformers import Seq2SeqTrainer

import evaluate
import numpy as np

# 需要Seq2Swq的DataCollator, TrainingArguments, AutoModel, Trainer

model_checkpoint = "Helsinki-NLP/opus-mt-en-fr"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, return_tensors="pt")
# 吾人认为，从逻辑上讲先分词再模型, 所以tokenizer应该放在model的前面
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)  # 之前还看到过AutoModelForCausalLM, AutoModelForMaskedLM, AutoModelForSequenceClassification

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
max_length = 128


def preprocess_function(examples):
    inputs = [ex["en"] for ex in examples["translation"]]
    targets = [ex["fr"] for ex in examples["translation"]]
    model_inputs = tokenizer(inputs, text_target=targets, max_length=max_length, truncation=True)

    return model_inputs


raw_datasets = load_dataset("kde4", lang1="en", lang2="fr")
print(raw_datasets)

split_datasets = raw_datasets["train"].train_test_split(train_size=0.9, seed=20)
split_datasets["validation"] = split_datasets.pop("test")

tokenized_datasets = split_datasets.map(
    preprocess_function,
    batched=True,
    remove_columns=split_datasets["train"].column_names,
)

metric = evaluate.load("sacrebleu")

'''
batch = data_collator([tokenized_datasets["train"][i] for i in range(1, 3)])
# print(batch)
print(batch.keys())
print(batch["labels"])

for i in range(1, 3):
    print(tokenized_datasets["train"][i]["labels"])

predictions = [
    "This plugin lets you translate web pages between several languages automatically."
]
references = [
    [
        "This plugin allows you to automatically translate web pages between several languages."
    ]
]

print(metric.compute(predictions=predictions, references=references))

predictions = ["This This This This"]
references = [
    [
        "This plugin allows you to automatically translate web pages between several languages."
    ]
]
print(metric.compute(predictions=predictions, references=references))

predictions = ["This plugin"]
references = [
    [
        "This plugin allows you to automatically translate web pages between several languages."
    ]
]
print(metric.compute(predictions=predictions, references=references))
'''


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # In case the model returns more than the prediction logits
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100s in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)

    return {"bleu": result["score"]}


args = Seq2SeqTrainingArguments(
    f"marian-finetuned-kde4-en-to-fr",  # output_dir
    evaluation_strategy="no",
    save_strategy="epoch",  # 每个epoch保存一次模型。
    learning_rate=2e-5,
    per_device_train_batch_size=32,  # 每个设备的训练批次大小
    per_device_eval_batch_size=64,  # 每个设备的评估批次大小
    weight_decay=0.01,  # 权重衰减（L2正则化）参数，用于控制模型的过拟合程度
    save_total_limit=3,  # 保存模型文件的总数限制
    num_train_epochs=3,  # 训练的总轮数
    predict_with_generate=True,  # 是否使用生成模式进行预测
    fp16=True,  # 是否使用混合精度训练
    push_to_hub=False,
)

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# 在此之前还有一个进度条，不知道是干啥的，问问GPT4
# 训练前分数
print("score before train:", trainer.evaluate(max_length=max_length))
# score before train: {'eval_loss': 1.6964517831802368, 'eval_bleu': 39.17232664444521, 'eval_runtime': 358.3067, 'eval_samples_per_second': 58.659, 'eval_steps_per_second': 0.918}

trainer.train()

# 训练后分数
print("score after train:", trainer.evaluate(max_length=max_length))
# score after train: {'eval_loss': 0.8559961318969727, 'eval_bleu': 52.83080930043425, 'eval_runtime': 350.4657, 'eval_samples_per_second': 59.972, 'eval_steps_per_second': 0.939, 'epoch': 3.0}



