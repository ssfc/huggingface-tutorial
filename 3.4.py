from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AdamW
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import get_scheduler
import torch


raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(example):
    # # `truncation`: 是否进行截断，如果为 `True`，会根据最长的序列进行截断。
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)   # (required): 要使用的 tokenizer，用于将文本转换为模型可接受的输入格式。


# 3.4.1 Prepare for training
tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")
print("tokenized_datasets[train].column_names:", tokenized_datasets["train"].column_names)


train_dataloader = DataLoader(
    tokenized_datasets["train"],  # 要加载的数据集。通常是一个 `torch.utils.data.Dataset` 类的实例。
    shuffle=True,  # 是否在每个 epoch 开始时对数据进行洗牌（随机排序）。
    batch_size=8,  # 每个批次中的样本数量。
    collate_fn=data_collator  # 用于将多个样本组合成一个批次的函数。通常用于处理不同长度的序列数据。
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"],  # 要加载的数据集。通常是一个 `torch.utils.data.Dataset` 类的实例。
    batch_size=8,  # 每个批次中的样本数量。
    collate_fn=data_collator  # # 用于将多个样本组合成一个批次的函数。通常用于处理不同长度的序列数据。
)

for batch in train_dataloader:
    break
print({k: v.shape for k, v in batch.items()})

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
outputs = model(**batch)
print("outputs.loss:", outputs.loss)
print("outputs.logits.shape:", outputs.logits.shape)

optimizer = AdamW(model.parameters(),  # 一个包含要优化的参数的可迭代对象，通常是模型的参数列表。
                  lr=5e-5)  # 学习率（Learning Rate），控制每次参数更新的步长。

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",  # 要获取的学习率调度器的名称或类型
    optimizer=optimizer,  # 要应用学习率调度器的优化器。
    num_warmup_steps=0,  # 预热步数，即在此步数之前进行线性或余弦预热。
    num_training_steps=num_training_steps,  # 总的训练步数。
)
print("num_training_steps:", num_training_steps)

# 3.4.2 The training loop
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
print("device:", device)

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)



