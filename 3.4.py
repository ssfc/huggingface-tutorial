from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AdamW
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import get_scheduler
import evaluate
import torch


##############################################################################################################
# Step 1: prepare dataset;
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

##############################################################################################################
# Step 2: Design Model;
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
outputs = model(**batch)
print("outputs.loss:", outputs.loss)
print("outputs.logits.shape:", outputs.logits.shape)

# ################################################################################################################
# Step 3: Construct loss and optimizer
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
################################################################################################################
# Step 4: Training loop
for epoch in range(num_epochs):
    for batch in train_dataloader:
        # (1) Prepare data
        # .items() 是 Python 字典（Dictionary）对象的一个方法，用于获取字典中所有的键值对（键和对应的值）。
        # 只移动值 `v`到GPU 而不移动键 `k`，是因为键（key）通常是标识数据的标签，它们是元数据而不是需要进行计算的实际数据。
        # 在这种情况下，通常只需要将值（例如张量）移动到指定的设备上，以便在该设备上进行模型计算。
        # 键（标签）并不需要在设备之间移动，因为它们只是用于标识数据的标签，不直接参与模型的计算。键在字典中起到索引作用，用于访问对应的值。
        # 所以，一般情况下，只需要将值移动到指定设备上，以便在该设备上进行实际计算，而键通常不需要进行设备间的移动。
        batch = {k: v.to(device) for k, v in batch.items()}
        # (2) Forward
        outputs = model(**batch)
        loss = outputs.loss
        # (3) Backward
        optimizer.zero_grad()  # 将模型参数的梯度清零，以便为下一次迭代做准备。
        loss.backward()  # 计算损失函数的梯度
        # (4) Update
        optimizer.step()  # 更新模型参数，使其朝着梯度下降方向移动
        lr_scheduler.step()  # 调用该方法来执行学习率的调整，通常在每个训练周期结束时调用。
        progress_bar.update(1)

# 3.4.3 The evaluation loop
metric = evaluate.load("glue", "mrpc")
model.eval()  # 将模型切换到评估模式，这会影响模型中一些层（如 Dropout 层），使其在评估时表现更稳定。
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

print("metric.compute:", metric.compute())

# 3.4.4 Supercharge your training loop with Accelerate
accelerator = Accelerator()

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
optimizer = AdamW(model.parameters(),  # 一个包含要优化的参数的可迭代对象，通常是模型的参数列表。
                  lr=3e-5)  # 学习率（Learning Rate），控制每次参数更新的步长。

(train_dl,  # 经过加速器处理后的训练数据加载器。
 eval_dl,  # 经过加速器处理后的评估数据加载器。
 model_accelerated,  # 经过加速器处理后的模型。
 optimizer) = (  # 经过加速器处理后的优化器。
    accelerator.prepare(
        train_dataloader,  # 训练数据加载器，用于迭代训练数据批次的迭代器。
        eval_dataloader,  # 评估数据加载器，用于迭代评估数据批次的迭代器。
        model,  # 要训练的深度学习模型。
        optimizer  # 用于优化模型的优化器，例如 Adam、SGD 等。
))

num_epochs = 3
num_training_steps = num_epochs * len(train_dl)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

progress_bar = tqdm(range(num_training_steps))

model_accelerated.train()
for epoch in range(num_epochs):
    for batch in train_dl:
        outputs = model_accelerated(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

