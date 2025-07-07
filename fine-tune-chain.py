from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, Dataset
import torch

# 1. 加载 tokenizer 和模型（GPT2）
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # gpt2 默认没有pad_token

model = AutoModelForCausalLM.from_pretrained(model_name)

# 2. 构造小型CoT数据集
data = [
    {"question": "What is 2 + 3?",
     "cot": "To solve 2 + 3, we add 2 and 3. The result is 5."},
    {"question": "If you have 10 apples and give away 4, how many are left?",
     "cot": "You start with 10 apples. Giving away 4 leaves you with 6 apples."},
    {"question": "What is 7 times 6?",
     "cot": "7 times 6 is the same as adding 7 six times. 7+7+7+7+7+7 = 42."}
]

dataset = Dataset.from_list(data)

# 3. 预处理函数
def preprocess(example):
    prompt = f"Question: {example['question']}\nAnswer: "
    target = example["cot"]
    input_text = prompt + target
    tokenized = tokenizer(input_text, truncation=True, padding="max_length", max_length=128)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_dataset = dataset.map(preprocess)

# 4. 微调配置
training_args = TrainingArguments(
    output_dir="./cot-gpt2",
    per_device_train_batch_size=2,
    num_train_epochs=5,
    logging_steps=1,
    save_steps=5,
    warmup_steps=5,
    weight_decay=0.01,
    logging_dir="./logs",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer
)

# 5. 微调
trainer.train()


model.eval()
prompt = "Question: If a pen costs $2 and you buy 3 pens, how much do you spend?\nAnswer: "
inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(**inputs, max_length=100, do_sample=False)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
