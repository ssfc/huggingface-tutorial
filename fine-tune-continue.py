from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import torch


# 加载数据
dataset = load_dataset('csv', data_files={'train': 'data.csv'}, delimiter=',')

model_name = 'gpt2'  # 也可以用大一点的gpt2-medium等
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # GPT2没有pad_token默认, 需要指定


def preprocess_function(examples):
    inputs = [str(p) for p in examples['prompt']]
    outputs = [str(o) for o in examples['output']]
    texts = [i + tokenizer.eos_token + o + tokenizer.eos_token for i, o in zip(inputs, outputs)]
    encoding = tokenizer(
        texts, truncation=True, max_length=128, padding="max_length", return_tensors='pt'
    )
    input_ids = encoding['input_ids']
    # 只让输出部分计算loss
    labels = input_ids.clone()
    for idx, (p, o) in enumerate(zip(inputs, outputs)):
        prompt_len = len(tokenizer(p + tokenizer.eos_token)['input_ids'])
        # -100的位置不计loss
        labels[idx][:prompt_len] = -100
    encoding['labels'] = labels
    return encoding

encoded_dataset = dataset.map(preprocess_function, batched=True)


model = AutoModelForCausalLM.from_pretrained(model_name)

training_args = TrainingArguments(
    output_dir='./results',
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=100,
    fp16=True if torch.cuda.is_available() else False,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset['train'],
)

trainer.train()

# 保存
model.save_pretrained('./my_gpt2_story')
tokenizer.save_pretrained('./my_gpt2_story')


from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained('./my_gpt2_story')
tokenizer = AutoTokenizer.from_pretrained('./my_gpt2_story')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()

prompt = "Once upon a time"
input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
with torch.no_grad():
    gen_output = model.generate(
        input_ids,
        max_length=100,
        do_sample=True,
        top_p=0.95,
        num_return_sequences=1
    )
result = tokenizer.decode(gen_output[0], skip_special_tokens=True)
print("生成续写:", result)


