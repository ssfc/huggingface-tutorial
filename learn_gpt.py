from transformers import pipeline
import time


start_time = time.time()
# 模型链接: https://huggingface.co/distilbert/distilgpt2
# generator = pipeline("text-generation", model="distilgpt2", device="cuda")
# device cpu: 22.94s
# device 4090: 21.40s


# generator = pipeline("text-generation", model="gpt2", device="cuda")
# device cpu: 24.30s
# device 4090: 21.68s

# generator = pipeline("text-generation", model="gpt2-medium", device="cuda")
# device cpu: 32.33s
# device 1660s: 23.55s
# device 4090: 23.33s

generator = pipeline("text-generation", model="gpt2-large", device="cuda")
# device cpu: 45.14s
# device 1660s: 25.05s
# device 4090: 26.03s

# generator = pipeline("text-generation", model="gpt2-xl", device="cuda")
# device cpu: 69.23s
# device 4090: 30.95s

result = generator(
    # "In this course, we will teach you how to",
    "I took off my wife's clothes, rub her nipples",
    # "Jack cum in Lucy's cunt",
    max_length=100,
    num_return_sequences=2,
)

print(result[0])
print(result[1])

print("Pipeline device:", generator.device)

end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed Time:", elapsed_time, "seconds")




