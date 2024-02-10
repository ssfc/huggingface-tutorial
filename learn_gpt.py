from transformers import pipeline
import time


start_time = time.time()
# 模型链接: https://huggingface.co/distilbert/distilgpt2
# generator = pipeline("text-generation", model="distilgpt2")
# device 4090 default: 22.94s
# device 1660s default: 23.46
# device 0: 21.34s
# device -1: 22.89s.
# device cuda: 21.39s.
# device cpu: 22.88s

generator = pipeline("text-generation", model="gpt2")
# device 4090 default: 24.30s
# device 1660s default: 24.71s

# generator = pipeline("text-generation", model="gpt2-medium")
# device default: 32.33s

# generator = pipeline("text-generation", model="gpt2-large")
# device default: 45.14s

# generator = pipeline("text-generation", model="gpt2-xl")
# device default: 69.23s

result = generator(
    # "In this course, we will teach you how to",
    "I took off my wife's clothes, rub her nipples",
    # "Jack cum in Lucy's cunt",
    max_length=100,
    num_return_sequences=2,
)

print(result[0])
print(result[1])

end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed Time:", elapsed_time, "seconds")




