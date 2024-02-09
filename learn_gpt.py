from transformers import pipeline
import time


start_time = time.time()
# 模型链接: https://huggingface.co/distilbert/distilgpt2
generator = pipeline("text-generation", model="distilgpt2")
# device default: 22.94s
# device 0: 21.34s
# device -1: 22.89s.
# device cuda: 21.39s.

# generator = pipeline("text-generation", model="gpt2")
# generator = pipeline("text-generation", model="gpt2-medium")
# generator = pipeline("text-generation", model="gpt2-large")
# generator = pipeline("text-generation", model="gpt2-xl", device=0)

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




