from transformers import pipeline

# Section 1: sentiment analysis
'''
classifier = pipeline("sentiment-analysis")
result = classifier("I've been waiting for a HuggingFace course my whole life.")
print("send one sentence: ", result)

result = classifier(["I've been waiting for a HuggingFace course my whole life.", "I hate this so much!"])
print("send two sentences: ", result)
'''

# Section 2: text classification
'''
classifier = pipeline("zero-shot-classification")
result = classifier("This is a course about the Transformers library", candidate_labels=["education", "politics", "business"],)
print(result)
'''

# Section 3: text generation
'''
generator = pipeline("text-generation")
result = generator("In this course, we will teach you how to")
print(result)
'''
'''
generator = pipeline("text-generation", model="distilgpt2")
result = generator(
    "In this course, we will teach you how to",
    max_length=30,
    num_return_sequences=2,
)
print(result)
'''

# Section 4: fill-mask
'''
unmasker = pipeline("fill-mask")
result = unmasker("This course will teach you all about <mask> models.", top_k=2)
print(result)
'''

# Section 5: Named entity recognition
'''
ner = pipeline("ner", grouped_entities=True)
result = ner("My name is Sylvain and I work at Hugging Face in Brooklyn.")
print(result)
'''

# Section 5: Question answering
question_answerer = pipeline("question-answering")
result = question_answerer(
    question="Where do I work?",
    context="My name is Sylvain and I work at Hugging Face in Brooklyn",
)
print(result)



