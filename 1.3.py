from transformers import pipeline
# Q: huggingface pipline是干啥的？
# Hugging Face Transformers 库中的 `pipeline` 是一个方便的工具，用于通过预训练模型快速实现各种自然语言处理（NLP）任务，
# 包括文本分类、命名实体识别、问答等。`pipeline` 提供了一种简单的方式，可以在几行代码内加载预训练模型并对文本进行处理，无需编写复杂的模型代码。
# `pipeline` 函数的主要作用是自动加载指定任务的预训练模型和相关配置，然后对输入文本进行预测，输出相应的任务结果。
# 你只需要指定任务名称，`pipeline` 会自动选择和加载与任务相对应的预训练模型，并返回一个可以直接用于预测的函数。

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
result = generator("I suck a girl's nipples and ", max_length=60,)
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

# Section 6: Question answering
'''
question_answerer = pipeline("question-answering")
result = question_answerer(
    question="Where do I work?",
    context="My name is Sylvain and I work at Hugging Face in Brooklyn",
)
print(result)
'''

# Section 7: summarization
'''
summarizer = pipeline("summarization")
result = summarizer(
    """
    America has changed dramatically during recent years. Not only has the number of 
    graduates in traditional engineering disciplines such as mechanical, civil, 
    electrical, chemical, and aeronautical engineering declined, but in most of 
    the premier American universities engineering curricula now concentrate on 
    and encourage largely the study of engineering science. As a result, there 
    are declining offerings in engineering subjects dealing with infrastructure, 
    the environment, and related issues, and greater concentration on high 
    technology subjects, largely supporting increasingly complex scientific 
    developments. While the latter is important, it should not be at the expense 
    of more traditional engineering.

    Rapidly developing economies such as China and India, as well as other 
    industrial countries in Europe and Asia, continue to encourage and advance 
    the teaching of engineering. Both China and India, respectively, graduate 
    six and eight times as many traditional engineers as does the United States. 
    Other industrial countries at minimum maintain their output, while America 
    suffers an increasingly serious decline in the number of engineering graduates 
    and a lack of well-educated engineers.
"""
)
print(result)
'''

# Section 8: translation
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
result = translator("Ce cours est produit par Hugging Face.")
print(result)


