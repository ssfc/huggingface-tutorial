# Huggingface tutorial

[简介 - 拥抱脸NLP课程 (huggingface.co)](https://huggingface.co/learn/nlp-course/chapter0/1?fw=pt)

# 0. Setup

# 1. Transfomer Models

## 1.1 Introduction

Comment: 吾人把重点放在如何使用transformer处理NLP任务，最后几章处理CV任务的不用看。

## 1.2 Natural Language Processing

## 1.3 Transformers, what can they do?

Section 1: sentiment analysis

Section 2: text classification

Section 3: text generation

Section 4: fill-mask

Section 5: Named entity recognition

Section 6: Question answering

Section 7: summarization

Section 8: translation

## 1.4 How do Transformers work?

A bit of Transformer history

- GPT-like (also called *auto-regressive* Transformer models)
- BERT-like (also called *auto-encoding* Transformer models)
- BART/T5-like (also called *sequence-to-sequence* Transformer models)

任务的一个例子是在阅读前 *n 个*单词后预测句子中的下一个单词。这称为*因果语言建模*，因为输出取决于过去和现在的输入，而不是未来的输入。(外插)

另一个例子是掩码语言建模，其中模型预测句子中的*掩码*单词。（内插）

fine-tune属于迁移学习。