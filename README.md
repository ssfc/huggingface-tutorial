# Huggingface tutorial

[简介 - 拥抱脸NLP课程 (huggingface.co)](https://huggingface.co/learn/nlp-course/chapter0/1?fw=pt)

# 0. Setup

# 1. Transfomer Models

## 1.1 Introduction

https://huggingface.co/learn/nlp-course/chapter1/1?fw=pt

Comment: 吾人把重点放在如何使用transformer处理NLP任务，最后几章处理CV任务的不用看。

## 1.2 Natural Language Processing

https://huggingface.co/learn/nlp-course/chapter1/2?fw=pt

## 1.3 Transformers, what can they do?

https://huggingface.co/learn/nlp-course/chapter1/3?fw=pt

Comment: 用代码示范了下列各项任务。

### 1.3.1 sentiment analysis

### 1.3.2 text classification

### 1.3.3 text generation

### 1.3.4 fill-mask

### 1.3.5 Named entity recognition

### 1.3.6 Question answering

### 1.3.7 summarization

### 1.3.8 translation

## 1.4 How do Transformers work?

https://huggingface.co/learn/nlp-course/chapter1/4?fw=pt

Comment: 仅介绍，无代码。

A bit of Transformer history

- GPT-like (also called *auto-regressive* Transformer models)
- BERT-like (also called *auto-encoding* Transformer models)
- BART/T5-like (also called *sequence-to-sequence* Transformer models)

任务的一个例子是在阅读前 *n 个*单词后预测句子中的下一个单词。这称为*因果语言建模*，因为输出取决于过去和现在的输入，而不是未来的输入。(外插)

另一个例子是掩码语言建模，其中模型预测句子中的*掩码*单词。（内插）

fine-tune属于迁移学习。

预训练模型已经在与微调数据集有一些相似之处的数据集上进行训练。

- **仅编码器模型**：适用于需要理解输入的任务，例如句子分类和命名实体识别。
- **仅解码器模型**：适用于文本生成等生成任务。（这个适用于SSFC）
- **编码器-解码器**模型或序列**到序列模型**：适用于需要输入的生成任务，例如翻译或摘要。

在编码器中，注意力层可以使用句子中的所有单词（因为，正如我们刚刚看到的，给定单词的翻译可能取决于句子中它之后和之前的内容）。但是，解码器按顺序工作，并且只能注意它已经翻译的句子中的单词（因此，只能注意当前正在生成的单词之前的单词）。

解码器块中的第一个注意层关注解码器的所有（过去的）输入，但第二个注意层使用编码器的输出。

## 1.5 Encoder models

https://huggingface.co/learn/nlp-course/chapter1/5?fw=pt

Comment: 仅介绍，无代码。

编码器型号仅使用变压器型号的编码器。在每个阶段，注意力层可以访问初始句子中的所有单词。这些模型通常被描述为具有“双向”注意力，并且通常被称为*自动编码模型*。

这些模型的预训练通常围绕着以某种方式破坏给定的句子（例如，通过屏蔽其中的随机单词）并要求模型查找或重建初始句子。

编码器模型最适合需要理解完整句子的任务，例如句子分类、命名实体识别（以及更一般的单词分类）和抽取式问答。

ALBERT
BERT
DistilBERT
ELECTRA
RoBERTa

### Q: 怎样理解编码器模型中的”双向“？

编码器模型是一种基于Transformer的神经网络结构，可以提取文本的特征表示。编码器模型中的“双向”指的是模型可以同时考虑输入序列的左右两边的上下文信息，而不是只考虑一个方向。这样可以增强模型的语言理解能力，捕获更多的语义和句法信息。

编码器模型中的“双向”通常是通过一种预训练任务实现的，叫做**随机遮蔽词语预测**或**Masked语言模型**。这个任务的目标是从输入序列中随机遮住一些词语，然后让模型根据剩下的词语预测被遮住的词语。这样就迫使模型利用两边的上下文信息，而不是只依赖一个方向。

一个典型的编码器模型是**BERT**（Bidirectional Encoder Representations from Transformers），它使用了多层双向Transformer encoder作为文本特征提取器，并且使用了随机遮蔽词语预测和“是否下一句”二分类两个预训练任务。BERT在很多自然语言处理任务上都取得了很好的效果，也有很多变体和改进版本。

如果你想了解更多关于编码器模型中的“双向”的内容，你可以参考以下链接：

- [基于Transformer的双向编码器表示(BERT)——结构和训练 - 知乎](https://zhuanlan.zhihu.com/p/125655365)：这篇文章对BERT的来由、结构和训练方式，以及关键环节的动机和细节，进行了简单的介绍。
- [基于Transformer的双向编码器表示：BERT - 知乎](https://zhuanlan.zhihu.com/p/105601731)：这篇文章对BERT的输入输出表示、预训练任务、微调方法等进行了说明，并给出了一些代码示例。
- [“编码器”模型 - Hugging Face NLP Course](https://huggingface.co/course/zh-CN/chapter1/5?fw=pt)：这个网页是Hugging Face提供的一个NLP课程，其中介绍了编码器模型和自编码模型的概念，并展示了如何使用Hugging Face库来加载和使用BERT等预训练模型。

源: 与必应的对话， 2023/7/8
(1) 基于Transformer的双向编码器表示(BERT)——结构和训练 .... https://zhuanlan.zhihu.com/p/125655365.
(2) 基于Transformer的双向编码器表示：BERT - 知乎. https://zhuanlan.zhihu.com/p/105601731.
(3) “编码器”模型 - Hugging Face NLP Course. https://huggingface.co/course/zh-CN/chapter1/5?fw=pt.

## 1.6 Decoder models

https://huggingface.co/learn/nlp-course/chapter1/6?fw=pt

Comment: 仅介绍，无代码。

解码器模型仅使用转换器模型的解码器。在每个阶段，对于给定的单词，注意力层只能访问句子中位于其前面的单词。这些模型通常称为*自回归模型*。

解码器模型的预训练通常围绕预测句子中的下一个单词。

这些模型最适合涉及文本生成的任务。

CTRL
GPT
GPT-2
Transformer XL

## 1.7 Sequence-to-sequence models[sequence-to-sequence-models]

https://huggingface.co/learn/nlp-course/chapter1/7?fw=pt

Comment: 仅介绍，无代码。

编码器-解码器模型（也称为序列*到序列模型*）使用转换器体系结构的两个部分。在每个阶段，编码器的注意力层可以访问初始句子中的所有单词，而解码器的注意层只能访问输入中给定单词之前的位置。

序列到序列模型最适合围绕根据给定输入生成新句子的任务，例如摘要、翻译或生成问答。

BART
mBART
Marian
T5

## 1.8 Bias and limitations

https://huggingface.co/learn/nlp-course/chapter1/8?fw=pt

political correctness.

## 1.9 Summary

https://huggingface.co/learn/nlp-course/chapter1/9?fw=pt

| Model           | Examples                                   | Tasks                                                        |
| --------------- | ------------------------------------------ | ------------------------------------------------------------ |
| Encoder         | ALBERT, BERT, DistilBERT, ELECTRA, RoBERTa | Sentence classification, named entity recognition, extractive question answering |
| Decoder         | CTRL, GPT, GPT-2, Transformer XL           | Text generation                                              |
| Encoder-decoder | BART, T5, Marian, mBART                    | Summarization, translation, generative question answering    |

## 1.10 Quiz

https://huggingface.co/learn/nlp-course/chapter1/10?fw=pt

# 2. USING 🤗 TRANSFORMERS

## 2.1 Introduction

https://huggingface.co/learn/nlp-course/chapter2/1?fw=pt

Comment: 仅介绍，无代码。

## 2.2 Behind the pipeline

https://huggingface.co/learn/nlp-course/chapter2/2?fw=pt

Comment: 将端到端pipeline分解为各个步骤。

*分词器*，它将负责：

- 将输入拆分为称为*标记*的单词、子单词或符号（如标点符号）
- 将每个标记映射到一个整数
- 添加可能对模型有用的其他输入

### 2.2.1 Preprocessing with a tokenizer

### 2.2.2 Going through the model

### 2.3.3 A high-dimensional vector?

### 2.3.4 Model heads: Making sense out of numbers

### 2.3.5 Postprocessing the output

### Q: 什么是logits?

在机器学习和深度学习中，Logits 是指模型的输出层（或最后一层）的原始预测结果，尚未经过概率化转换的值。Logits 是一个向量，其中的每个元素表示模型对于不同类别的预测得分或原始输出值。

Logits 的值通常不具有直接的概率解释，它们可以是任意实数，可以是正数、负数或零。在分类问题中，对于每个类别，logits 的数值越高，表示模型对该类别的预测置信度越高；而数值越低，则表示模型对该类别的预测置信度越低。

为了将 logits 转换为概率分布，常见的做法是使用 softmax 函数，将 logits 映射为概率值。Softmax 函数将每个 logits 值转换为一个介于 0 到 1 之间的概率，并且所有类别的概率之和为 1。这样可以更直观地解释模型的输出，得到每个类别的概率预测。

总结起来，logits 是指模型输出层的原始预测结果，表示模型对不同类别的预测得分或原始输出值，尚未经过概率化转换。通过应用 softmax 函数，可以将 logits 转换为概率分布，提供类别预测的概率信息。

## 2.3 Models

https://huggingface.co/learn/nlp-course/chapter2/3?fw=pt

### 2.3.1 Creating a Transformer

### 2.3.2 Different loading methods

### 2.3.3 Saving methods

### 2.3.4 Using a Transformer model for inference

## 2.4 Tokenizers

https://huggingface.co/learn/nlp-course/chapter2/4?fw=pt

### 2.4.1 Word-based

如果我们想用基于单词的分词器完全覆盖一种语言，我们需要为语言中的每个单词都有一个标识符，这将生成大量的标记。例如，英语中有超过 500，000 个单词，因此要构建从每个单词到输入 ID 的映射，我们需要跟踪这么多 ID。此外，像“dog”这样的单词与像“dogs”这样的单词表示不同，并且模型最初无法知道“dog”和“dogs”是相似的：它会将这两个单词识别为不相关的。这同样适用于其他类似的单词，如“run”和“running”，模型最初不会看到它们相似。

最后，我们需要一个自定义标记来表示不在词汇表中的单词。这称为“未知”标记，通常表示为“[UNK]”或“”。如果您看到分词器正在生成大量这些标记，这通常是一个不好的迹象，因为它无法检索单词的合理表示，并且您在此过程中丢失了信息。制作词汇表时的目标是以这样的方式进行，即分词器将尽可能少的单词标记到未知标记中。

### 2.4.2 Character-based

### 2.4.3 Subword tokenization

### 2.4.4 And more!

### 2.4.5 Loading and saving

### 2.4.6 Encoding

### 2.4.7 Tokenization

原来一个单词transformer也可以分成两个。

### 2.4.8 From tokens to input IDs

### 2.4.9 Decoding

## 2.5 Handling multiple sequences

https://huggingface.co/learn/nlp-course/chapter2/5?fw=pt

### 2.5.1 Models expect a batch of inputs

### 2.5.2 Padding the inputs

类似于computer vision中给缺失的部分填充上。也类似于python的broadcast. 

### 2.5.3 Attention masks

### 2.5.4 Longer sequences

## 2.6 Putting it all together

https://huggingface.co/learn/nlp-course/chapter2/6?fw=pt

### 2.6.1 Special tokens

```
[CLS] i've been waiting for a huggingface course my whole life. [SEP]
```

### 2.6.2 Wrapping up: From tokenizer to model

## 2.7 Basic usage completed!

https://huggingface.co/learn/nlp-course/chapter2/7?fw=pt

## 2.8 End-of-chapter quiz

https://huggingface.co/learn/nlp-course/chapter2/8?fw=pt

# 3. FINE-TUNING A PRETRAINED MODEL

## 3.1 Introduction

https://huggingface.co/learn/nlp-course/chapter3/1?fw=pt

## 3.2 Processing the data

https://huggingface.co/learn/nlp-course/chapter3/2?fw=pt

### 3.2.1 Preprocessing a dataset

### 3.2.2 Dynamic padding
