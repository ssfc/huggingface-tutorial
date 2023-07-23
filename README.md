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

### Q: huggingface pipline是干啥的？

Hugging Face Transformers 库中的 `pipeline` 是一个方便的工具，用于通过预训练模型快速实现各种自然语言处理（NLP）任务，包括文本分类、命名实体识别、问答等。`pipeline` 提供了一种简单的方式，可以在几行代码内加载预训练模型并对文本进行处理，无需编写复杂的模型代码。

`pipeline` 函数的主要作用是自动加载指定任务的预训练模型和相关配置，然后对输入文本进行预测，输出相应的任务结果。你只需要指定任务名称，`pipeline` 会自动选择和加载与任务相对应的预训练模型，并返回一个可以直接用于预测的函数。

使用 `pipeline` 函数的示例代码如下：

```python
from transformers import pipeline

# 加载文本分类任务的预训练模型
text_classification = pipeline(task="text-classification", model="bert-base-uncased")

# 输入文本，返回分类结果
result = text_classification("This is a positive sentence.")
print(result)
```

在上述示例中，我们使用 `pipeline` 函数加载了一个文本分类任务的预训练模型（使用 BERT 模型），然后将文本 "This is a positive sentence." 输入到模型中，返回了相应的分类结果。`pipeline` 函数的返回结果是一个包含任务预测结果的字典或列表，具体取决于任务类型。

使用 `pipeline` 可以简化 NLP 任务的实现，尤其对于非常常见的任务，你可以快速部署和测试预训练模型，而不需要深入了解模型细节和编写复杂的代码。这使得 Hugging Face Transformers 库的功能更加易用和适用于广泛的 NLP 应用。

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

### Q: Hugging Face Transformers 库是什么？

Hugging Face Transformers 库是由 Hugging Face 开发的一个开源自然语言处理（NLP）库，它提供了丰富的预训练模型、分词器、优化器以及用于自然语言处理任务的工具和功能。这个库是基于 PyTorch 和 TensorFlow 等深度学习框架的，并且提供了一种简单而强大的方式来使用和部署预训练的 NLP 模型。

Hugging Face Transformers 库的主要特点和功能包括：

1. 预训练模型：提供了大量的预训练 NLP 模型，包括 BERT、GPT、RoBERTa、DistilBERT 等，可以用于各种 NLP 任务。

2. 分词器：提供了灵活高效的分词器，可以将文本序列转换为模型可接受的输入格式。

3. 模型架构：支持多种不同的模型架构，包括编码器-解码器模型、自回归模型、文本分类模型等。

4. 任务支持：提供了丰富的任务支持，如文本分类、序列标注、问答等，可以轻松地进行各种 NLP 任务。

5. Fine-tuning：支持对预训练模型进行微调（Fine-tuning），使其适用于特定任务和数据集。

6. 预训练模型库：提供了 Hugging Face 社区的模型分享和开源，用户可以从社区获取和共享预训练模型。

7. 易用性：提供了简洁的 API 和工具，使得使用预训练模型和实现 NLP 任务变得非常简单和高效。

Hugging Face Transformers 库受到广泛的欢迎和使用，它在学术界和工业界都有着很高的影响力。许多最新的 NLP 研究成果和应用都采用了 Hugging Face Transformers 库中的预训练模型和工具，因为它为 NLP 任务提供了强大而便捷的解决方案。

## 2.2 Behind the pipeline

https://huggingface.co/learn/nlp-course/chapter2/2?fw=pt

Comment: 将端到端pipeline分解为各个步骤。

*分词器*，它将负责：

- 将输入拆分为称为*标记*的单词、子单词或符号（如标点符号）
- 将每个标记映射到一个整数
- 添加可能对模型有用的其他输入

### Q: huggingface AutoModel是干啥的？

`AutoModel` 是 Hugging Face Transformers 库中的一个类，它是一个方便的工具，用于根据指定的模型名称或 checkpoint 来自动选择和加载相应的预训练模型。

在深度学习中，使用预训练模型是一种常见的方式，可以为各种自然语言处理（NLP）任务提供强大的基础。然而，Hugging Face Transformers 库提供了许多不同类型的预训练模型（例如 BERT、GPT、RoBERTa 等），而每个模型可能有多个不同的变体和配置。为了方便用户加载和使用不同模型，`AutoModel` 类允许用户通过一个简单的 API 来加载预训练模型，而无需手动选择特定的模型或配置。

`AutoModel` 的主要作用是根据给定的模型名称或 checkpoint 来自动选择和加载对应的预训练模型。这样，你只需要指定模型名称或 checkpoint，`AutoModel` 就会自动选择和加载与之对应的预训练模型。

使用 `AutoModel` 的示例代码如下：

```python
from transformers import AutoModel

# 指定模型名称或 checkpoint
checkpoint = "bert-base-uncased"

# 自动选择和加载对应的预训练模型
model = AutoModel.from_pretrained(checkpoint)

# 在模型中进行相应的处理和预测
# ...
```

在上述示例中，我们使用 `AutoModel.from_pretrained(checkpoint)` 来自动加载名为 "bert-base-uncased" 的 BERT 模型。之后，我们可以在模型中进行相应的处理和预测，而不需要手动选择特定的模型或配置。

总之，`AutoModel` 类是 Hugging Face Transformers 库中的一个实用工具，它为用户提供了一种简单的方式来自动加载指定模型的预训练权重，使得在不同任务和模型配置中的转换变得非常方便和灵活。

### Q: huggingface AutoModelForSequenceClassification是干啥的？

`AutoModelForSequenceClassification` 是 Hugging Face Transformers 库中的一个类，它是用于序列分类任务的预训练模型的自动加载工具。在自然语言处理（NLP）中，序列分类任务是指将输入的文本序列（例如句子、段落）分类到预定义的类别或标签中的任务。

`AutoModelForSequenceClassification` 类的作用是根据指定的模型名称或 checkpoint 来自动选择和加载适用于序列分类任务的预训练模型。它是 `AutoModel` 类的一个子类，在加载模型的同时，它会自动设置模型的输出层，以适应序列分类任务的特定需求。

使用 `AutoModelForSequenceClassification` 的示例代码如下：

```python
from transformers import AutoModelForSequenceClassification

# 指定模型名称或 checkpoint
checkpoint = "bert-base-uncased"

# 自动选择和加载适用于序列分类任务的预训练模型
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

# 在模型中进行序列分类任务的处理和预测
# ...
```

在上述示例中，我们使用 `AutoModelForSequenceClassification.from_pretrained(checkpoint)` 来自动加载名为 "bert-base-uncased" 的 BERT 模型，并设置它适用于序列分类任务。之后，我们可以在模型中进行序列分类任务的处理和预测，而不需要手动设置模型的输出层和其他配置。

`AutoModelForSequenceClassification` 类的存在使得在序列分类任务中使用预训练模型变得非常方便和高效。它提供了一种简单的方式来加载预训练模型，并且自动设置模型以适应序列分类任务的需求，减少了用户的工作量，并提高了实验和应用的效率。

### Q: huggingface AutoTokenizer是干啥的？

`AutoTokenizer` 是 Hugging Face Transformers 库中的一个类，它是一个自动加载预训练模型的分词器（Tokenizer）的工具类。在自然语言处理（NLP）中，分词器用于将输入文本（句子、段落等）拆分成单词或子词的序列，以便机器学习模型能够处理和理解文本。

`AutoTokenizer` 的主要作用是根据给定的模型名称或 checkpoint 来自动选择和加载对应的预训练模型的分词器。这个类是 `AutoModel` 和 `AutoModelForSequenceClassification` 类的分词器版本，它使得在加载预训练模型和分词器时变得非常方便和灵活。

使用 `AutoTokenizer` 有以下几个优点：

1. 自动选择模型：无需手动指定模型名称，`AutoTokenizer` 会根据提供的模型名称自动选择和加载对应的分词器。

2. 多种模型支持：`AutoTokenizer` 支持加载各种不同的预训练模型的分词器，如 BERT、GPT、RoBERTa 等。

3. 方便的代码迁移：如果你在代码中使用了 `AutoTokenizer` 来加载分词器，当你改变模型时，只需更改模型名称，而不需要修改其他代码。

使用 `AutoTokenizer` 的示例代码如下：

```python
from transformers import AutoTokenizer

# 指定模型名称或 checkpoint
checkpoint = "bert-base-uncased"

# 自动选择和加载对应的分词器
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# 使用分词器对输入文本进行分词
text = "Hello, how are you?"
tokens = tokenizer(text)

# 输出分词结果
print(tokens)
```

在上述示例中，我们使用 `AutoTokenizer.from_pretrained(checkpoint)` 来自动加载名为 "bert-base-uncased" 的 BERT 模型的分词器。之后，我们使用分词器对输入文本进行分词，并输出分词结果。通过 `AutoTokenizer` 的使用，我们可以很方便地加载和切换不同的预训练模型的分词器，从而进行文本处理和 NLP 任务。

### Q: model(**inputs)为什么inputs前面有\*\*?

在 Python 中，双星号（`**`）用于解包字典（或关键字参数）并将其作为关键字参数传递给函数。在 `model(**inputs)` 中，`**inputs` 表示将 `inputs` 这个字典中的键值对解包，并将其中的键作为关键字参数名，将对应的值作为关键字参数值传递给函数 `model`。

在 Hugging Face Transformers 库中，`model` 是一个预训练模型的实例，可以接受一组特定的输入参数，例如 `input_ids`、`attention_mask` 等。而 `inputs` 是一个字典，其中包含了这些输入参数，键为参数名，值为相应的张量数据。

通过使用 `**inputs`，我们可以将字典 `inputs` 中的键值对解包，并将其传递给 `model` 函数作为关键字参数。这样做的好处是，可以根据需要轻松地添加或修改输入参数，而不需要显式地逐个传递每个参数。

示例：

假设 `inputs` 字典如下：

```python
inputs = {
    "input_ids": [1, 2, 3, 4, 5],
    "attention_mask": [1, 1, 1, 1, 1],
    "token_type_ids": [0, 0, 0, 0, 0]
}
```

如果我们想将这些输入参数传递给函数 `model`，传统的方式是：

```python
model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], token_type_ids=inputs["token_type_ids"])
```

但使用 `**inputs` 可以更简洁地实现相同的效果：

```python
model(**inputs)
```

这样，Python 会自动将字典 `inputs` 中的键值对解包，并将其作为关键字参数传递给函数 `model`，使得代码更加简洁和易读。

### 2.2.1 Preprocessing with a tokenizer

### 2.2.2 Going through the model

### 2.3.3 A high-dimensional vector?

### Q: outputs.last_hidden_state是在logits前一层吗？

是的，`outputs.last_hidden_state` 在 `logits` 前一层。在 Transformers 模型中，`outputs.last_hidden_state` 包含了模型的中间表示，也称为上下文向量（contextual embeddings）或特征向量（feature embeddings）。它是模型在输入序列上进行处理后的最终输出，记录了输入序列中每个位置（单词或子词）的编码表示。

`outputs.last_hidden_state` 的维度是 `[batch_size, sequence_length, hidden_size]`，其中 `batch_size` 表示批量大小，`sequence_length` 表示序列的长度，`hidden_size` 表示隐藏层的大小（即每个位置的特征向量维度）。

`logits` 则是在 `outputs.last_hidden_state` 基础上通过模型的最后一层或特定输出层得到的预测输出。对于分类任务，`logits` 是模型在每个类别上的分数或得分，表示输入属于每个类别的概率。`logits` 的维度通常是 `[batch_size, num_classes]`，其中 `num_classes` 表示分类任务中的类别数。

在文本分类任务中，通常会使用 `outputs.last_hidden_state` 来获取输入序列的表示，然后通过一些后续的操作（例如平均池化、最大池化等）来获得整个序列的固定表示。然后，这个固定表示可以作为输入传递给分类层，生成最终的 `logits`，并通过 softmax 函数获得分类概率。

总结：
- `outputs.last_hidden_state` 是模型的中间表示，包含输入序列中每个位置的编码表示。
- `logits` 是模型的预测输出，是在 `outputs.last_hidden_state` 基础上得到的，并用于计算类别的得分或概率分布。

Comment: 其实last_hidden_state的维数远远大于logits的维数。

### 2.3.4 Model heads: Making sense out of numbers

### 2.3.5 Postprocessing the output

### Q: 什么是logits?

在机器学习和深度学习中，Logits 是指模型的输出层（或最后一层）的原始预测结果，尚未经过概率化转换的值。Logits 是一个向量，其中的每个元素表示模型对于不同类别的预测得分或原始输出值。

Logits 的值通常不具有直接的概率解释，它们可以是任意实数，可以是正数、负数或零。在分类问题中，对于每个类别，logits 的数值越高，表示模型对该类别的预测置信度越高；而数值越低，则表示模型对该类别的预测置信度越低。

为了将 logits 转换为概率分布，常见的做法是使用 softmax 函数，将 logits 映射为概率值。Softmax 函数将每个 logits 值转换为一个介于 0 到 1 之间的概率，并且所有类别的概率之和为 1。这样可以更直观地解释模型的输出，得到每个类别的概率预测。

总结起来，logits 是指模型输出层的原始预测结果，表示模型对不同类别的预测得分或原始输出值，尚未经过概率化转换。通过应用 softmax 函数，可以将 logits 转换为概率分布，提供类别预测的概率信息。

### Q: predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)中dim=-1是什么意思？

在 PyTorch 中，`dim=-1` 表示在张量的最后一个维度上进行操作。在 `torch.nn.functional.softmax` 函数中，`dim` 参数用于指定 softmax 操作应该沿着哪个维度进行计算。

考虑一个张量 `x`，它的维度是 `[a, b, c, ...]`，其中 `a`, `b`, `c`, ... 是张量的维度大小。如果我们使用 `dim=-1`，那么 softmax 操作会在最后一个维度上进行计算。例如，对于维度为 `[batch_size, num_classes]` 的张量，使用 `dim=-1` 表示 softmax 操作会在 `num_classes` 这个维度上进行计算，即对每个样本的类别分数进行 softmax。

在代码中，`outputs.logits` 是模型在分类任务上的原始预测输出，是一个维度为 `[batch_size, num_classes]` 的张量。通过 `torch.nn.functional.softmax(outputs.logits, dim=-1)`，我们对每个样本的类别分数进行 softmax，得到分类概率分布。

示例：

```python
import torch
import torch.nn.functional as F

# 假设有一个维度为 [batch_size, num_classes] 的张量 predictions
predictions = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

# 在最后一个维度上进行 softmax 计算
softmax_predictions = F.softmax(predictions, dim=-1)

print(softmax_predictions)
```

输出结果：

```
tensor([[0.0900, 0.2447, 0.6652],
        [0.0900, 0.2447, 0.6652]])
```

在上述示例中，我们使用 `dim=-1` 对张量 `predictions` 进行 softmax 计算，即在 `num_classes` 这个维度上进行 softmax。最后，得到了两个样本的分类概率分布。注意，每个样本的分类概率之和等于 1。

### Q: model.config.id2label是什么意思？

在 Hugging Face Transformers 中，`model.config.id2label` 是一个词典（字典），用于将模型输出的类别标签 ID（整数值）映射回对应的类别标签（字符串标签）。

在文本分类任务中，通常模型的输出是一个张量，每个元素代表一个类别的得分或概率。为了将这些得分或概率转换为对应的类别标签，我们需要一个映射关系，将类别标签的 ID 映射回实际的类别名称。

`model.config.id2label` 就是用来构建这种映射关系的词典。它的键是类别标签的 ID（整数值），值是对应的类别名称（字符串标签）。

例如，假设我们有一个文本分类任务，共有三个类别，其对应的 ID 和类别名称如下：

```python
id2label = {
    0: "positive",
    1: "neutral",
    2: "negative"
}
```

在这个例子中，类别 "positive" 对应 ID 0，类别 "neutral" 对应 ID 1，类别 "negative" 对应 ID 2。如果模型输出了一个包含三个类别得分的张量 `[0.1, 0.5, 0.4]`，我们可以使用 `model.config.id2label` 将得分映射回实际的类别标签：

```python
output = [0.1, 0.5, 0.4]
predicted_label_id = torch.argmax(output, dim=-1)
predicted_label = model.config.id2label[predicted_label_id.item()]
print(predicted_label)
```

在上述代码中，我们首先使用 `torch.argmax` 函数找到得分最高的类别标签 ID，然后使用 `model.config.id2label` 将其映射回实际的类别名称。最后，输出预测的类别名称，例如 "neutral"。

总结：`model.config.id2label` 是一个词典，用于将模型输出的类别标签 ID 映射回实际的类别名称，便于理解和展示模型的预测结果。

## 2.3 Models

https://huggingface.co/learn/nlp-course/chapter2/3?fw=pt

### 2.3.1 Creating a Transformer

### Q: huggingface中的BertConfig是什么？

在 Hugging Face Transformers 库中，`BertConfig` 是用于配置 BERT 模型的类。BERT（Bidirectional Encoder Representations from Transformers）是一种流行的预训练模型，在自然语言处理（NLP）任务中取得了很好的效果。`BertConfig` 类用于定义和设置 BERT 模型的超参数和配置选项。

通过 `BertConfig` 类，你可以指定以下超参数和配置选项：

1. `vocab_size`: 词汇表的大小，即 BERT 模型支持的唯一词汇数量。

2. `hidden_size`: BERT 模型的隐藏层大小（即每个位置的特征向量维度）。

3. `num_hidden_layers`: BERT 模型的隐藏层数量。

4. `num_attention_heads`: BERT 模型的注意力头（attention head）数量。

5. `intermediate_size`: BERT 模型中全连接层的中间层大小。

6. `hidden_act`: BERT 模型中的激活函数。

7. `hidden_dropout_prob`: 隐藏层的 dropout 概率，用于模型训练时的随机失活。

8. `attention_probs_dropout_prob`: 注意力层的 dropout 概率，用于模型训练时的随机失活。

9. `max_position_embeddings`: 输入序列的最大长度（最大位置编码）。

10. `type_vocab_size`: 类型词汇表的大小，用于区分单句和双句输入。

11. `initializer_range`: 初始化权重的范围。

等等。

通过创建一个 `BertConfig` 的实例，你可以定义一个 BERT 模型的配置，然后将其传递给 `BertModel` 或 `BertForSequenceClassification` 等模型的构造函数，以创建相应的 BERT 模型。

示例：

```python
from transformers import BertConfig, BertModel

# 创建一个 BERT 模型的配置
config = BertConfig(
    vocab_size=30522,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    hidden_act="gelu",
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    max_position_embeddings=512,
    type_vocab_size=2,
    initializer_range=0.02
)

# 使用配置创建 BERT 模型
model = BertModel(config)
```

在上述示例中，我们创建了一个 `BertConfig` 的实例 `config`，然后使用这个配置创建了一个 BERT 模型 `model`。通过 `config` 中的各种超参数和配置选项，可以灵活地定义和配置 BERT 模型的结构和特性。

### Q: huggingface中的BertModel是什么？

在 Hugging Face Transformers 库中，`BertModel` 是 BERT 模型的基本实现类。BERT（Bidirectional Encoder Representations from Transformers）是一种预训练的自然语言处理（NLP）模型，由 Google 在 2018 年提出，通过双向编码和自注意力机制来学习文本的上下文表示。

`BertModel` 是 BERT 模型的基础类，它由多个堆叠的 Transformer 编码器组成，每个编码器由多层自注意力和前馈神经网络组成。这些编码器通过堆叠来形成一个深层的神经网络，可以用于处理不同的 NLP 任务，如文本分类、序列标注、问答等。

在使用 `BertModel` 之前，通常需要先加载一个预训练的 BERT 模型，可以使用 `from_pretrained()` 方法从 Hugging Face 模型库中加载预训练的权重。然后，你可以通过 `BertModel` 类来进行文本编码、特征提取等操作。

示例：

```python
from transformers import BertModel, BertTokenizer

# 加载预训练的 BERT 模型和分词器
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# 输入文本进行编码
text = "Hello, how are you?"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)

# 获取输出的隐藏状态（上下文向量）
hidden_states = outputs.last_hidden_state
```

在上述示例中，我们加载了预训练的 "bert-base-uncased" 模型和相应的分词器。然后，我们使用分词器对输入文本进行编码，得到一个输入字典 `inputs`。接着，我们将 `inputs` 传递给 `BertModel` 类的实例 `model` 进行处理，并得到输出结果 `outputs`。`outputs` 包含了模型的隐藏状态（上下文向量）等信息，可以用于后续的任务，比如文本分类或其他自然语言处理任务。

总结：`BertModel` 是 Hugging Face Transformers 库中 BERT 模型的基本实现类，可以用于进行文本编码和特征提取等操作，是进行自然语言处理任务的基础。

### Q: bert-base-uncased是什么模型？

`bert-base-uncased` 是 Hugging Face Transformers 库中预训练的 BERT 模型之一。BERT（Bidirectional Encoder Representations from Transformers）是由 Google 在 2018 年提出的一种基于 Transformer 架构的预训练自然语言处理模型，以无监督的方式从大规模文本语料中学习文本的上下文表示。

在 Hugging Face Transformers 库中，BERT 模型有多个不同的预训练版本，其中 `bert-base-uncased` 是其中之一。`uncased` 表示该模型使用的是小写字母形式的文本，即在预训练时**将所有文本转换为小写形式，不区分大小写**。例如，"Hello" 和 "hello" 在模型的输入中被视为相同的单词。

`bert-base-uncased` 模型是 BERT 的基本版本，它具有以下特点：

1. 模型的输入文本被处理为小写形式，不区分大小写。
2. 模型的词汇表包含了大量的英文单词和子词，使其具有较强的语义表达能力。
3. 模型的隐藏层大小为 768，有 12 个隐藏层，多头注意力机制等。

该模型是经过大规模预训练的，并且在多个自然语言处理任务上表现出色。因此，你可以使用 `bert-base-uncased` 模型进行文本编码、特征提取，或在下游任务中进行微调（fine-tuning），例如文本分类、序列标注、问答等。

示例：

```python
from transformers import BertTokenizer, BertModel

# 加载预训练的 bert-base-uncased 模型和分词器
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# 输入文本进行编码
text = "Hello, how are you?"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)

# 获取输出的隐藏状态（上下文向量）
hidden_states = outputs.last_hidden_state
```

在上述示例中，我们加载了预训练的 `bert-base-uncased` 模型和相应的分词器，并使用它对输入文本进行编码。然后，我们可以获取模型的输出结果 `outputs`，其中包含了上下文向量（隐藏状态），可以用于后续的任务。

### Q: BertTokenizer是干啥的？

`BertTokenizer` 是 Hugging Face Transformers 库中用于将文本转换为 BERT 模型可接受的输入格式的工具类。BERT（Bidirectional Encoder Representations from Transformers）模型在输入文本之前需要进行一系列的预处理，包括分词（tokenization）和添加特殊标记（special tokens）。`BertTokenizer` 就是用来完成这些预处理步骤的。

在使用 BERT 模型之前，我们需要先对输入文本进行分词，将文本拆分成多个单词或子词，同时添加一些特殊标记以便模型理解输入的结构。`BertTokenizer` 就是为了帮助我们实现这个过程。

`BertTokenizer` 的主要功能有：

1. 分词：将输入文本分割成一个个单词或子词，构成模型输入的标记（tokens）序列。
2. 添加特殊标记：在输入文本的开头和结尾添加 `[CLS]` 和 `[SEP]` 等特殊标记，用于标识句子的开始和结束，以及分隔多个句子。
3. 编码：将分词后的标记序列转换为模型输入的张量形式，包括转换为对应的整数 ID 和添加对齐用的填充标记。

示例：

```python
from transformers import BertTokenizer

# 加载预训练的 bert-base-uncased 分词器
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)

# 输入文本
text = "Hello, how are you?"

# 使用 tokenizer 对文本进行分词和编码
inputs = tokenizer(text, return_tensors="pt")

# 输出编码后的结果
print(inputs)
```

在上述示例中，我们加载了预训练的 `bert-base-uncased` 分词器，并使用它对输入文本进行了分词和编码。最后，我们得到了包含模型输入信息的字典 `inputs`，其中包括了分词后的标记 IDs 和其他有关模型输入的信息。这样的处理方式使得我们可以直接将 `inputs` 输入到 BERT 模型中进行文本编码和特征提取。

### Q:  [CLS] 和 [SEP]是什么意思？

`[CLS]` 和 `[SEP]` 是两个特殊标记（special tokens），在 BERT 模型中用于辅助输入文本的处理。它们在文本序列的开头和结尾添加，并在训练和推理过程中有着特殊的作用。

1. `[CLS]`：这个标记表示“分类”（classification），它通常被添加在输入文本的开头。在预训练阶段，BERT 模型使用 `[CLS]` 标记来获取整个输入句子的上下文信息，并生成一个特殊的句子表示（sentence embedding）。这个句子表示在进行下游任务时，比如文本分类或问答，通常被用作模型的输出。在分类任务中，`[CLS]` 标记后面的隐藏状态（上下文向量）被用来进行分类预测。

2. `[SEP]`：这个标记表示“分隔”（separator），它通常被添加在不同句子之间的位置。在 BERT 模型中，输入可以是单个句子或者是两个句子的组合。为了区分多个句子的输入，我们需要使用 `[SEP]` 标记将它们分隔开。在预训练阶段，BERT 模型会学习到句子之间的关系，并且将 `[SEP]` 标记用于对输入句子进行分隔。

举例来说，在进行文本分类任务时，假设我们有两个句子 "Hello, how are you?" 和 "I'm doing well."。在输入到 BERT 模型前，我们需要添加 `[CLS]` 和 `[SEP]` 标记来表示句子的开始和结束，得到输入序列为：`[CLS] Hello, how are you? [SEP] I'm doing well. [SEP]`。BERT 模型会对这个输入序列进行处理，生成对应的隐藏状态（上下文向量），然后在分类任务中使用 `[CLS]` 标记后面的隐藏状态进行分类预测。

总结：`[CLS]` 和 `[SEP]` 是 BERT 模型中的特殊标记，用于表示句子的开始和结束，以及分隔不同句子的位置，在预训练和下游任务中有着特殊的作用。

### Q: 也就是说，[CLS]只有一个位于开头，[SEP]可以有好多个用于分隔句子？

正是的！在 BERT 模型中，`[CLS]` 标记只出现一次，并且通常位于输入序列的开头。它用于表示整个输入句子的特殊句子表示，供后续的任务使用，比如文本分类、序列标注等。

而 `[SEP]` 标记可以出现多次，并且用于分隔不同的句子或文本段落。在单个句子的情况下，只会有一个 `[SEP]` 标记，用于表示句子的结束。但在处理多个句子的情况下，每个句子之间都会用 `[SEP]` 标记进行分隔，以帮助 BERT 模型识别不同句子之间的边界和关系。

例如，假设我们有两个句子 "I love NLP." 和 "It's fascinating."，为了将它们输入到 BERT 模型中，我们需要添加 `[CLS]` 和两个 `[SEP]` 标记，得到以下输入序列：

```
[CLS] I love NLP. [SEP] It's fascinating. [SEP]
```

这样的输入序列可以用于 BERT 模型的预训练和下游任务处理。每个 `[SEP]` 标记帮助模型识别不同句子之间的边界，从而更好地学习句子之间的关系。

总结：`[CLS]` 标记只出现一次，位于输入序列开头，用于句子的特殊表示；`[SEP]` 标记可以出现多次，用于分隔不同句子或文本段落，帮助模型识别句子之间的边界和关系。

### Q: tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")是啥意思？

这行代码使用 Hugging Face Transformers 库中的 `AutoTokenizer` 类来加载预训练的 BERT 模型的分词器（Tokenizer）。`AutoTokenizer` 是一个方便的工具，用于根据模型名称自动选择合适的分词器。

具体来说，`AutoTokenizer.from_pretrained("bert-base-cased")` 的意思是：

1. `"bert-base-cased"` 是一个预训练的 BERT 模型的名称。在 Hugging Face Transformers 库中，模型名称通常由两部分组成，第一部分表示模型的基本结构和配置，第二部分表示模型的大小和是否区分大小写。在这个例子中，`"bert-base-cased"` 表示一个基本的 BERT 模型，且模型在预训练时区分了大小写。

2. `AutoTokenizer.from_pretrained("bert-base-cased")` 通过指定模型名称，自动选择并加载与该模型对应的分词器。这样，我们就可以直接使用 `tokenizer` 对象来进行文本分词，将文本转换为模型可以接受的输入格式。

示例：

```python
from transformers import AutoTokenizer

# 使用 AutoTokenizer 加载预训练的 bert-base-cased 分词器
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

# 输入文本
text = "Hello, how are you?"

# 使用 tokenizer 对文本进行分词和编码
inputs = tokenizer(text, return_tensors="pt")

# 输出编码后的结果
print(inputs)
```

在上述示例中，我们使用 `AutoTokenizer.from_pretrained("bert-base-cased")` 加载了预训练的 "bert-base-cased" 模型对应的分词器，并将文本 "Hello, how are you?" 分词并编码为模型可接受的输入格式。这样，我们就可以使用 `inputs` 输入到 BERT 模型中进行文本编码和特征提取。

### Q: tokenizer.save_pretrained("directory_on_my_computer")是干啥？

`tokenizer.save_pretrained("directory_on_my_computer")` 是 Hugging Face Transformers 库中的方法，用于将预训练的分词器（Tokenizer）保存到本地文件夹中，以便以后可以重新加载和使用该分词器。

具体来说，`tokenizer.save_pretrained("directory_on_my_computer")` 的意思是：

1. `"directory_on_my_computer"` 是本地文件夹的路径，你可以自己指定保存分词器的文件夹位置。该路径应该是一个有效的目录路径，用于存储分词器的相关文件。

2. 调用这个方法后，Hugging Face Transformers 库会将分词器的相关配置、词汇表文件、特殊标记等信息保存到指定的目录中。

为什么要使用 `tokenizer.save_pretrained()` 呢？通常情况下，当我们加载预训练的模型和分词器时，每次都要从 Hugging Face 模型库中下载并加载，这可能会花费一些时间和网络资源。为了避免重复下载和加载，我们可以使用 `tokenizer.save_pretrained()` 将分词器保存到本地文件夹中，然后在以后的使用中，只需要从本地文件夹加载分词器即可，这样会更快捷方便。

示例：

```python
from transformers import AutoTokenizer

# 加载预训练的分词器
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

# 将分词器保存到本地文件夹
tokenizer.save_pretrained("directory_on_my_computer")
```

在上述示例中，我们加载了预训练的 "bert-base-cased" 分词器，并将它保存到名为 "directory_on_my_computer" 的本地文件夹中。之后，我们可以使用 `AutoTokenizer.from_pretrained("directory_on_my_computer")` 来加载这个本地的分词器，而不必再次从远程模型库下载。这样可以提高代码的运行效率，特别是在多次运行代码时。

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

### Q: AutoTokenizer是干啥的？

`AutoTokenizer` 是 Hugging Face Transformers 库中的一个类，它是一个自动加载预训练模型的分词器（Tokenizer）的工具类。在自然语言处理（NLP）中，分词器用于将输入文本（句子、段落等）拆分成单词或子词的序列，以便机器学习模型能够处理和理解文本。

`AutoTokenizer` 的主要作用是根据给定的模型名称或 checkpoint 来自动选择和加载对应的预训练模型的分词器。这样，你可以通过一个简单的 API 调用来加载不同模型的分词器，而不需要手动指定特定模型的分词器。

使用 `AutoTokenizer` 有以下几个优点：

1. 自动选择模型：无需手动指定模型名称，`AutoTokenizer` 会根据提供的模型名称自动选择和加载对应的分词器。

2. 多种模型支持：`AutoTokenizer` 支持加载各种不同的预训练模型的分词器，如 BERT、GPT、RoBERTa 等。

3. 方便的代码迁移：如果你在代码中使用了 `AutoTokenizer` 来加载分词器，当你改变模型时，只需更改模型名称，而不需要修改其他代码。

使用 `AutoTokenizer` 的示例代码如下：

```python
from transformers import AutoTokenizer

# 指定模型名称或 checkpoint
checkpoint = "bert-base-uncased"

# 自动选择和加载对应的分词器
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# 使用分词器对输入文本进行分词
text = "Hello, how are you?"
tokens = tokenizer(text)

# 输出分词结果
print(tokens)
```

Output:  

{'input_ids': [101, 7592, 1010, 2129, 2024, 2017, 1029, 102], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1]}

在上述示例中，我们使用 `AutoTokenizer.from_pretrained(checkpoint)` 来自动加载名为 "bert-base-uncased" 的 BERT 模型的分词器。之后，我们使用分词器对输入文本进行分词，并输出分词结果。通过 `AutoTokenizer` 的使用，我们可以很方便地加载和切换不同的预训练模型的分词器，从而进行文本处理和 NLP 任务。

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

### Q: 结果中的token_type_ids是什么？

在 Hugging Face Transformers 库中，`token_type_ids` 是用于处理输入序列中的分段信息的一种编码方式。对于一些预训练模型（例如 BERT、RoBERTa 等），它们在输入时需要同时考虑两个句子或分段的信息，因此需要一种方式来区分不同分段的内容。

在处理文本序列时，通常将输入序列切分成多个片段，每个片段对应一个输入序列。例如，当处理句子对任务时，一个输入序列可能包含两个句子，其中一个句子放在前面，另一个句子放在后面，中间可能有一个特殊的分隔符。`token_type_ids` 就是用来区分这些不同片段的标识。

`token_type_ids` 是一个与输入序列等长的向量，其长度与输入序列中的 token 数目相同。在处理句子对任务时，如果一个 token 属于第一个句子，那么它的 `token_type_ids` 就会被标记为 0；如果它属于第二个句子，那么它的 `token_type_ids` 就会被标记为 1。这样，模型就可以根据 `token_type_ids` 来区分不同句子的信息。

示例：

假设我们有两个句子："I love NLP." 和 "It is fascinating."，并且分词后的结果如下：

```
tokens = ['[CLS]', 'I', 'love', 'NLP', '.', '[SEP]', 'It', 'is', 'fascinating', '.', '[SEP]']
```

对应的 `token_type_ids` 为：

```
token_type_ids = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
```

在上面的示例中，`token_type_ids` 将第一个句子中的 token 标记为 0，将第二个句子中的 token 标记为 1。这样，模型就可以根据 `token_type_ids` 来区分两个句子的内容，以更好地处理句子对任务。

### Q: 如果有三个分段，token_type_ids显示什么？

如果输入序列有三个分段（例如三个句子），则对应的 `token_type_ids` 将有三个不同的标识来区分这三个分段。每个分段的 `token_type_ids` 标识都是一个独立的整数值。

例如，假设有三个句子："I love NLP."，"It is fascinating."，"Natural Language Processing is amazing."，并且分词后的结果如下：

```
tokens = ['[CLS]', 'I', 'love', 'NLP', '.', '[SEP]', 'It', 'is', 'fascinating', '.', '[SEP]', 'Natural', 'Language', 'Processing', 'is', 'amazing', '.', '[SEP]']
```

对应的 `token_type_ids` 可能为：

```
token_type_ids = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2]
```

在上面的示例中，`token_type_ids` 将第一个句子中的 token 标记为 0，第二个句子中的 token 标记为 1，第三个句子中的 token 标记为 2。这样，模型就可以根据 `token_type_ids` 来区分三个不同分段的内容，以更好地处理多句子的任务。

### 3.2.2 Dynamic padding

### Q: 你知道transformer的Dynamic padding吗？

在 Transformer 模型中，Dynamic Padding 是一种技术，用于处理输入序列的不等长问题。在自然语言处理等任务中，输入文本的长度往往会不同，而 Transformer 模型要求输入序列的长度固定，因此需要将不等长的输入序列转换成等长的形式。

Dynamic Padding 是一种动态填充技术，它根据输入序列的实际长度来动态地在序列的末尾添加特殊的填充标记（通常是 0），从而将所有输入序列填充成相同长度。这样，所有输入序列都具有相同的长度，可以一起批量处理，以提高模型训练的效率。

具体来说，对于一个 batch 中的多个序列，首先找到 batch 中最长序列的长度，然后将其他较短的序列进行填充，使它们与最长序列长度相同。填充通常在序列的末尾添加特殊的填充标记（例如 0），以示区别原始数据和填充数据。在训练过程中，模型会忽略填充标记的影响，以确保填充部分不会影响模型的预测结果。

Dynamic Padding 技术在 Transformer 中是一种非常常用的方法，它使得 Transformer 可以处理不等长的输入序列，使得模型更具通用性和适用性，同时减少了内存和计算资源的浪费。

### Q: Hugging Face Datasets 库中的 map 函数是干啥的？

在 Hugging Face Datasets 库中，`map()` 函数是用于对数据集中的每个样本应用指定的函数，并返回一个新的数据集。这个函数类似于 Python 内置的 `map()` 函数，但是它专门用于处理数据集，方便对数据进行批量处理和转换。

`map()` 函数在 Hugging Face Datasets 库中的基本用法如下：

```python
map(function, batched=True, batch_size=None, num_proc=1, remove_columns=None, load_from_cache_file=True, cache_file_name=None)
```

- `function`：表示要对数据集中每个样本应用的函数，可以是一个函数或者一个处理数据的类。

- `batched`：表示是否将数据集以批量的方式处理，默认为 `True`。如果为 `True`，`function` 将会以批量的方式处理数据，提高处理效率。

- `batch_size`：表示批量处理的大小，默认为 `None`。当 `batched=True` 且 `batch_size` 不为 `None` 时，数据集会以指定的批量大小进行处理。

- `num_proc`：表示并行处理的进程数，默认为 1。如果 `num_proc` 大于 1，则 `map()` 函数会使用多进程并行处理数据。

- `remove_columns`：表示要从数据集中移除的列，通常用于移除不需要处理的列，以减少内存占用。

- `load_from_cache_file`：表示是否从缓存文件中加载数据，默认为 `True`。如果为 `True`，并且数据集有缓存文件，则会从缓存文件中加载数据，加快数据加载速度。

- `cache_file_name`：表示缓存文件的名称，默认为 `None`。如果指定了缓存文件名称，则会将数据缓存到指定的文件中，以便下次使用。

`map()` 函数在 Hugging Face Datasets 库中通常用于对数据进行预处理、转换或者特征提取。通过传入不同的 `function` 参数，可以对数据集中的每个样本进行不同的处理。这个函数在深度学习中数据预处理和数据加载过程中非常实用，可以帮助提高数据处理效率和灵活性。

## 3.3 Fine-tuning a model with the Trainer API
