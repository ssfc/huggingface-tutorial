# Huggingface tutorial

https://huggingface.co/learn/llm-course/chapter1/1

吾人认为，要想把握huggingface tutorial, 要从两个维度着手：(1) 把握tutorial整体的各章结构；(2) 以句读的方式把握各章的代码，把重要内容翻译到markdown文档。(2024年1月7日)

# 0. Setup

## 0.1 介绍

## 0.2 使用 Google Colab 笔记本

## 0.3 使用 Python 虚拟环境

## 0.4 安装依赖项

# 1. Transfomer Models

## 1.1 Introduction

https://huggingface.co/learn/nlp-course/chapter1/1?fw=pt

### 1.1.1 欢迎来到课程🤗！

本课程将教您如何使用 Hugging Face 生态系统中的库（[🤗 Transformers](https://github.com/huggingface/transformers), [🤗 Datasets](https://github.com/huggingface/datasets), [🤗 Tokenizers](https://github.com/huggingface/tokenizers), and [🤗 Accelerate](https://github.com/huggingface/accelerate)）以及 [Hugging Face Hub](https://huggingface.co/models) 中的自然语言处理 （[🤗](https://github.com/huggingface/transformers) NLP）。

Comment:  原来huggingface space是网页端应用的意思。(2024年2月9日)

### 1.1.2 期待什么

以下是该课程的简要概述：

| Introduction                      | Diving in                             | Advanced                      |
| --------------------------------- | ------------------------------------- | ----------------------------- |
| 1. Transformer models             | 5. The huggingface datasets library   | 9. Building and sharing demos |
| 2. Using huggingface transformers | 6. The huggingface tokenizers library | Transformers can hear         |
| 3. Fine-tuning a pretrained model | 7. Main NLP tasks                     | Transformers can see          |
| 4. Sharing models and tokenizers  | 8. How to ask for help                | Optimizing for production     |

- 第 1 章至第 4 章介绍了 Transformers 库的主要概念🤗。在课程的这一部分结束时，您将熟悉 Transformer 模型的工作原理，并知道如何使用 [Hugging Face](https://huggingface.co/models) Hub 中的模型，在数据集上对其进行微调，并在 Hub 上分享您的结果！
- 第 5 章到第 8 章在深入研究经典的 NLP 任务之前教授数据集和🤗分词器的基础知识🤗。在这一部分结束时，您将能够自己解决最常见的 NLP 问题。
- 第 9 章至第 12 章超越了 NLP，探讨了如何使用 Transformer 模型来处理语音处理和计算机视觉中的任务。在此过程中，您将学习如何构建和共享模型的演示，并针对生产环境对其进行优化。在完成这一部分时，您将准备好将 Transformer 应用于🤗（几乎）任何机器学习问题！

Comment: 吾人把重点放在如何使用transformer处理NLP任务，最后几章处理CV任务的不用看。

### 1.1.3 我们是谁？

### 1.1.4 常见问题

### 1.1.5 我们走吧

你准备好了吗？在本章中，您将了解：

- 如何使用该函数解决文本生成和分类等 NLP 任务`pipeline()`
- 关于 Transformer 架构
- 如何区分编码器、解码器和编码器-解码器架构和用例

## 1.2 Natural Language Processing

https://huggingface.co/learn/nlp-course/chapter1/2?fw=pt

### 1.2.1 什么是NLP？

以下是常见的 NLP 任务列表，以及每个任务的一些示例：

- **对整个句子进行分类**：获取评论的情绪，检测电子邮件是否是垃圾邮件，确定句子在语法上是否正确或两个句子在逻辑上是否相关
- **对句子中的每个单词进行分类**：识别句子的语法成分（名词、动词、形容词）或命名实体（人、位置、组织）
- **生成文本内容**：使用自动生成的文本完成提示，用遮罩字填充文本中的空白
- **从文本中提取**答案：给定一个问题和一个上下文，根据上下文中提供的信息提取问题的答案
- **从输入文本生成新句子**：将文本翻译成另一种语言，总结文本

### 1.2.2 为什么它具有挑战性？

## 1.3 Transformers, what can they do?

https://huggingface.co/learn/nlp-course/chapter1/3?fw=pt

Comment: 用代码示范了下列各项任务。

### 1.3.1 Transformers are everywhere!

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

### 1.3.2 Working with pipelines

Transformers 库中最🤗基本的对象是`pipeline()`函数。它将模型与其必要的预处理和后处理步骤连接起来，允许我们直接输入任何文本并获得可理解的答案：

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
classifier("I've been waiting for a HuggingFace course my whole life.")
```

[{'label': 'POSITIVE', 'score': 0.9598047137260437}]

输入也可以设为几句话！

```python
classifier(
    ["I've been waiting for a HuggingFace course my whole life.", "I hate this so much!"]
)
```

[{'label': 'POSITIVE', 'score': 0.9598047137260437},
 {'label': 'NEGATIVE', 'score': 0.9994558095932007}]

默认情况下，此管道选择一个特定的预训练模型，该模型已针对英语情绪分析进行了微调。`classifier`创建对象时，将下载并缓存模型。如果重新运行该命令，将改用缓存的模型，无需再次下载模型。

将一些文本传递到管道时，涉及三个主要步骤：

1. 文本被预处理为模型可以理解的格式。
2. 预处理的输入将传递给模型。
3. 模型的预测是经过后处理的，因此您可以理解它们。

目前[可用的一些管道](https://huggingface.co/transformers/main_classes/pipelines.html)包括：

- `feature-extraction`（获取文本的向量表示）
- `fill-mask`
- `ner`（命名实体识别）
- `question-answering`
- `sentiment-analysis`
- `summarization`
- `text-generation`
- `translation`
- `zero-shot-classification`

让我们来看看其中的一些！

### 1.3.3 Zero-shot classification

我们将从解决一项更具挑战性的任务开始，我们需要对未标记的文本进行分类。这在实际项目中很常见，因为注释文本通常很耗时，并且需要领域专业知识。对于此用例，`zero-shot-classification`管道非常强大：它允许您指定用于分类的标签，因此您不必依赖预训练模型的标签。您已经了解了该模型如何使用这两个标签将句子分类为正面或负面，但它也可以使用您喜欢的任何其他标签集对文本进行分类。

```python
from transformers import pipeline

classifier = pipeline("zero-shot-classification")
classifier(
    "This is a course about the Transformers library",
    candidate_labels=["education", "politics", "business"],
)
```

{'sequence': 'This is a course about the Transformers library',
 'labels': ['education', 'business', 'politics'],
 'scores': [0.8445963859558105, 0.111976258456707, 0.043427448719739914]}

此管道称为*零样本*，因为无需对数据进行微调模型即可使用它。它可以直接返回您想要的任何标签列表的概率分数！

### 1.3.4 Text generation

现在让我们看看如何使用管道生成一些文本。这里的主要思想是提供提示，模型将通过生成剩余的文本来自动完成它。这类似于许多手机上的预测文本功能。文本生成涉及随机性，因此如果您没有得到如下所示的相同结果，这是正常的。

```python
from transformers import pipeline

generator = pipeline("text-generation")
generator("In this course, we will teach you how to")
```

[{'generated_text': 'In this course, we will teach you how to understand and use '
                    'data flow and data interchange when handling user data. We '
                    'will be working with one or more of the most commonly used '
                    'data flows — data flows of various types, as seen by the '
                    'HTTP'}]

您可以使用`num_return_sequences`参数控制生成多少个不同的序列，并使用`max_length`参数控制输出文本的总长度。

Comment: 感觉和吾人相关的也就文本生成而已。（翻译可能算半个）

Comment 1:  通过 `generator = pipeline("text-generation", model="gpt2-large", device="cuda")` 可以设置device, 通过 `print("Pipeline device:", generator.device)` 可以输出当前使用的device. (2024年2月10日)

### 1.3.5 Using any model from the Hub in a pipeline

前面的示例使用手头任务的默认模型，但您也可以从 Hub 中选择特定模型，以便在管道中用于特定任务（例如，文本生成）。转到模型中心，然后单击左侧的相应标签，以仅显示该任务支持的[模型](https://huggingface.co/models)。你应该进入这样的[页面。](https://huggingface.co/models?pipeline_tag=text-generation)

让我们试试 [`distilgpt2`](https://huggingface.co/distilgpt2) 模型！下面介绍如何在与以前相同的管道中加载它：

```python
from transformers import pipeline

generator = pipeline("text-generation", model="distilgpt2")
generator(
    "In this course, we will teach you how to",
    max_length=30,
    num_return_sequences=2,
)
```

[{'generated_text': 'In this course, we will teach you how to manipulate the world and '
                    'move your mental and physical capabilities to your advantage.'},
 {'generated_text': 'In this course, we will teach you how to become an expert and '
                    'practice realtime, and with a hands on experience on both real '
                    'time and real'}]

您可以通过单击语言标记来优化对模型的搜索，并选择将生成另一种语言文本的模型。模型中心甚至包含支持多种语言的多语言模型的检查点。

通过单击选择模型后，您会看到有一个小部件可让您直接在线试用。这样，您可以在下载模型之前快速测试模型的功能。

#### 推理 API

所有模型都可以使用推理 API 直接通过浏览器进行测试，该 API 可在 Hugging Face [网站上](https://huggingface.co/)找到。您可以通过输入自定义文本并观察模型处理输入数据，直接在此页面上使用模型。

为小组件提供支持的推理 API 也可作为付费产品使用，如果您的工作流程需要它，它会派上用场。有关更多详细信息，请参阅[定价页面](https://huggingface.co/pricing)。

Comment: 好多模型的推理API被关闭或者不支持免费节点，支持的包括：

+ bloom  [bigscience/bloom · Hugging Face](https://huggingface.co/bigscience/bloom)
+ bloom-560m  [bigscience/bloom-560m · Hugging Face](https://huggingface.co/bigscience/bloom-560m) 
+ StarCoder  [bigcode/starcoder · Hugging Face](https://huggingface.co/bigcode/starcoder)  这是一个代码模型。
+ Mistral-7B-v0.1  [mistralai/Mistral-7B-v0.1 · Hugging Face](https://huggingface.co/mistralai/Mistral-7B-v0.1)
+ GPT-2  [gpt2 · Hugging Face](https://huggingface.co/gpt2)
+ distilgpt2  [distilgpt2 · Hugging Face](https://huggingface.co/distilgpt2)  

模型的大小可以从Files and versions选项卡中看出。

Comment: 有意思的是，在hugging face官网上，bert对应的标签是fill-mask而不是text generation. 官网给出的推理API任务也是fill-mask. (2024年1月7日)

### 1.3.6 fill-mask 完形填空

您将尝试的下一个管道是 `fill-mask`。这个任务的想法是填补给定文本中的空白：

```python
from transformers import pipeline

unmasker = pipeline("fill-mask")
unmasker("This course will teach you all about <mask> models.", top_k=2)
```

[{'sequence': 'This course will teach you all about mathematical models.',
  'score': 0.19619831442832947,
  'token': 30412,
  'token_str': ' mathematical'},
 {'sequence': 'This course will teach you all about computational models.',
  'score': 0.04052725434303284,
  'token': 38163,
  'token_str': ' computational'}]

`top_k``<mask>`参数控制要显示多少种可能性。请注意，此处的模型填充了特殊单词，该单词通常称为*掩码标记*。其他掩码填充模型可能具有不同的掩码标记，因此在探索其他模型时，最好验证正确的掩码字。检查它的一种方法是查看小部件中使用的掩码词。

### 1.3.7 Named entity recognition 命名实体识别

### 1.3.8 Question answering 问答

### 1.3.9 summarization 摘要提取

### 1.3.10 translation 翻译

## 1.4 How do Transformers work?

https://huggingface.co/learn/nlp-course/chapter1/4?fw=pt

Comment: 仅介绍，无代码。

### 1.4.1 A bit of Transformer history

- GPT-like (also called *auto-regressive* Transformer models)
- BERT-like (also called *auto-encoding* Transformer models)
- BART/T5-like (also called *sequence-to-sequence* Transformer models)

### 1.4.2 Transformers are language models

任务的一个例子是在阅读前 *n 个*单词后预测句子中的下一个单词。这称为*因果语言建模*，因为输出取决于过去和现在的输入，而不是未来的输入。(外插)

另一个例子是掩码语言建模，其中模型预测句子中的*掩码*单词。（内插）

### 1.4.3 Transformers are big models

除了一些异常值（如DistilBERT）之外，实现更好性能的一般策略是增加模型的大小以及预训练的数据量。

### 1.4.4 Transfer Learning

fine-tune属于迁移学习。

### 1.4.5 General architecture

### 1.4.6 Introduction

预训练模型已经在与微调数据集有一些相似之处的数据集上进行训练。

- **仅编码器模型**：适用于需要理解输入的任务，例如句子分类和命名实体识别。
- **仅解码器模型**：适用于文本生成等生成任务。（这个适用于PCF）
- **编码器-解码器**模型或序列**到序列模型**：适用于需要输入的生成任务，例如翻译或摘要。

### 1.4.7 Attention layers

Transformer 模型的一个关键特征是它们使用称为注意力层的特殊*层*构建。事实上，介绍 Transformer 架构的论文的标题是[“注意力就是你所需要的一切”](https://arxiv.org/abs/1706.03762)！我们将在课程的后面探讨注意力层的细节;现在，你需要知道的是，这一层会告诉模型在处理每个单词的表示时，特别注意你传递的句子中的某些单词（或多或少忽略其他单词）。

为了将其置于上下文中，请考虑将文本从英语翻译成法语的任务。给定输入“你喜欢这门课程”，翻译模型还需要注意相邻的单词“你”，以获得单词“喜欢”的正确翻译，因为在法语中，动词“喜欢”的变位因主语而异。然而，句子的其余部分对该词的翻译没有用处。同样，在翻译“this”时，模型还需要注意“course”这个词，因为“this”的翻译方式会有所不同，具体取决于相关名词是阳性还是阴性。同样，句子中的其他单词对于“this”的翻译无关紧要。对于更复杂的句子（以及更复杂的语法规则），模型需要特别注意可能在句子中出现得更远的单词，以便正确翻译每个单词。

同样的概念也适用于与自然语言相关的任何任务：一个词本身有一个含义，但这个意义深受上下文的影响，上下文可以是被研究的单词之前或之后的任何其他单词（或单词）。

现在您已经了解了注意力层的全部内容，让我们仔细看看 Transformer 架构。

### 1.4.8 The original architecture

Transformer 架构最初是为翻译而设计的。在训练过程中，编码器接收某种语言的输入（句子），而解码器接收所需目标语言的相同句子。在编码器中，注意力层可以使用句子中的所有单词（因为，正如我们刚才看到的，给定单词的翻译可能取决于句子中它后面和前面的内容）。但是，解码器是按顺序工作的，并且只能注意它已经翻译的句子中的单词（因此，只能注意当前生成的单词之前的单词）。例如，当我们预测了翻译目标的前三个单词时，我们将它们提供给解码器，然后解码器使用编码器的所有输入来尝试预测第四个单词。

为了在训练过程中加快速度（当模型可以访问目标句子时），解码器被输入整个目标，但不允许使用将来的单词（如果它在尝试预测位置 2 的单词时可以访问位置 2 的单词，问题就不会很困难！例如，当尝试预测第四个单词时，注意力层将只能访问位置 1 到 3 中的单词。

最初的 Transformer 架构如下所示，编码器在左侧，解码器在右侧：

![变形金刚模型的架构](https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/transformers.svg)

请注意，解码器模块中的第一个注意力层关注解码器的所有（过去）输入，但第二个注意力层使用编码器的输出。因此，它可以访问整个输入句子，以最好地预测当前单词。这非常有用，因为不同的语言可能具有将单词按不同顺序排列的语法规则，或者句子后面提供的一些上下文可能有助于确定给定单词的最佳翻译。

*注意力掩码*也可以在编码器/解码器中使用，以防止模型注意某些特殊单词，例如，在将句子批处理在一起时，用于使所有输入的长度相同的特殊填充词。

### 1.4.9 Architectures vs. checkpoints

当我们在本课程中深入研究 Transformer 模型时，您会看到*对架构*和*检查点*以及*模型*的提及。这些术语的含义略有不同：

- **架构**：这是模型的骨架，即模型中每一层和每个操作的定义。
- **检查点**：这些是将在给定体系结构中加载的权重。
- **模型**：这是一个总称，不像“架构”或“检查点”那样精确：它可以同时表示两者。本课程将在减少歧义时指定*架构*或*检查点*。

例如，BERT 是一种架构，而 Google 团队为 BERT 的第一个版本训练的一组权重`bert-base-cased`是一个检查点。但是，可以说“BERT模型”和“`bert-base-cased`模型”。

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

解码器模型仅使用转换器模型的解码器。在每个阶段，对于给定的单词，注意力层只能访问句子中位于其前面的单词。这些模型通常称为**自回归模型**。

解码器模型的预训练通常围绕预测句子中的下一个单词。

这些模型最适合涉及**文本生成**的任务。

CTRL
GPT
GPT-2
Transformer XL

comment:  完形填空是内插，自回归是外推。

## 1.7 Sequence-to-sequence models (Encoder-decoder models)

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

正如您在[第 1 章](https://huggingface.co/course/chapter1)中看到的，Transformer 模型通常非常大。由于有数百万到数百*亿*个参数，训练和部署这些模型是一项复杂的工作。此外，由于几乎每天都有新模型发布，并且每个模型都有自己的实现，因此尝试它们并非易事。

🤗 Transformers 库就是为了解决这个问题而创建的。它的目标是提供一个单一的 API，通过该 API 可以加载、训练和保存任何 Transformer 模型。该库的主要特点是：

- **易**用性：只需两行代码即可下载、加载和使用最先进的 NLP 模型进行推理。
- **灵活性**：从本质上讲，所有模型都是简单的 PyTorch 或 TensorFlow 类，可以像在各自的机器学习 （ML） 框架中的任何其他模型一样进行处理。`nn.Module``tf.keras.Model`
- **简单性**：整个库中几乎没有任何抽象。“多合一文件”是一个核心概念：模型的前向传递完全在单个文件中定义，因此代码本身是可理解和可破解的。

最后一个功能使 🤗 Transformer 与其他 ML 库完全不同。这些模型不是建立在模块之上的 跨文件共享的;相反，每个模型都有自己的层。除了使模型更易于理解之外，这还允许您轻松地在一个模型上进行试验，而不会影响其他模型。

本章将从一个端到端示例开始，在这个示例中，我们一起使用模型和分词器来复制[第 1 章](https://huggingface.co/course/chapter1)中介绍的`pipeline()`函数。接下来，我们将讨论模型 API：我们将深入探讨模型和配置类，并向您展示如何加载模型以及它如何处理数值输入以输出预测。

然后，我们将查看分词器 API，它是`pipeline()`函数的另一个主要组件。分词器负责第一个和最后一个处理步骤，处理神经网络从文本到数字输入的转换，并在需要时转换回文本。最后，我们将向您展示如何在准备好的批处理中通过模型发送多个句子，然后通过仔细研究高级`tokenizer()`函数来总结这一切。

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

让我们从一个完整的示例开始，看看我们在[第 1 章](https://huggingface.co/course/chapter1)中执行以下代码时幕后发生了什么：

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
classifier(
    [
        "I've been waiting for a HuggingFace course my whole life.",
        "I hate this so much!",
    ]
)
```

并获得：

[{'label': 'POSITIVE', 'score': 0.9598047137260437},
 {'label': 'NEGATIVE', 'score': 0.9994558095932007}]

正如我们在第 [1 章](https://huggingface.co/course/chapter1)中看到的，此管道将三个步骤组合在一起：预处理、通过模型传递输入和后处理：

![完整的 NLP 管道：文本的标记化、到 ID 的转换以及通过 Transformer 模型和模型头进行推理。](https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter2/full_nlp_pipeline.svg)

让我们快速回顾一下其中的每一个。

Comment: Raw text(This course is amazing) -> (Tokenizer) -> input IDs([101, 2023, 2607, 2003, 6429, 999, 102]) -> (Model) -> Logits([-4.36, 4,68]) -> (Post Processing) -> predictions(Positive 99%, Negative 1%)

### Q: huggingface AutoModel是干啥的？

`AutoModel` 是 Hugging Face Transformers 库中的一个类，它是一个方便的工具，用于根据指定的模型名称或 checkpoint 来自动选择和加载相应的预训练模型。

在深度学习中，使用预训练模型是一种常见的方式，可以为各种自然语言处理（NLP）任务提供强大的基础。然而，Hugging Face Transformers 库提供了许多不同类型的预训练模型（例如 BERT、GPT、RoBERTa 等），而每个模型可能有多个不同的变体和配置。为了方便用户加载和使用不同模型，`AutoModel` 类允许用户通过一个简单的 API 来加载预训练模型，而无需手动选择特定的模型或配置。

`AutoModel` 的主要作用是**根据给定的模型名称或 checkpoint 来自动选择和加载对应的预训练模型**。这样，你只需要指定模型名称或 checkpoint，`AutoModel` 就会自动选择和加载与之对应的预训练模型。

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

### 2.2.1 Preprocessing with a tokenizer

与其他神经网络一样，Transformer 模型无法直接处理原始文本，因此我们管道的第一步是将文本输入转换为模型可以理解的数字。为此，我们使用一个**分词器**，它将负责：

- 将输入拆分为称为标记的单词、子单词或符号（如标点符号*）*
- 将每个令牌映射到整数
- 添加可能对模型有用的其他输入

所有这些预处理都需要以与预训练模型时完全相同的方式完成，因此我们首先需要从[模型中心](https://huggingface.co/models)下载该信息。为此，我们使用`AutoTokenizer`类及`from_pretrained()`方法。使用模型的检查点名称，它将自动获取与模型的分词器关联的数据并缓存它（因此仅在您第一次运行下面的代码时下载它）。

由于管道`sentiment-analysis`的默认检查点是`distilbert-base-uncased-finetuned-sst-2-english`（您可以[在此处](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english)查看其模型卡），因此我们运行以下命令：

```python
from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
```

一旦我们有了分词器，我们就可以直接将我们的句子传递给它，我们将得到一个准备提供给我们的模型的字典！剩下唯一要做的就是将输入 ID 列表转换为张量。

您可以使用 🤗 Transformers，而不必担心将哪个 ML 框架用作后端;对于某些模型，它可能是 PyTorch 或 TensorFlow，或者是 Flax。但是，Transformer 模型只接受*张量*作为输入。如果这是你第一次听说张量，你可以把它们看作是 NumPy 数组。NumPy 数组可以是标量 （0D）、向量 （1D）、矩阵 （2D） 或具有更多维度。它实际上是一个张量;其他 ML 框架的张量行为类似，通常与 NumPy 数组一样易于实例化。

为了指定我们想要返回的张量类型（PyTorch、TensorFlow 或普通 NumPy），我们使用参数`return_tensors`：

```python
raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
print(inputs)
```

暂时不要担心填充和截断;我们稍后会解释这些。这里要记住的主要事情是，你可以传递一个句子或一个句子列表，以及指定你想要返回的张量类型（如果没有传递类型，你会得到一个列表列表）。

以下是 PyTorch 张量的结果：

{
    'input_ids': tensor([
        [  101,  1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172, 2607,  2026,  2878,  2166,  1012,   102],
        [  101,  1045,  5223,  2023,  2061,  2172,   999,   102,     0,     0,     0,     0,     0,     0,     0,     0]
    ]), 
    'attention_mask': tensor([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    ])
}

输出本身是一个字典，包含两个键`input_ids`和 `attention_mask`. `input_ids`包含两行整数（每个句子一个），它们是每个句子中标记的唯一标识符。我们将在本章后面解释`attention_mask`是什么。

comment:  PCF模型可以用分词器来预处理PCF生成的xml。

### 2.2.2 Going through the model

我们可以像使用分词器一样下载预训练模型。🤗 Transformers 提供了一个`AutoModel`类，该类也有一个`from_pretrained()`方法：

```python
from transformers import AutoModel

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModel.from_pretrained(checkpoint)
```

在此代码片段中，我们下载了之前在管道中使用的相同检查点（它实际上应该已经缓存），并用它实例化了模型。

此体系结构仅包含基本的 Transformer 模块：给定一些输入，它会输出我们所说的*隐藏状态*，也称为*功能*。对于每个模型输入，我们将检索一个高维向量，表示 **Transformer 模型对该输入的上下文理解**。

如果这没有意义，请不要担心。我们稍后会解释这一切。

虽然这些隐藏状态本身很有用，但它们通常是模型另一部分（称为*头部*）的输入。在第 [1 章](https://huggingface.co/course/chapter1)中，可以使用相同的架构执行不同的任务，但每个任务都有与之关联的不同头。

### 2.2.3 A high-dimensional vector?

Transformer 模块输出的矢量通常很大。它通常有三个维度：

- **批量大小**：一次处理的序列数（在我们的示例中为 2 个）。
- 序列长度：序列的数字表示的**长度**（在我们的示例中为 16）。
- **隐藏大小**：每个模型输入的向量维度。

由于最后一个值，它被称为“高维”。隐藏尺寸可能非常大（768 对于较小的型号很常见，在较大的型号中可以达到 3072 或更多）。

如果我们将预处理的输入馈送到模型中，我们可以看到这一点：

```python
outputs = model(**inputs)
print(outputs.last_hidden_state.shape)
```

torch.Size([2, 16, 768])

请注意，变形金刚模型的🤗输出行为类似于 `namedtuple`或字典。你可以通过属性（就像我们所做的那样）或键`outputs["last_hidden_state"]`来访问元素，如果你确切地知道你要找的东西在哪里（`outputs[0]`），甚至可以通过索引来访问元素。

Comment: inputs是

{
    'input_ids': tensor([
        [  101,  1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172, 2607,  2026,  2878,  2166,  1012,   102],
        [  101,  1045,  5223,  2023,  2061,  2172,   999,   102,     0,     0,     0,     0,     0,     0,     0,     0]
    ]), 
    'attention_mask': tensor([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    ])
}

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

### Q: outputs.last_hidden_state是在logits前一层吗？

是的，`outputs.last_hidden_state` 在 `logits` 前一层。在 Transformers 模型中，`outputs.last_hidden_state` 包含了模型的中间表示，也称为上下文向量（contextual embeddings）或特征向量（feature embeddings）。它是模型在输入序列上进行处理后的最终输出，记录了输入序列中每个位置（单词或子词）的编码表示。

`outputs.last_hidden_state` 的维度是 `[batch_size, sequence_length, hidden_size]`，其中 `batch_size` 表示批量大小，`sequence_length` 表示序列的长度，`hidden_size` 表示隐藏层的大小（即每个位置的特征向量维度）。

`logits` 则是在 `outputs.last_hidden_state` 基础上通过模型的最后一层或特定输出层得到的预测输出。对于分类任务，`logits` 是模型在每个类别上的分数或得分，表示输入属于每个类别的概率。`logits` 的维度通常是 `[batch_size, num_classes]`，其中 `num_classes` 表示分类任务中的类别数。

在文本分类任务中，通常会使用 `outputs.last_hidden_state` 来获取输入序列的表示，然后通过一些后续的操作（例如平均池化、最大池化等）来获得整个序列的固定表示。然后，这个固定表示可以作为输入传递给分类层，生成最终的 `logits`，并通过 softmax 函数获得分类概率。

总结：
- `outputs.last_hidden_state` 是模型的中间表示，包含输入序列中每个位置的编码表示。
- `logits` 是模型的预测输出，是在 `outputs.last_hidden_state` 基础上得到的，并用于计算类别的得分或概率分布。

Comment: 其实last_hidden_state的维数远远大于logits的维数。

### 2.2.4 Model heads: Making sense out of numbers

模型头将隐藏状态的高维向量作为输入，并将它们投影到不同的维度上。它们通常由一个或几个线性层组成：

![一个 Transformer 网络，旁边有一个 Transformer。](https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter2/transformer_and_head.svg)

Transformer 模型的输出直接发送到要处理的模型头。

在此图中，模型由其嵌入层和后续层表示。嵌入层将标记化输入中的每个输入 ID 转换为表示关联标记的向量。随后的层使用注意力机制操纵这些向量，以产生句子的最终表示。

变形金刚中有🤗许多不同的架构，每个架构都围绕处理特定任务而设计。以下是非详尽列表：

- `*Model`（检索隐藏状态）
- `*ForCausalLM`
- `*ForMaskedLM`
- `*ForMultipleChoice`
- `*ForQuestionAnswering`
- `*ForSequenceClassification`
- `*ForTokenClassification`
- 🤗 和其他人

在我们的示例中，我们需要一个带有序列分类头的模型（以便能够将句子分类为正句或负句）。因此，我们实际上不会使用`AutoModel`类，而是`AutoModelForSequenceClassification`：

```python
from transformers import AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model(**inputs)
```

现在，如果我们看一下输出的形状，维数会低得多：模型头部将我们之前看到的高维向量作为输入，并输出包含两个值（每个标签一个）的向量：

```python
print(outputs.logits.shape)
```

torch.Size([2, 2])

由于我们只有两个句子和两个标签，因此我们从模型中得到的结果是 2 x 2 的形状。

### Q: 什么是transformer中的model head?

结合上下文，说的应该是model的功能（问答、多选、完形填空）

### 2.2.5 Postprocessing the output

我们从模型中获得的输出值本身并不一定有意义。让我们来看看：

```python
print(outputs.logits)
```

tensor([[-1.5607,  1.6123], 
              [ 4.1692, -3.3464]], grad_fn=<AddmmBackward>)

我们的模型预测了第一句话和第二句话。这些不是概率，而是*logits*，即模型最后一层输出的原始非归一化分数。要转换为概率，它们需要经过一个 SoftMax 层（所有 🤗 Transformer 模型都输出对数，因为用于训练的损失函数通常会将最后一个激活函数（如 [SoftMax](https://en.wikipedia.org/wiki/Softmax_function)）与实际损失函数（如交叉熵）融合）：`[-1.5607, 1.6123]``[ 4.1692, -3.3464]`

```python
import torch

predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)
```

tensor([[4.0195e-02, 9.5980e-01],
              [9.9946e-01, 5.4418e-04]], grad_fn=<SoftmaxBackward>)

现在我们可以看到模型预测了第一句话`[0.0402, 0.9598]`和第二句话`[0.9995, 0.0005]`。这些是可识别的概率分数。

为了获得每个位置对应的标签，我们可以检查模型配置的`id2label`属性（下一节会详细介绍）：

```python
model.config.id2label
```

{0: 'NEGATIVE', 1: 'POSITIVE'}

现在我们可以得出结论，该模型预测了以下内容：

- 第一句话：否定：0.0402，正面：0.9598
- 第二句：否定：0.9995，正：0.0005

我们已经成功地重现了管道的三个步骤：使用分词器进行预处理(comment:  word to vector)、通过模型传递输入和后处理(comment:  vector to label)！现在，让我们花一些时间更深入地了解其中的每一个步骤。

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

在本节中，我们将仔细研究如何创建和使用模型。我们将使用`AutoModel`类，当您想从检查点实例化任何模型时，该类非常方便。

`AutoModel`类及其所有亲戚实际上是库中各种可用模型的简单包装器。这是一个聪明的包装器，因为它可以自动猜测检查点的适当模型架构，然后使用此架构实例化模型。

但是，如果您知道要使用的模型类型，则可以直接使用定义其体系结构的类。让我们来看看它是如何与BERT模型一起工作的。

### 2.3.1 Creating a Transformer

初始化 BERT 模型，我们需要做的第一件事是加载一个配置对象：

```python
from transformers import BertConfig, BertModel

# Building the config
config = BertConfig()

# Building the model from the config
model = BertModel(config)
```

该配置包含许多用于构建模型的属性：

```python
print(config)
```

BertConfig {
  [...]
  "hidden_size": 768,
  "intermediate_size": 3072,
  "max_position_embeddings": 512,
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  [...]
}

虽然您还没有看到所有这些属性的作用，但您应该认识到其中的一些：`hidden_size`属性定义了矢量`hidden_states`的大小，`num_hidden_layers`定义了 Transformer 模型的层数。

comment:  从中可以看出，BertModel还是比较大的。

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
    vocab_size=30522, # 词汇表的大小，即 BERT 模型支持的唯一词汇数量。
    hidden_size=768, # BERT 模型的隐藏层大小（即每个位置的特征向量维度）。
    num_hidden_layers=12, # BERT 模型的隐藏层数量。
    num_attention_heads=12, # BERT 模型的注意力头（attention head）数量。
    intermediate_size=3072, # BERT 模型中全连接层的中间层大小。
    hidden_act="gelu", # 选择在 BERT 模型中使用的激活函数。默认是 GELU（Gaussian Error Linear Unit）
    hidden_dropout_prob=0.1, # 隐藏层的 dropout 概率，用于模型训练时的随机失活。
    attention_probs_dropout_prob=0.1, # 注意力层的 dropout 概率，用于模型训练时的随机失活。
    max_position_embeddings=512, # 输入序列的最大长度（最大位置编码）。
    type_vocab_size=2, # 类型词汇表的大小，用于区分单句和双句输入。
    initializer_range=0.02 # 初始化权重的范围。
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

### 2.3.2 Different loading methods

从默认配置创建模型会使用随机值对其进行初始化：

```python
from transformers import BertConfig, BertModel

config = BertConfig()
model = BertModel(config)

# Model is randomly initialized!
```

该模型可以在这种状态下使用，但它会输出乱码;它需要先进行训练。我们可以在手头的任务上从头开始训练模型，但正如您在[第 1 章](https://huggingface.co/course/chapter1)中看到的那样，这将需要很长时间和大量数据，并且对环境的影响不可忽视。为了避免不必要和重复的工作，必须能够共享和重用已经训练过的模型。

加载一个已经训练好的 Transformer 模型很简单——我们可以使用`from_pretrained()`方法做到这一点：

```python
from transformers import BertModel

model = BertModel.from_pretrained("bert-base-cased")
```

正如您之前所看到的，我们可以用等效的`BertModel`类替换`AutoModel`。从现在开始，我们将这样做，因为这会生成与检查点无关的代码;如果您的代码适用于一个检查点，则它应该与另一个检查点无缝协作。即使体系结构不同，只要检查点针对类似任务（例如，情绪分析任务）进行了训练，这也适用。

在上面的代码示例中，我们没有使用 `BertConfig`，而是通过`bert-base-cased`标识符加载了预训练模型。这是一个模型检查点，由 BERT 的作者自己训练;您可以在其[模型卡](https://huggingface.co/bert-base-cased)中找到有关它的更多详细信息。

现在，此模型已使用检查点的所有权重进行初始化。它可以直接用于对训练任务进行推理，也可以对新任务进行微调。通过使用预先训练的权重而不是从头开始训练，我们可以快速取得良好的结果。

权重已下载并缓存在缓存文件夹中（因此将来对`from_pretrained()`方法的调用不会重新下载它们），该文件夹默认为 *~/.cache/huggingface/transformers*。您可以通过设置环境变量来自定义缓存文件夹。`HF_HOME`

用于加载模型的标识符可以是模型中心上任何模型的标识符，只要它与 BERT 架构兼容即可。可以[在此处](https://huggingface.co/models?filter=bert)找到可用 BERT 检查点的完整列表。

### 2.3.3 Saving methods

保存模型就像加载模型一样简单 - 我们使用的方法类似于该方法：`save_pretrained()``from_pretrained()`

```python
model.save_pretrained("directory_on_my_computer")
```

这会将两个文件保存到磁盘：

```python
ls directory_on_my_computer

config.json pytorch_model.bin
```

如果查看 *config.json* 文件，您将认识到构建模型体系结构所需的属性。此文件还包含一些元数据，例如检查点的来源以及🤗上次保存检查点时使用的 Transformers 版本。

*pytorch_model.bin*文件称为*状态字典*;它包含模型的所有权重。这两个文件齐头并进;配置对于了解模型的架构是必要的，而模型权重是模型的参数。

### 2.3.4 Using a Transformer model for inference

现在您已经知道如何加载和保存模型，让我们尝试使用它来进行一些预测。Transformer 模型只能处理数字 — 分词器生成的数字。但在讨论分词器之前，让我们先探讨一下模型接受哪些输入。

分词器可以负责将输入转换为相应框架的张量，但为了帮助您了解正在发生的事情，我们将快速了解在将输入发送到模型之前必须执行的操作。

假设我们有几个序列：

```python
sequences = ["Hello!", "Cool.", "Nice!"]
```

分词器将这些转换为词汇索引，这些索引通常称为*输入 ID*。每个序列现在都是一个数字列表！结果输出为：

encoded_sequences = [
    [101, 7592, 999, 102],
    [101, 4658, 1012, 102],
    [101, 3835, 999, 102],
]

这是编码序列的列表：列表的列表。张量只接受矩形（想想矩阵）。这个“数组”已经是矩形的，所以把它转换为张量很容易：

```python
import torch

model_inputs = torch.tensor(encoded_sequences)
```

### 2.3.5 Using the tensors as inputs to the model

在模型中使用张量非常简单——我们只需使用输入调用模型：

```python
output = model(model_inputs)
```

虽然模型接受许多不同的参数，但只有输入 ID 是必需的。我们将解释其他参数的作用以及稍后何时需要它们， 但首先，我们需要仔细研究构建 Transformer 模型可以理解的输入的分词器。

## 2.4 Tokenizers

https://huggingface.co/learn/nlp-course/chapter2/4?fw=pt

分词器是 NLP 管道的核心组件之一。它们有一个目的：**将文本转换为可由模型处理的数据**。模型只能处理数字，因此分词器需要将我们的文本输入转换为数字数据。在本节中，我们将探讨令牌化管道中发生的情况。

在 NLP 任务中，通常处理的数据是原始文本。下面是此类文本的示例：

```
Jim Henson was a puppeteer
```

但是，模型只能处理数字，因此我们需要找到一种方法将原始文本转换为数字。这就是分词器所做的，有很多方法可以解决这个问题。目标是找到最有意义的表示形式，即对模型最有意义的表示形式，如果可能的话，找到最小的表示形式。

让我们看一下标记化算法的一些示例，并尝试回答您可能遇到的有关标记化的一些问题。

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

### 2.4.1 Word-based

如果我们想用基于单词的分词器完全覆盖一种语言，我们需要为语言中的每个单词都有一个标识符，这将生成大量的标记。例如，英语中有超过 500，000 个单词，因此要构建从每个单词到输入 ID 的映射，我们需要跟踪这么多 ID。此外，像“dog”这样的单词与像“dogs”这样的单词表示不同，并且模型最初无法知道“dog”和“dogs”是相似的：它会将这两个单词识别为不相关的。这同样适用于其他类似的单词，如“run”和“running”，模型最初不会看到它们相似。

最后，我们需要一个自定义标记来表示不在词汇表中的单词。这称为“未知”标记，通常表示为“[UNK]”或“”。如果您看到分词器正在生成大量这些标记，这通常是一个不好的迹象，因为它无法检索单词的合理表示，并且您在此过程中丢失了信息。制作词汇表时的目标是以这样的方式进行，即分词器将尽可能少的单词标记到未知标记中。

### 2.4.2 Character-based

基于字符的分词器将文本拆分为字符，而不是单词。这有两个主要好处：

- 词汇量要小得多。
- 词汇外（未知）标记要少得多，因为每个单词都可以由字符构建。

但这里也出现了一些关于空格和标点符号的问题：

![基于字符的标记化示例。](https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter2/character_based_tokenization.svg)

这种方法也不完美。由于现在的表示是基于字符而不是单词，因此人们可以争辩说，从直觉上讲，它的意义不大：每个字符本身并没有太大的意义，而单词就是这种情况。但是，这又因语言而异;例如，在中文中，每个字符比拉丁语中的字符携带更多的信息。

另一件需要考虑的事情是，我们的模型最终将处理大量的标记：虽然一个单词只是一个带有基于单词的标记器的标记，但当转换为字符时，它可以很容易地变成 10 个或更多标记。

为了两全其美，我们可以使用结合这两种方法的第三种技术：*子词标记化*。

### 2.4.3 Subword tokenization

子词标记化算法所依据的原则是，常用词不应拆分为较小的子词，而应将生僻词分解为有意义的子词。

例如，“烦人地”可能被认为是一个罕见的词，可以分解为“烦人”和“ly”。它们都可能更频繁地作为独立的子词出现，而同时“烦人”的含义由“烦人”和“ly”的复合含义保留。

下面是一个示例，显示了子词标记化算法如何标记序列“Let's do tokenization！”：

![A subword tokenization algorithm.](https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter2/bpe_subword.svg)

这些子词最终提供了大量的语义含义：例如，在上面的例子中，“tokenization”被拆分为“token”和“ization”，这两个词具有语义意义，同时又节省空间（只需要两个词来表示一个长词）。这使我们能够在小词汇量下获得相对良好的覆盖率，并且几乎没有未知的标记。

这种方法在凝集语言（如土耳其语）中特别有用，您可以在其中通过将子词串在一起来形成（几乎）任意长的复杂单词。

### 2.4.4 And more!

不出所料，还有更多的技术。仅举几例：

- GPT-2 中使用的字节级 BPE
- WordPiece，用于 BERT
- SentencePiece 或 Unigram，用于多个多语言模型

现在，您应该对分词器的工作原理有足够的了解，以便开始使用 API。

### 2.4.5 Loading and saving

加载和保存分词器与使用模型一样简单。实际上，它基于相同的两种方法：`from_pretrained()`和 `save_pretrained()`. 这些方法将加载或保存分词器使用的算法（有点像模型的*架构*）及其词汇表（有点像模型的*权重*）。

加载使用与 BERT 相同的检查点训练的 BERT 分词器与加载模型的方式相同，只是我们使用类：`BertTokenizer`

```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
```

与`AutoModel`类似，`AutoTokenizer`类将根据检查点名称在库中获取适当的 tokenizer 类，并且可以直接与任何检查点一起使用：

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
```

现在，我们可以使用分词器，如上一节所示：

```python
tokenizer("Using a Transformer network is simple")
```

{'input_ids': [101, 7993, 170, 11303, 1200, 2443, 1110, 3014, 102],
 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0],
 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1]}

保存分词器与保存模型相同：

```python
tokenizer.save_pretrained("directory_on_my_computer")
```

我们将在第 [3 章](https://huggingface.co/course/chapter3)中详细讨论`token_type_ids`，稍后我们将解释`attention_mask` key。首先，让我们看看`input_ids`是如何生成的。为此，我们需要查看分词器的中间方法。

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

### 2.4.6 Encoding

将文本转换为数字称为*编码*。编码分两步完成：标记化，然后转换为输入 ID。

正如我们所看到的，第一步是将文本拆分为单词（或单词的一部分、标点符号等），通常称为*标记*。有多个规则可以控制该过程，这就是为什么我们需要使用模型的名称实例化分词器，以确保我们使用与预训练模型时相同的规则。

第二步是将这些标记转换为数字，这样我们就可以用它们构建一个张量并将它们提供给模型。为此，分词器有一个*词汇表*，这是我们使用`from_pretrained()`方法实例化它时下载的部分。同样，我们需要使用与模型预训练时相同的词汇表。

为了更好地理解这两个步骤，我们将分别探讨它们。请注意，我们将使用一些单独执行部分标记化管道的方法，以显示这些步骤的中间结果，但实际上，您应该直接在输入上调用标记器（如第 2 节所示）。

### 2.4.7 Tokenization

标记化过程是通过标记器的方法完成的：`tokenize()`

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

sequence = "Using a Transformer network is simple"
tokens = tokenizer.tokenize(sequence)

print(tokens)
```

此方法的输出是字符串或标记的列表：

['Using', 'a', 'transform', '##er', 'network', 'is', 'simple']

这个分词器是一个子词分词器：它拆分单词，直到它获得可以由其词汇表表示的分词。`transformer`就是这种情况，它被拆分为两个标记：`transform`和`##er` 。

### 2.4.8 From tokens to input IDs

到输入 ID 的转换由`convert_tokens_to_ids()`  tokenizer 方法处理：

```python
ids = tokenizer.convert_tokens_to_ids(tokens)

print(ids)
```

[7993, 170, 11303, 1200, 2443, 1110, 3014]

这些输出一旦转换为适当的框架张量，就可以用作模型的输入，如本章前面所述。

### 2.4.9 Decoding

*解码*则相反：从词汇索引中，我们想要得到一个字符串。这可以通过以下方法完成：`decode()`

```python
decoded_string = tokenizer.decode([7993, 170, 11303, 1200, 2443, 1110, 3014])
print(decoded_string)
```

'Using a Transformer network is simple'

请注意，`decode`方法不仅将索引转换回标记，而且还将属于相同单词的标记组合在一起以生成可读的句子。当我们使用预测新文本的模型（从提示生成的文本，或用于翻译或摘要等序列到序列问题）时，此行为将非常有用。

到现在为止，您应该了解分词器可以处理的原子操作：分词化、转换为 ID 以及将 ID 转换回字符串。然而，我们只是刮掉了冰山一角。在下一节中，我们将采用我们的方法来克服其局限性，并了解如何克服它们。

### Q: AutoTokenizer是干啥的？

`AutoTokenizer` 是 Hugging Face Transformers 库中的一个类，它是一个自动加载预训练模型的分词器（Tokenizer）的工具类。在自然语言处理（NLP）中，分词器用于将输入文本（句子、段落等）拆分成单词或子词的序列，以便机器学习模型能够处理和理解文本。

`AutoTokenizer` 的主要作用是根据给定的模型名称或 checkpoint 来自动选择和加载对应的预训练模型的分词器。这样，你可以通过一个简单的 API 调用来加载不同模型的分词器，而不需要手动指定特定模型的分词器。

使用 `AutoTokenizer` 有以下几个优点：

1. 自动选择模型：无需手动指定模型名称，`AutoTokenizer` 会根据**提供的模型名称自动选择和加载**对应的分词器。

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

### Q: 已知tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")，那么函数tokenizer有哪些参数？

`AutoTokenizer.from_pretrained("bert-base-uncased")` 返回的是一个名为 `tokenizer` 的 `AutoTokenizer` 类的实例。`tokenizer` 是用于处理文本数据，将文本转换为模型可以理解的输入格式（例如，token IDs、attention masks 等）。

以下是部分常见的 `tokenizer` 函数参数：

1. `text`: 要处理的文本数据。

2. `padding`: 是否进行填充，如果为 `True`，会根据最长的序列进行填充。

3. `truncation`: 是否进行截断，如果为 `True`，会根据最长的序列进行截断。

4. `return_tensors`: 指定返回的数据类型，如 `"pt"` 表示返回 PyTorch 张量。

5. `max_length`: 设置最大序列长度，超过这个长度会进行截断或填充。

6. `add_special_tokens`: 是否在文本的开始和结束添加特殊的 tokens，如 `[CLS]` 和 `[SEP]`。

7. `return_attention_mask`: 是否返回注意力掩码（attention mask），用于标识 padding 部分。

8. `return_token_type_ids`: 是否返回 token type IDs，用于区分不同的句子。

9. `return_overflowing_tokens`: 是否返回超过最大长度的 token。

10. `return_special_tokens_mask`: 是否返回特殊 tokens 的掩码。

11. `return_offsets_mapping`: 是否返回每个 token 在原始文本中的偏移。

等等...

这些参数可能会因不同的 tokenizer 和模型而有所不同。要查看完整的参数列表和说明，可以查阅相关的文档或源代码。

## 2.5 Handling multiple sequences

https://huggingface.co/learn/nlp-course/chapter2/5?fw=pt

在上一节中，我们探讨了最简单的用例：对小长度的单个序列进行推理。但是，已经出现了一些问题：

- 我们如何处理多个序列？
- 我们如何处理*多个不同长度的*序列？
- 词汇索引是允许模型正常工作的唯一输入吗？
- 有没有序列太长这样的事情？

让我们看看这些问题会带来什么样的问题，以及如何使用 🤗 Transformers API 解决这些问题。

### 2.5.1 Models expect a batch of inputs

在上一个练习中，您了解了如何将序列转换为数字列表。让我们将这个数字列表转换为张量并将其发送到模型：

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence = "I've been waiting for a HuggingFace course my whole life."

tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)
input_ids = torch.tensor(ids)
# This line will fail.
model(input_ids)
```

IndexError: Dimension out of range (expected to be in range of [-1, 0], but got 1)

哦不！为什么会失败？“我们遵循了第 2 节中管道中的步骤。

问题在于我们向模型发送了一个序列，而 🤗 Transformer 模型默认需要多个句子。在这里，我们尝试在将分词器应用于`sequence` .但如果你仔细观察，你会发现分词器不仅将输入 ID 列表转换为张量，还在它上面添加了一个维度：

```python
tokenized_inputs = tokenizer(sequence, return_tensors="pt")
print(tokenized_inputs["input_ids"])
```

tensor([[  101,  1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172,
          2607,  2026,  2878,  2166,  1012,   102]])

让我们再试一次，添加一个新维度：

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence = "I've been waiting for a HuggingFace course my whole life."

tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)

input_ids = torch.tensor([ids])
print("Input IDs:", input_ids)

output = model(input_ids)
print("Logits:", output.logits)
```

我们打印输入 ID 以及生成的日志 — 这是输出：

Input IDs: [[ 1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172,  2607, 2026,  2878,  2166,  1012]]
Logits: [[-2.7276,  2.8789]]

*批处理*是通过模型一次发送多个句子的行为。如果只有一个句子，则可以只使用单个序列构建一个批处理：

```python
batched_ids = [ids, ids]
```

这是一批两个相同的序列！

批处理允许模型在向其提供多个句子时工作。使用多个序列就像使用单个序列构建批处理一样简单。不过，还有第二个问题。当您尝试将两个（或更多）句子批处理在一起时，它们的长度可能不同。如果您以前使用过张量，您就会知道它们必须是矩形的，因此您将无法将输入 ID 列表直接转换为张量。为了解决这个问题，我们通常会*填充*输入。

### 2.5.2 Padding the inputs

以下列表列表无法转换为张量：

```python
batched_ids = [
    [200, 200, 200],
    [200, 200]
]
```

为了解决这个问题，我们将使用*填充*来使我们的张量具有矩形。填充通过向值较少的句子添加一个称为*填充标记*的特殊单词来确保我们所有的句子都具有相同的长度。例如，如果您有 10 个句子包含 10 个单词，1 个句子包含 20 个单词，则填充将确保所有句子都有 20 个单词。在我们的示例中，生成的张量如下所示：

```python
padding_id = 100

batched_ids = [
    [200, 200, 200],
    [200, 200, padding_id],
]
```

填充令牌 ID 可在 `tokenizer.pad_token_id`中找到。让我们使用它，将我们的两个句子单独发送到模型中，并一起批处理：

```python
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence1_ids = [[200, 200, 200]]
sequence2_ids = [[200, 200]]
batched_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id],
]

print(model(torch.tensor(sequence1_ids)).logits)
print(model(torch.tensor(sequence2_ids)).logits)
print(model(torch.tensor(batched_ids)).logits)
```

tensor([[ 1.5694, -1.3895]], grad_fn=\<AddmmBackward>)
tensor([[ 0.5803, -0.4125]], grad_fn=\<AddmmBackward>)
tensor([[ 1.5694, -1.3895],
        [ 1.3373, -1.2163]], grad_fn=\<AddmmBackward>)

我们批量预测中的对数有问题：第二行应该与第二句话的对数相同，但我们的值完全不同！

这是因为 Transformer 模型的关键特征是将每个令牌*上下文化*的注意力层。这些将考虑填充标记，因为它们涉及序列的所有标记。为了在模型中传递不同长度的单个句子时，或者在传递具有相同句子和填充的批处理时获得相同的结果，我们需要告诉这些注意层忽略填充标记。这是通过使用注意力掩码来完成的。

Comment:  类似于computer vision中给缺失的部分填充上。也类似于python的broadcast.  填充短的句子，截断长的句子，使所有句子表示长度相同。不过也要结合Attention masks才能保证使句子不受填充层的影响。(2024年1月14日)

### 2.5.3 Attention masks

*注意力掩码*是与输入 ID 张量形状完全相同的张量，填充了 0 和 1：1 表示应该关注相应的令牌，0 表示不应该关注相应的标记（即，它们应该被模型的注意力层忽略）。

让我们用注意力掩码完成前面的例子：

```python
batched_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id],
]

attention_mask = [
    [1, 1, 1],
    [1, 1, 0],
]

outputs = model(torch.tensor(batched_ids), attention_mask=torch.tensor(attention_mask))
print(outputs.logits)
```

tensor([[ 1.5694, -1.3895],
              [ 0.5803, -0.4125]], grad_fn=\<AddmmBackward>)

对比之前

tensor([[ 1.5694, -1.3895]], grad_fn=\<AddmmBackward>)
tensor([[ 0.5803, -0.4125]], grad_fn=\<AddmmBackward>)

现在，我们得到批处理中第二个句子的相同日志。

请注意，第二个序列的最后一个值是填充 ID，即注意力掩码中的 0 值。

### Q: Attention masks是干啥的？

Attention masks在自然语言处理中是一种用于控制注意力的机制，特别是在Transformer模型中。Transformer模型是一种基于自注意力机制的神经网络，它在处理序列数据时，可以根据输入的注意力掩码（Attention masks）来决定是否忽略特定位置的信息。

在NLP任务中，输入通常是一个由单词或子词组成的序列。由于序列的长度可能不一致，我们需要使用注意力掩码来处理不同长度的输入。

具体来说，Attention masks是一个二进制张量，其大小与输入序列的大小相同。Attention masks的值通常为0或1，其中1表示对应位置的单词或子词需要被关注（参与注意力计算），0表示对应位置的单词或子词不需要被关注（忽略该位置的信息）。

在Transformer模型中，注意力掩码用于控制Self-Attention（自注意力）层的计算。Self-Attention可以将输入序列的每个位置与其他位置进行交互，计算注意力权重，并在计算过程中根据注意力掩码来忽略不需要关注的位置。

例如，在处理一个由4个单词组成的句子"Hello, how are you?"时，如果使用Attention masks，可能会得到如下掩码张量：

```
[1, 1, 1, 1]
```

这意味着在Self-Attention计算中，所有位置的信息都需要被关注。而如果处理一个由3个单词组成的句子"Hello, how are?"，使用Attention masks可能会得到如下掩码张量：

```
[1, 1, 1, 0]
```

这意味着在Self-Attention计算中，只有前3个位置的信息需要被关注，最后一个位置的信息将被忽略。

Attention masks使得Transformer模型可以有效地处理可变长度的输入序列，并且能够在不同任务中共享预训练模型，从而大大提高了模型的性能和泛化能力。

### Q: 填充tokenizer.pad_token_id的地方是否应对于attention mask的0？

是的，填充 `tokenizer.pad_token_id` 的地方通常应对应于 Attention masks 中的值为 0 的位置。

在使用 Transformer 模型进行文本序列处理时，为了处理不同长度的输入序列，通常会对输入序列进行填充（padding）使其长度统一。填充后的位置需要被忽略，因为它们实际上并不包含有效的信息，只是为了使序列长度一致。

为了在 Transformer 模型中正确处理填充的位置，我们需要使用 Attention masks 来控制注意力。Attention masks 中的值为 0 的位置对应于填充的位置，我们可以将这些位置的注意力权重设置为非常小的值或者直接忽略它们，从而使 Transformer 模型在计算注意力时不考虑这些填充位置的影响。

对于使用 Hugging Face Transformers 库中的 `tokenizer` 和 `model`，可以通过调用 `tokenizer` 的 `padding` 方法来进行填充，并生成对应的 Attention masks。在这种情况下，填充的位置会使用 `tokenizer.pad_token_id` 来表示，而 Attention masks 中的值为 0 的位置与填充位置一一对应。

示例代码：

```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# 输入文本序列
text = ["Hello, how are you?", "I am fine, thank you!"]

# 进行填充并生成 Attention masks
inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")

# 输出填充后的输入张量和对应的 Attention masks
print("Input IDs:", inputs["input_ids"])
print("Attention Masks:", inputs["attention_mask"])

# 获取模型的输出
outputs = model(**inputs)
```

在上述代码中，我们使用 `tokenizer` 对输入文本序列进行填充，并生成对应的 Attention masks。填充的位置会使用 `tokenizer.pad_token_id` 来表示，而 Attention masks 中的值为 0 的位置与填充位置一一对应。通过这样的方式，Transformer 模型可以正确处理填充位置，并忽略填充位置对最终结果的影响。

### 2.5.4 Longer sequences

对于 Transformer 模型，我们可以传递模型的序列长度是有限制的。大多数模型最多处理 512 或 1024 个令牌的序列，当要求处理更长的序列时会崩溃。此问题有两种解决方案：

- 使用具有较长受支持序列长度的模型。
- 截断序列。

模型具有不同的受支持序列长度，有些模型专门用于处理非常长的序列。[Longformer](https://huggingface.co/docs/transformers/model_doc/longformer) 就是一个例子，另一个是 [LED](https://huggingface.co/docs/transformers/model_doc/led)。如果您正在处理需要很长序列的任务，我们建议您查看这些模型。

否则，我们建议您通过指定`max_sequence_length`参数来截断序列：

```python
sequence = sequence[:max_sequence_length]
```

## 2.6 Putting it all together

https://huggingface.co/learn/nlp-course/chapter2/6?fw=pt

在最后几个部分中，我们一直在尽最大努力手动完成大部分工作。我们探讨了分词器的工作原理，并研究了分词化、转换为输入 ID、填充、截断和注意力掩码。

但是，正如我们在第 2 节中看到的，🤗 Transformers API 可以通过一个高级函数为我们处理所有这些问题，我们将在这里深入探讨。当您直接对句子调用你的 `tokenizer`时，您会得到准备通过模型的输入：

```python
from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

sequence = "I've been waiting for a HuggingFace course my whole life."

model_inputs = tokenizer(sequence)
```

在这里，`model_inputs`变量包含模型正常运行所需的一切。对于 DistilBERT，这包括输入 ID 以及注意力掩码。其他接受额外输入的模型也将具有`tokenizer`对象输出的这些输入。

正如我们将在下面的一些示例中看到的那样，这种方法非常强大。首先，它可以标记单个序列：

```python
sequence = "I've been waiting for a HuggingFace course my whole life."

model_inputs = tokenizer(sequence)
```

它还**一次处理多个序列**，而 API 没有变化：

```python
sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

model_inputs = tokenizer(sequences)
```

它可以根据以下几个目标进行padding：

```python
# Will pad the sequences up to the maximum sequence length
model_inputs = tokenizer(sequences, padding="longest")

# Will pad the sequences up to the model max length
# (512 for BERT or DistilBERT)
model_inputs = tokenizer(sequences, padding="max_length")

# Will pad the sequences up to the specified max length
model_inputs = tokenizer(sequences, padding="max_length", max_length=8)
```

它还可以**截断**序列：

```python
sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

# Will truncate the sequences that are longer than the model max length
# (512 for BERT or DistilBERT)
model_inputs = tokenizer(sequences, truncation=True)

# Will truncate the sequences that are longer than the specified max length
model_inputs = tokenizer(sequences, max_length=8, truncation=True)
```

`tokenizer`对象可以处理到特定框架张量的转换，然后可以将其直接发送到模型。例如，在以下代码示例中，我们提示分词器从不同的框架返回张量 — `"pt"`返回 PyTorch 张量，`"tf"`返回 TensorFlow 张量，`"np"`返回 NumPy 数组：

```python
sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

# Returns PyTorch tensors
model_inputs = tokenizer(sequences, padding=True, return_tensors="pt")

# Returns TensorFlow tensors
model_inputs = tokenizer(sequences, padding=True, return_tensors="tf")

# Returns NumPy arrays
model_inputs = tokenizer(sequences, padding=True, return_tensors="np")
```

Comment:  这一节似乎想表达分词器tokenizer非常强大，功能众多。(2024年2月5日)

Comment 2:  不过因为分词还有后续的处理，所以默认类型就可以了。(2024年2月5日)

### 2.6.1 Special tokens

如果我们看一下分词器返回的输入 ID，我们会发现它们与我们之前的输入略有不同：

```python
sequence = "I've been waiting for a HuggingFace course my whole life."

model_inputs = tokenizer(sequence)
print(model_inputs["input_ids"])

tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)
[101, 1045, 1005, 2310, 2042, 3403, 2005, 1037, 17662, 12172, 2607, 2026, 2878, 2166, 1012, 102]
[1045, 1005, 2310, 2042, 3403, 2005, 1037, 17662, 12172, 2607, 2026, 2878, 2166, 1012]
```

开头添加一个令牌 ID，末尾添加一个令牌 ID。让我们解码上面的两个 ID 序列，看看这是关于什么的：

```python
print(tokenizer.decode(model_inputs["input_ids"]))
print(tokenizer.decode(ids))
```

"[CLS] i've been waiting for a huggingface course my whole life. [SEP]"
"i've been waiting for a huggingface course my whole life."

分词器在开头添加了特殊词`[CLS]`，在末尾添加了特殊词`[SEP]`。这是因为模型是用这些预训练的，所以为了获得相同的推理结果，我们也需要添加它们。请注意，某些模型不会添加特殊单词，或添加不同的单词;模型也可以只在开头或结尾添加这些特殊单词。无论如何，分词器知道哪些是预期的，并会为您处理这个问题。

### 2.6.2 Wrapping up: From tokenizer to model

现在我们已经了解了`tokenizer`对象在应用于文本时使用的所有单个步骤，让我们最后一次看看它如何使用其主要 API 处理多个序列（填充！）、非常长的序列（截断！）和多种类型的张量：

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

tokens = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
output = model(**tokens)
```

Comment:  办法就是加上各种参数嘛。

## 2.7 Basic usage completed!

https://huggingface.co/learn/nlp-course/chapter2/7?fw=pt

跟着课程到这里干得好！回顾一下，在本章中，您将：

- 学习了 Transformer 模型的基本构建块。
- 了解了标记化管道的组成。
- 了解如何在实践中使用 Transformer 模型。
- 学习了如何利用分词器将文本转换为模型可理解的张量。
- 同时设置分词器和模型，以便从文本到预测。
- 了解了输入 ID 的局限性，并了解了注意力掩码。
- 尝试使用通用且可配置的分词器方法。

从现在开始，您应该能够自由浏览🤗变形金刚文档：词汇听起来很熟悉，并且您已经看到了大部分时间会使用的方法。

## 2.8 End-of-chapter quiz

https://huggingface.co/learn/nlp-course/chapter2/8?fw=pt

# 3. FINE-TUNING A PRETRAINED MODEL

## 3.1 Introduction

https://huggingface.co/learn/nlp-course/chapter3/1?fw=pt

在第 [2 章](https://huggingface.co/course/chapter2)中，我们探讨了如何使用分词器和预训练模型进行预测。但是，如果您想为自己的数据集微调预训练模型，该怎么办？这就是本章的主题！您将学习：

- 如何从 Hub 准备大型数据集
- 如何使用高级 API `Trainer`微调模型
- 如何使用自定义训练循环
- 如何利用 🤗 Accelerate 库在任何分布式设置上轻松运行自定义训练循环

为了将训练过的检查点上传到 Hugging Face Hub，您需要一个 huggingface.co 帐户：[创建一个帐户](https://huggingface.co/join)

Comment:  原来和训练有关的trainer在这里。(2024年2月10日)

## 3.2 Processing the data

https://huggingface.co/learn/nlp-course/chapter3/2?fw=pt

继续[上一章](https://huggingface.co/course/chapter2)中的示例，以下是我们如何在 PyTorch 中的一个批处理上训练序列分类器：

```python
import torch
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification

# Same as before
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = [
    "I've been waiting for a HuggingFace course my whole life.",
    "This course is amazing!",
]
batch = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")

# This is new
batch["labels"] = torch.tensor([1, 1])

optimizer = AdamW(model.parameters())
loss = model(**batch).loss
loss.backward()
optimizer.step()
```

当然，仅仅用两个句子来训练模型不会产生很好的结果。为了获得更好的结果，您需要准备一个更大的数据集。

在本节中，我们将使用MRPC（Microsoft Research Paraphrase Corpus）数据集作为示例，该数据集在William B. Dolan和Chris Brockett的[论文](https://www.aclweb.org/anthology/I05-5002.pdf)中介绍。该数据集由 5,801 对句子组成，并带有一个标签，指示它们是否是释义（即，如果两个句子的意思相同）。我们之所以选择它作为本章，是因为它是一个小型数据集，因此很容易对其进行训练。

### 3.2.1 Loading a dataset from the Hub

该中心不仅包含模型;它还具有许多不同语言的多个数据集。您可以[在此处](https://huggingface.co/datasets)浏览数据集，我们建议您在完成本节后尝试加载和处理新数据集（请参阅[此处](https://huggingface.co/docs/datasets/loading)的一般文档）。但现在，让我们专注于MRPC数据集！这是构成 [GLUE 基准的](https://gluebenchmark.com/) 10 个数据集之一，GLUE 基准是一个学术基准，用于衡量 ML 模型在 10 个不同文本分类任务中的性能。

🤗 数据集库提供了一个非常简单的命令，用于在 Hub 上下载和缓存数据集。我们可以像这样下载 MRPC 数据集：

```python
from datasets import load_dataset

raw_datasets = load_dataset("glue", "mrpc")
raw_datasets
DatasetDict({
    train: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 3668
    })
    validation: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 408
    })
    test: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 1725
    })
})
```

正如你所看到的，我们得到一个`DatasetDict`对象，其中包含训练集、验证集和测试集。其中每个都包含几列（`sentence1`、`sentence2`、`label`和`idx`）和可变行数，即每个集合中的元素数（因此，训练集中有 3,668 对句子，验证集中有 408 对，测试集中有 1,725 对）。

此命令下载并缓存数据集，默认存储在 *~/.cache/huggingface/datasets* 中。回想一下第 2 章，您可以通过设置`HF_HOME`环境变量来自定义缓存文件夹。

我们可以通过索引来访问`raw_datasets`对象中的每一对句子，就像使用字典一样：

```python
raw_train_dataset = raw_datasets["train"]
raw_train_dataset[0]
{'idx': 0,
 'label': 1,
 'sentence1': 'Amrozi accused his brother , whom he called " the witness " , of deliberately distorting his evidence .',
 'sentence2': 'Referring to him as only " the witness " , Amrozi accused his brother of deliberately distorting his evidence .'}
```

我们可以看到标签已经是整数，因此我们不必在那里进行任何预处理。要知道哪个整数对应于哪个标签，我们可以检查我们`raw_train_dataset`的`features` .这将告诉我们每列的类型：

```python
raw_train_dataset.features
```

{'sentence1': Value(dtype='string', id=None),
 'sentence2': Value(dtype='string', id=None),
 'label': ClassLabel(num_classes=2, names=['not_equivalent', 'equivalent'], names_file=None, id=None),
 'idx': Value(dtype='int32', id=None)}

在后台，`label` 是 `ClassLabel` 类型，整数到标签名称的映射存储在 *names* 文件夹中。`0` 对应于 `not_equivalent`，`1`对应于 `equivalent`。

Comment:  看起来raw_datasets是个二维字典。(2024年2月14日)

### 3.2.2 Preprocessing a dataset

为了预处理数据集，我们需要将文本转换为模型可以理解的数字。正如您在[上一章](https://huggingface.co/course/chapter2)中看到的，这是通过分词器完成的。我们可以向分词器提供一句话或一个句子列表，这样我们就可以直接标记每对的所有第一句话和所有第二句话，如下所示：

```python
from transformers import AutoTokenizer

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenized_sentences_1 = tokenizer(raw_datasets["train"]["sentence1"])
tokenized_sentences_2 = tokenizer(raw_datasets["train"]["sentence2"])
```

但是，我们不能只将两个序列传递给模型并预测这两个句子是否是释义。我们需要将两个序列作为一对处理，并应用适当的预处理。幸运的是，分词器还可以获取一对序列，并按照我们的 BERT 模型期望的方式进行准备：

```python
tokenized_ids = tokenizer("This is the first sentence.", "This is the second one.")
```

inputs
{ 
  'input_ids': [101, 2023, 2003, 1996, 2034, 6251, 1012, 102, 2023, 2003, 1996, 2117, 2028, 1012, 102],
  'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
  'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
}

Comment:  这里的input_ids我运行出来和文档中是一样的。可见分词结果是个定数。(2024年2月14日)

我们在第 [2 章](https://huggingface.co/course/chapter2)中讨论了 `input_ids` 和 `attention_mask` 键，但我们推迟了讨论 `token_type_ids`。在此示例中，`token_type_ids`就是告诉模型输入的哪一部分是第一句话，哪一部分是第二句话。

✏️ 试试看！采用训练集的元素 15，将两个句子分别标记化，并成对标记。这两个结果之间有什么区别？

如果我们将`input_ids`里面的 ID 解码回单词：

```python
tokenizer.convert_ids_to_tokens(inputs["input_ids"])
```

我们将得到：

['[CLS]', 'this', 'is', 'the', 'first', 'sentence', '.', '[SEP]', 'this', 'is', 'the', 'second', 'one', '.', '[SEP]']

因此，我们看到模型期望输入是当有两个句子`[CLS] sentence1 [SEP] sentence2 [SEP]`时的形式。将其与`token_type_ids`保持一致，可以让我们：

['[CLS]', 'this', 'is', 'the', 'first', 'sentence', '.', '[SEP]', 'this', 'is', 'the', 'second', 'one', '.', '[SEP]']
[      0,      0,      0,     0,       0,          0,            0,     0,        1,     1,     1,        1,             1,   1,       1]

如您所见，输入中对应于 `[CLS] sentence1 [SEP]` 的部分的令牌类型 ID 为`0` ，而对应于 `sentence2 [SEP]` 的其他部分的令牌类型 ID 均为 `1`。

请注意，如果选择其他检查点，则不一定在标记化输入中包含 `token_type_ids`（例如，如果使用 DistilBERT 模型，则不会返回它们）。只有当模型知道如何处理它们时，它们才会返回，因为它在预训练期间已经看到了它们。

在这里，BERT 使用令牌类型 ID (Comment: 就是那些大小各异的整型数) 进行预训练，除了我们在第 [1 章](https://huggingface.co/course/chapter1)中讨论的掩码语言建模目标之外，它还有一个额外的目标，称为 **下一句预测**。此任务的目标是对句子对之间的关系进行建模。

通过下一个句子预测，模型被提供成对的句子（带有随机掩码标记），并要求预测第二个句子是否在第一个句子之后。为了使任务不平凡，**一半的时间句子在原始文档中彼此跟随，另一半时间这两个句子来自两个不同的文档**。(Comment:  paper evolve relation是否可以用这种办法，一半数量具有link关系，另一半数量不具有link关系)

一般来说，你不需要担心你的标记化输入中是否有 `token_type_ids`：只要你对标记器和模型使用相同的检查点，一切都会好起来的，因为标记器知道要向它的模型提供什么。

现在我们已经了解了我们的分词器如何处理一对句子，我们可以使用它来标记我们的整个数据集：就像[在上一章](https://huggingface.co/course/chapter2)中一样，我们可以通过给分词器提供第一个句子列表，然后是第二个句子列表来为分词器提供句子对列表。这也与我们[在第 2 章](https://huggingface.co/course/chapter2)中看到的填充和截断选项兼容。因此，预处理训练数据集的一种方法是：

```python
tokenized_dataset = tokenizer(
    raw_datasets["train"]["sentence1"],
    raw_datasets["train"]["sentence2"],
    padding=True,
    truncation=True,
)
```

这很好用，但它的缺点是返回字典（使用我们的键、`input_ids`、`attention_mask` 和 `token_type_ids`、以及作为列表列表的值）。它也只有在标记化期间有足够的 RAM 来存储整个数据集时才有效（而数据集库中的🤗数据集是存储在磁盘上的 [Apache Arrow](https://arrow.apache.org/) 文件，因此您只将您要求的样本加载到内存中）。（Comment:  比如说存储文件是glue-train.arrow）

为了将数据保留为数据集，我们将使用 [`Dataset.map（）`](https://huggingface.co/docs/datasets/package_reference/main_classes#datasets.Dataset.map) 方法。这也为我们提供了一些额外的灵活性，如果我们需要做更多的预处理，而不仅仅是标记化。`map()`方法的工作原理是在数据集的每个元素上应用一个函数，因此让我们定义一个函数来标记我们的输入：

```python
def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)
```

此函数采用字典（如数据集中的项目）并返回一个带有键 `input_ids`、`attention_mask` 和 `token_type_ids` 的新字典。请注意，如果`example` 字典包含多个样本（每个键作为句子列表），`tokenizer` 也有效，因为如前所述，它适用于句子对列表。这将允许我们在调用 `map()` 中使用 `batched=True` 选项，这将大大加快标记化的速度。`tokenizer`由 Tokenizers 库中用 Rust [🤗](https://github.com/huggingface/tokenizers) 编写的分词器提供支持。这个分词器可以非常快，但前提是我们一次给它很多输入。

请注意，我们暂时在标记化函数中省略了 `padding` 参数。这是因为**将所有样本填充到最大长度是效率不高的：最好在构建批处理时填充样本**，因为这样我们只需要填充到该批次中的最大长度，而不是整个数据集中的最大长度。当输入的长度非常可变时，这可以节省大量时间和处理能力！

以下是我们如何同时在所有数据集上应用标记化函数。我们在调用 `map` 中使用 `batched=True`，因此该函数一次应用于数据集的多个元素，而不是单独应用于每个元素。这样可以加快预处理速度。

```python
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
tokenized_datasets
```

Datasets 库应用此处理的方式🤗**是向数据集添加新字段**，预处理函数返回的字典中的每个键对应一个字段（Comment:  就是把tokenized函数得到的三个值加进去了）：

DatasetDict({
    train: Dataset({
        features: ['attention_mask', '**idx**', 'input_ids', '**label**', '**sentence1**', '**sentence2**', 'token_type_ids'],

​        num_rows: 3668
​    })
​    validation: Dataset({
​        features: ['attention_mask', 'idx', 'input_ids', 'label', 'sentence1', 'sentence2', 'token_type_ids'],
​        num_rows: 408
​    })
​    test: Dataset({
​        features: ['attention_mask', 'idx', 'input_ids', 'label', 'sentence1', 'sentence2', 'token_type_ids'],
​        num_rows: 1725
​    })
})

在应用预处理函数时，您甚至可以通过 `map()` 传递参数 `num_proc`  来使用多重处理。我们在这里没有这样做，🤗因为 Tokenizers 库已经使用多个线程来更快地标记我们的样本，但如果你没有使用这个库支持的快速标记器，这可能会加快你的预处理速度。

我们 `tokenize_function` 返回一个带有键 `input_ids`、`attention_mask` 和 `token_type_ids` 的字典，因此这三个字段被添加到数据集的所有拆分中。请注意，如果我们的预处理函数为我们应用 `map()` 的数据集中的现有键返回新值，我们也可以更改现有字段。

我们需要做的最后一件事是，当我们将元素批处理在一起时，将所有示例填充到最长元素的长度上——我们称之为*动态填充*的技术。

Comment:  我看明白了，这是用glue-mrpc作为数据集，用bert-base-uncased作为分词器。(2024年2月14日)

### Q: huggingface的AdamW是干啥的？

在 Hugging Face Transformers 库中，`AdamW` 是一个优化器类，用于实现带有权重衰减（Weight Decay）的 Adam 优化算法。AdamW 是对 Adam 优化算法的一个变种，它在原始的 Adam 算法基础上添加了权重衰减的功能，以解决优化器在一些任务中可能会导致模型过拟合的问题。

Adam 优化算法是一种常用的梯度下降优化算法，它结合了 AdaGrad 和 RMSprop 的优点，并引入了动量（Momentum）的概念，以在训练过程中自适应地调整学习率。Adam 优化算法在很多任务上表现良好，但它存在一个问题，即在某些情况下可能会导致模型的权重过大，从而导致过拟合。

为了解决这个问题，AdamW 引入了权重衰减（Weight Decay）的概念，即在计算梯度时对权重参数进行额外的衰减（类似于 L2 正则化）。这样做可以有效地控制权重的大小，防止模型过拟合。

在使用 Hugging Face Transformers 库时，我们可以通过创建一个 `AdamW` 优化器，并将模型的参数传递给它，来对模型的权重进行优化。在创建 `AdamW` 优化器时，我们可以指定学习率、权重衰减率以及其他优化器的超参数。

示例代码：

```python
from transformers import AdamW, AutoModel

model = AutoModel.from_pretrained("bert-base-uncased")

# 创建 AdamW 优化器，并设置学习率和权重衰减率
optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)

# 在训练过程中使用 optimizer 进行参数优化
```

在上述代码中，我们首先创建了一个 BERT 模型 `model`，然后创建了一个 `AdamW` 优化器，并将模型的参数传递给它。我们设置了学习率为 1e-5 和权重衰减率为 0.01，并在训练过程中使用这个优化器来优化模型的参数。通过使用 `AdamW` 优化器，我们可以在训练过程中控制权重的大小，从而提高模型的泛化性能。

### Q: AdamW有哪些参数?

`AdamW` 是一种优化器，是 Adam 优化器的一种变体，它在参数更新时引入了权重衰减（Weight Decay）以控制正则化。以下是 `torch.optim.AdamW` 类的一些常见参数：

1. **params** (*iterable*)：一个包含要优化的参数的可迭代对象，通常是模型的参数列表。

2. **lr** (*float*)：学习率（Learning Rate），控制每次参数更新的步长。

3. **betas** (*Tuple[float, float], optional*)：用于计算梯度和梯度平方的移动平均值的系数。通常为 `(beta1, beta2)`，其中 `beta1` 控制梯度的移动平均，`beta2` 控制梯度平方的移动平均。

4. **eps** (*float, optional*)：用于数值稳定性的小值，防止分母为零。

5. **amsgrad** (*bool, optional*)：是否使用 AMSGrad 变体，它修正了 Adam 优化器的移动平均方差的计算方法，旨在提高稳定性。

6. **weight_decay** (*float, optional*)：权重衰减，控制正则化项的强度。在 AdamW 中，权重衰减被应用于所有参数，包括偏置项。

7. **correct_bias** (*bool, optional*)：是否对偏置修正进行偏差校正。如果设置为 True，偏置修正将进行偏差校正，否则不进行。

8. **params** (*iterable, optional*)：要优化的参数的可迭代对象。如果未提供，将默认为模型的所有参数。

9. **lr** (*float, optional*)：学习率。如果未提供，将默认为 1e-3。

10. **weight_decay** (*float, optional*)：权重衰减。如果未提供，将默认为 0。

11. **eps** (*float, optional*)：数值稳定性的小值。如果未提供，将默认为 1e-8。

12. **amsgrad** (*bool, optional*)：是否使用 AMSGrad 变体。如果未提供，将默认为 False。

13. **amsgrad_correct_bias** (*bool, optional*)：是否对 AMSGrad 变体的偏置修正进行偏差校正。如果未提供，将默认为 True。

这些参数将帮助你调整 AdamW 优化器的行为以适应你的训练需求。具体参数的选择需要根据实际问题和实验来确定。

### Q: 结果中的token_type_ids是什么？

在 Hugging Face Transformers 库中，`token_type_ids` 是用于处理输入序列中的分段信息的一种编码方式。对于一些预训练模型（例如 BERT、RoBERTa 等），它们在输入时需要同时考虑两个句子或分段的信息，因此需要一种方式来区分不同分段的内容。

在处理文本序列时，通常将输入序列切分成多个片段，每个片段对应一个输入序列。例如，当处理句子对任务时，一个输入序列可能包含两个句子，其中一个句子放在前面，另一个句子放在后面，中间可能有一个特殊的分隔符。`token_type_ids` 就是用来区分这些不同片段的标识。

`token_type_ids` 是一个与输入序列等长的向量，其长度与输入序列中的 token 数目相同。在处理句子对任务时，如果一个 token 属于第一个句子，那么它的 `token_type_ids` 就会被标记为 0；如果它属于第二个句子，那么它的 `token_type_ids` 就会被标记为 1。(如果属于第三个句子，就被标记为2) 这样，模型就可以根据 `token_type_ids` 来区分不同句子的信息。

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

### Q: tokenizer.convert_ids_to_tokens是干啥的？

`tokenizer.convert_ids_to_tokens` 是 Hugging Face Transformers 库中用于将词汇表中的标记 ID（token IDs）转换为对应的文本标记（token）的方法。

在自然语言处理任务中，文本数据通常被分成一个个的标记（tokens），例如单词、子词或字符。这些标记被映射到一个预训练的词汇表，每个标记都有一个唯一的标记 ID。在处理模型输出或进行后处理时，有时需要将这些标记 ID 转换回文本形式。

`tokenizer.convert_ids_to_tokens` 接受一个包含标记 ID 的列表（或张量），并返回一个包含对应文本标记的列表。这在生成文本、分析模型输出等场景中非常有用。

示例代码：

```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
token_ids = [101, 2023, 2003, 1037, 2232, 1999, 1996, 2190, 2226, 102]

tokens = tokenizer.convert_ids_to_tokens(token_ids)
print(tokens)
```

在上述示例中，`convert_ids_to_tokens` 方法将标记 ID 列表 `[101, 2023, 2003, ...]` 转换为对应的文本标记列表 `['[CLS]', 'what', 'is', 'a', 'good', 'answer', 'to', 'the', 'question', '[SEP]']`。这使得您可以更好地理解模型输出，并进行后续的分析和可视化。

### 3.2.3 Dynamic padding

负责将批处理中的样本放在一起的函数称为 *collate 函数*。这是一个参数，您可以在构建 `DataLoader` 时传递 ，默认值是一个函数，该函数只会将您的样本转换为 PyTorch 张量并将它们连接起来（如果您的元素是列表、元组或字典，则递归）。在我们的例子中，这是不可能的，因为我们拥有的输入不会都具有相同的大小。我们特意推迟了填充，只在每个批次上根据需要应用它，并避免有过长的输入和大量的填充。这将大大加快训练速度，但请注意，如果您在 TPU 上进行训练，可能会导致问题——TPU 更喜欢固定形状，即使这需要额外的填充。

为了在实践中做到这一点，我们必须定义一个 collate 函数，该函数将正确数量的填充应用于我们想要批处理在一起的数据集项目。幸运的是，🤗 Transformers 库通过 `DataCollatorWithPadding` 为我们提供了这样的功能。当你实例化它时，它需要一个分词器（以知道要使用哪个填充标记，以及模型期望填充是在输入的左侧还是右侧），并将执行您需要的所有操作：

```python
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```

为了测试这个新玩具，让我们从训练集中获取一些样本，我们想将它们分批在一起。在这里，我们删除了列`idx` 、`sentence1` 和 `sentence2`，因为它们不需要并且包含字符串（并且我们不能使用字符串创建张量），并查看批处理中每个条目的长度：

```python
samples = tokenized_datasets["train"][:8]
samples = {k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]}
print([len(x) for x in samples["input_ids"]])
```

[50, 59, 47, 67, 59, 50, 62, 32]

（Comment: 去掉列`idx` 、`sentence1` 和 `sentence2`很好理解，因为模型只需要tokenized_ids就可以进行后续计算了）

毫不奇怪，我们得到了不同长度的样本，从 32 到 67。动态填充意味着此批次中的样品都应填充到 67 的长度，即批次内的最大长度。如果没有动态填充，则必须将所有样本填充到整个数据集中的最大长度或模型可以接受的最大长度。让我们仔细检查我们的 `data_collator` 是否正确地动态填充了批处理：

```python
batch = data_collator(samples)
print({k: v.shape for k, v in batch.items()})
```

{'attention_mask': torch.Size([8, 67]),
 'input_ids': torch.Size([8, 67]),
 'token_type_ids': torch.Size([8, 67]),
 'labels': torch.Size([8])}

看起来不错！现在我们已经从原始文本变成了模型可以处理的批处理，我们准备对其进行微调！

✏️ 试试看！在 GLUE SST-2 数据集上复制预处理。它有点不同，因为它是由单个句子而不是成对组成的，但我们所做的其余部分应该看起来相同。对于更难的挑战，请尝试编写一个适用于任何 GLUE 任务的预处理函数。

### Q: 你知道transformer的Dynamic padding吗？

在 Transformer 模型中，Dynamic Padding 是一种技术，用于处理输入序列的不等长问题。在自然语言处理等任务中，输入文本的长度往往会不同，而 Transformer 模型要求输入序列的长度固定，因此需要将不等长的输入序列转换成等长的形式。

Dynamic Padding 是一种动态填充技术，它根据输入序列的实际长度来动态地在序列的末尾添加特殊的填充标记（通常是 0），从而将所有输入序列填充成相同长度。这样，所有输入序列都具有相同的长度，可以一起批量处理，以提高模型训练的效率。

具体来说，对于一个 batch 中的多个序列，首先**找到 batch 中最长序列的长度，然后将其他较短的序列进行填充，使它们与最长序列长度相同**。填充通常在序列的末尾添加特殊的填充标记（例如 0），以示区别原始数据和填充数据。在训练过程中，模型会忽略填充标记的影响，以确保填充部分不会影响模型的预测结果。

Dynamic Padding 技术在 Transformer 中是一种非常常用的方法，它使得 Transformer 可以处理不等长的输入序列，使得模型更具通用性和适用性，同时减少了内存和计算资源的浪费。

### Q: Hugging Face Datasets 库中的 map 函数是干啥的？

在 Hugging Face Datasets 库中，`map()` 函数是用于**对数据集中的每个样本应用指定的函数，并返回一个新的数据集**。这个函数类似于 Python 内置的 `map()` 函数，但是它专门用于处理数据集，方便对数据进行批量处理和转换。

`map()` 函数在 Hugging Face Datasets 库中的基本用法如下：

```python
map(function, # 表示要对数据集中每个样本应用的函数，可以是一个函数或者一个处理数据的类。
    batched=True, # 表示是否将数据集以批量的方式处理，默认为 `True`。
    batch_size=None, # 表示批量处理的大小，默认为 `None`。
    num_proc=1, # 表示并行处理的进程数，默认为 1。
    remove_columns=None, # 通常用于移除不需要处理的列，以减少内存占用。
    load_from_cache_file=True, # 是否从缓存文件中加载数据，默认为 `True`。
    cache_file_name=None) # 缓存文件的名称，默认为 `None`。
```

- `function`：表示要对数据集中每个样本应用的函数，可以是一个函数或者一个处理数据的类。

- `batched`：表示是否将数据集以批量的方式处理，默认为 `True`。如果为 `True`，`function` 将会以批量的方式处理数据，提高处理效率。

- `batch_size`：表示批量处理的大小，默认为 `None`。当 `batched=True` 且 `batch_size` 不为 `None` 时，数据集会以指定的批量大小进行处理。

- `num_proc`：表示并行处理的进程数，默认为 1。如果 `num_proc` 大于 1，则 `map()` 函数会使用多进程并行处理数据。

- `remove_columns`：表示要从数据集中移除的列，通常用于移除不需要处理的列，以减少内存占用。

- `load_from_cache_file`：表示是否从缓存文件中加载数据，默认为 `True`。如果为 `True`，并且数据集有缓存文件，则会从缓存文件中加载数据，加快数据加载速度。

- `cache_file_name`：表示缓存文件的名称，默认为 `None`。如果指定了缓存文件名称，则会将数据缓存到指定的文件中，以便下次使用。

`map()` 函数在 Hugging Face Datasets 库中通常用于对数据进行预处理、转换或者特征提取。通过传入不同的 `function` 参数，可以对数据集中的每个样本进行不同的处理。这个函数在深度学习中数据预处理和数据加载过程中非常实用，可以帮助提高数据处理效率和灵活性。

### Q: DataCollatorWithPadding是干啥的？

在 Hugging Face Transformers 库中，`DataCollatorWithPadding` 是一个用于处理数据并进行填充（padding）的类。在进行文本序列任务的训练时，由于输入序列长度可能不一致，我们需要将输入序列进行填充，以便将它们组织成一个批次（batch），并输入到模型中进行训练。

`DataCollatorWithPadding` 类的主要作用是将数据样本（例如文本对、单句文本等）组织成一个批次，并根据最大长度进行填充，以使整个批次中的每个样本的输入序列长度一致。

具体来说，`DataCollatorWithPadding` 类接受一个数据样本列表（例如一个由多个文本对组成的列表），并根据最大长度对这些样本进行填充，然后返回一个包含填充后输入序列的字典。在这个字典中，输入序列被组织成了一个批次，并且所有输入序列的长度都是相同的。

这样做的好处是，我们可以同时处理不同长度的样本，并将它们组织成一个批次输入到模型中，从而提高训练效率。

示例代码：

```python
from transformers import AutoTokenizer, DataCollatorWithPadding

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 数据样本列表，每个样本是一个文本对
data_samples = [
    {"text": "Hello, how are you?", "label": 0},
    {"text": "I am fine, thank you!", "label": 1},
    {"text": "Goodbye!", "label": 2},
]

# 将数据样本组织成一个批次，并进行填充
batch = data_collator(data_samples)

# 输出填充后的批次数据
print(batch)
```

在上述代码中，我们首先创建了一个 BERT tokenizer，并使用 `DataCollatorWithPadding` 类创建了一个 `data_collator` 对象。然后，我们定义了一个包含文本对和标签的数据样本列表 `data_samples`，并使用 `data_collator` 将数据样本组织成一个批次，并进行填充。最后，输出填充后的批次数据。

通过使用 `DataCollatorWithPadding` 类，我们可以轻松地将不同长度的文本序列组织成一个批次，并在模型训练过程中进行填充和处理，从而提高训练效率和性能。

### Q: DataCollatorWithPadding有哪些参数？

`DataCollatorWithPadding` 是 Hugging Face Transformers 库中用于数据批处理和填充的类，它的参数与用途如下：

1. `tokenizer` (required): 要使用的 tokenizer，用于将文本转换为模型可接受的输入格式。

2. `padding`: 是否进行填充，如果为 `True`，则对序列进行填充以使它们的长度一致。

3. `max_length`: 设置最大序列长度，超过这个长度会进行截断或填充。

4. `pad_to_multiple_of`: 如果设置了这个参数，填充到的序列长度将是这个值的倍数。

5. `return_tensors`: 返回的数据类型，如 `"pt"` 表示返回 PyTorch 张量。

6. `return_attention_mask`: 是否返回注意力掩码（attention mask），用于标识 padding 部分。

7. `return_token_type_ids`: 是否返回 token type IDs，用于区分不同的句子。

8. `return_overflowing_tokens`: 是否返回超过最大长度的 token。

9. `return_special_tokens_mask`: 是否返回特殊 tokens 的掩码。

10. `return_length`: 是否返回每个样本的实际长度。

11. `return_offsets_mapping`: 是否返回每个 token 在原始文本中的偏移。

等等...

这些参数可以根据数据和任务的需求进行调整。`DataCollatorWithPadding` 通常与 Hugging Face Transformers 库中的 `Trainer` 类一起使用，用于在训练过程中对数据进行批处理和填充。

## 3.3 Fine-tuning a model with the Trainer API

🤗 Transformers 提供了一个`Trainer`类，可帮助您微调它在数据集上提供的任何预训练模型。完成上一节中的所有数据预处理工作后，只需几个步骤即可定义 `Trainer` . 最难的部分可能是准备 `Trainer.train()` 运行环境，因为它在 CPU 上运行速度非常慢。如果您没有设置 GPU，可以在 [Google Colab](https://colab.research.google.com/) 上访问免费的 GPU 或 TPU。

下面的代码示例假定您已经执行了上一节中的示例。以下是概述您需要的简短摘要：

```python
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```

### 3.3.1 Training

在定义 `Trainer` 之前，第一步是定义一个 `TrainingArguments` 类，该类将包含 `Trainer` 将用于训练和评估的所有超参数。您必须提供的唯一参数是将保存经过训练的模型的目录，以及沿途的检查点。对于其余的，您可以保留默认值，这对于基本的微调应该很有效。

```python
from transformers import TrainingArguments

training_args = TrainingArguments("test-trainer")
```

（Comment:  test-trainer是checkpoint存储路径）

💡 If you want to automatically upload your model to the Hub during training, pass along `push_to_hub=True` in the `TrainingArguments`. 我们将在第 [4 章](https://huggingface.co/course/chapter4/3)中对此进行更多了解。

第二步是定义我们的模型。与[上一章](https://huggingface.co/course/chapter2)一样，我们将使用带有两个标签的类 `AutoModelForSequenceClassification`：

```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
```

您会注意到，与[第 2 章](https://huggingface.co/course/chapter2)不同，在实例化此预训练模型后，您会收到警告。这是**因为 BERT 尚未对句子对进行分类预训练，因此丢弃了预训练模型的头部，而是添加了适合序列分类的新头部**。警告表明某些权重未被使用（对应于丢弃的预训练头的权重），而其他一些权重是随机初始化的（新头的权重）。最后，它鼓励你训练模型，这正是我们现在要做的。

一旦我们有了模型，我们就可以通过向它传递到目前为止构建的所有对象来定义一个 `Trainer` ---- `model`、 `training_args`、 训练和验证数据集、我们的 `data_collator` 和 我们的 `tokenizer`：

```python
from transformers import Trainer

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)
```

Note that we create a new `TrainingArguments` with its `evaluation_strategy` set to `"epoch"` and a new model — otherwise, we would just be continuing the training of the model we have already trained. To launch a new training run, we execute:

```python
trainer.train()
```

这将开始微调（在 GPU 上应该需要几分钟），并每 500 步报告一次训练损失。但是，它不会告诉您模型的性能有多好（或有多差）。这是因为：

1. 我们没有通过设置 `evaluation_strategy` 为 `"steps"`（评估每个`eval_steps`）或 `"epoch"`（在每个时期结束时评估）来告诉 `Trainer` 训练期间进行评估。
2. 在上述评估期间，我们没有提供`Trainer`计算指标`compute_metrics()`的函数（否则评估只会打印损失，这不是一个非常直观的数字）。

### Q: 解释from transformers import TrainingArguments

`from transformers import TrainingArguments` 是从 Hugging Face Transformers 库中导入 `TrainingArguments` 类。这个类用于配置和管理模型训练的各种参数和选项。

在深度学习中，模型训练涉及许多参数，如学习率、批量大小、训练轮数等。`TrainingArguments` 类允许您通过对象属性设置这些参数，以便更轻松地管理训练过程。

具体来说，`TrainingArguments` 类允许您设置以下类型的参数：

- `output_dir`: 训练过程中保存模型和输出文件的目录。
- `per_device_train_batch_size`: 每个设备的训练批量大小。
- `num_train_epochs`: 训练的轮数。
- `learning_rate`: 学习率。
- `logging_dir`: 日志文件保存的目录。
- `save_total_limit`: 保存的检查点总数限制。
- 等等...

通过使用 `TrainingArguments`，您可以将这些训练参数集中在一起，并在模型训练时传递给 `Trainer` 类，从而更好地组织和管理训练配置。

示例用法：

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./results", # 训练过程中保存模型和输出文件的目录。
    per_device_train_batch_size=8, # 每个设备的训练批量大小。
    num_train_epochs=3, # 训练的轮数。
    learning_rate=5e-5, # 学习率。
    logging_dir="./logs", # 日志文件保存的目录。
    save_total_limit=3, # 保存的检查点总数限制。
)

# 在模型训练时将 training_args 传递给 Trainer 类
```

总之，`from transformers import TrainingArguments` 语句使您能够导入并使用 `TrainingArguments` 类来设置和管理模型训练的各种参数。

### Q: 解释： Some weights of the model checkpoint at bert-base-uncased were not used  when initializing BertForSequenceClassification:  ['cls.predictions.bias', 'cls.predictions.transform.dense.weight',  'cls.predictions.transform.dense.bias', 'cls.seq_relationship.weight',  'cls.predictions.transform.LayerNorm.bias',  'cls.predictions.transform.LayerNorm.weight',  'cls.seq_relationship.bias'] - This IS expected if you are initializing BertForSequenceClassification from the checkpoint of a model trained on another task or with another  architecture (e.g. initializing a BertForSequenceClassification model  from a BertForPreTraining model). - This IS NOT expected if you are initializing  BertForSequenceClassification from the checkpoint of a model that you  expect to be exactly identical (initializing a  BertForSequenceClassification model from a BertForSequenceClassification model). Some weights of BertForSequenceClassification were not initialized from  the model checkpoint at bert-base-uncased and are newly initialized:  ['classifier.weight', 'classifier.bias'] You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.

这段信息是在初始化 `BertForSequenceClassification` 模型时产生的警告和说明。这些警告和说明向您解释了模型初始化过程中遇到的情况。

解释如下：

- "Some weights of the model checkpoint at bert-base-uncased were not used when initializing BertForSequenceClassification": 这部分说明了在使用 `bert-base-uncased` 预训练模型的权重初始化 `BertForSequenceClassification` 模型时，一些权重并未被使用。这可能是因为预训练模型和当前要初始化的模型在架构或任务上存在不同。

- "This IS expected if you are initializing BertForSequenceClassification from the checkpoint of a model trained on another task or with another architecture": 如果您从一个在另一个任务上或使用另一种架构训练的模型的检查点初始化 `BertForSequenceClassification` 模型，那么上述情况是正常的。

- "This IS NOT expected if you are initializing BertForSequenceClassification from the checkpoint of a model that you expect to be exactly identical": 如果您从一个预期在架构和任务上与要初始化的模型完全相同的模型的检查点初始化 `BertForSequenceClassification` 模型，那么上述情况不应该出现。

- "Some weights of BertForSequenceClassification were not initialized from the model checkpoint at bert-base-uncased and are newly initialized": 这部分说明了 `BertForSequenceClassification` 模型的一些权重没有从 `bert-base-uncased` 的检查点中初始化，而是被新初始化。

- "You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference": 这部分建议您在下游任务上对这个模型进行训练，以便在预测和推断中使用它。

综合起来，这段信息告诉您模型初始化过程中遇到的权重使用情况，并提醒您可能需要对模型进行下游任务的训练，以便在预测和推断中使用它。

### Q: Trainer有哪些参数？

`Trainer` 类在 Hugging Face Transformers 库中有许多可配置的参数，用于定制模型训练的行为。以下是一些常用的 `Trainer` 类参数：

- `model`：要训练的模型。

- `args`：一个 `TrainingArguments` 类的实例，用于配置训练参数，如批量大小、学习率、训练轮数等。

- `data_collator`：数据收集器，用于将批次数据转换为模型输入。

- `train_dataset`：训练数据集。

- `eval_dataset`：验证数据集。

- `tokenizer`：用于将原始文本转换为模型输入的分词器。

- `compute_metrics`：用于计算评估指标的函数。

- `callbacks`：一个列表，包含在训练过程中调用的回调函数。

- `optimizers`：优化器（如AdamW）。

- `scheduler`：学习率调度器。

- `model_init`：一个可选的函数，用于自定义模型初始化。

- `device`：训练设备，可以是 "cpu" 或 "cuda"。

- `world_size`：多个 GPU 的训练时，设置的世界大小。

- `train_dataloader`：自定义的训练数据加载函数。

- `eval_dataloaders`：自定义的验证数据加载函数。

- `logger`：用于记录训练日志的对象。

- `resume_from_checkpoint`：从检查点文件恢复训练。

- 等等...

这些参数提供了对训练过程中的各种方面进行细粒度控制的能力，从数据加载到优化器设置，再到回调函数和评估指标计算。根据您的需求，您可以选择性地设置这些参数，以配置适合您任务和模型的训练流程。完整的参数列表和详细说明可以在 Hugging Face Transformers 文档中找到。

### 3.3.2 Evaluation

让我们看看如何构建一个有用的 `compute_metrics()` 函数，并在下次训练时使用它。该函数必须接受一个 `EvalPrediction` 对象（该对象是带有 `predictions` 字段和 `label_ids` 字段的命名元组），并将返回将字符串映射到浮点数的字典（字符串是返回的指标的名称，浮点数是它们的值）。为了从我们的模型中获取一些预测，我们可以使用 `Trainer.predict()` 命令：

```python
predictions = trainer.predict(tokenized_datasets["validation"])
print(predictions.predictions.shape, predictions.label_ids.shape)
```

(408, 2) (408,)

`predict()`方法的输出是另一个命名的元组，其中包含三个字段：`predictions`、 `label_ids`和`metrics` 。`metrics`字段将仅包含传递的数据集的损失，以及一些时间指标（预测所需的时间、总计和平均值）。一旦我们完成`compute_metrics()`函数并将其传递给 `Trainer`，该字段还将包含 `compute_metrics()` 返回的指标。

如您所见，`predictions`是一个形状为 408 x 2 的二维数组（408 是我们使用的数据集中的元素数）。这些是我们传递到`predict()`的数据集中每个元素的 logit（正如您在[上一章](https://huggingface.co/course/chapter2)中看到的，所有 Transformer 模型都返回 logits）。为了将它们转换为可以与标签进行比较的预测，我们需要在第二轴上取具有最大值的索引：

```python
import numpy as np

preds = np.argmax(predictions.predictions, axis=-1)
```

我们现在可以将`preds`与标签进行比较。为了构建我们的`compute_metric()`函数，我们将依赖于 [Evaluate](https://github.com/huggingface/evaluate/) 库中的🤗指标。我们可以像加载数据集一样轻松地加载与 MRPC 数据集相关的指标，这次使用函数`evaluate.load()`。返回的对象有一个可用于执行度量计算的`compute()`方法：

```python
import evaluate

metric = evaluate.load("glue", "mrpc")
metric.compute(predictions=preds, references=predictions.label_ids)
```

{'accuracy': 0.8578431372549019, 'f1': 0.8996539792387542}

我的未训练的结果1：{'accuracy': 0.6838235294117647, 'f1': 0.8122270742358079}

我的未训练的结果2：{'accuracy': 0.5563725490196079, 'f1': 0.6629422718808194}

我的训练后的结果1：{'accuracy': 0.8627450980392157, 'f1': 0.903448275862069}

我的训练后的结果2：{'accuracy': 0.8578431372549019, 'f1': 0.901023890784983}

您获得的确切结果可能会有所不同，因为模型头的随机初始化可能会改变它实现的指标。在这里，我们可以看到我们的模型**在验证集上**的准确率为 85.78%，F1 得分为 89.97。这是用于评估 GLUE 基准的 MRPC 数据集结果的两个指标。[BERT论文](https://arxiv.org/pdf/1810.04805.pdf)中的表格报告了基本模型的F1得分为88.9。That was the `uncased` model while we are currently using the `cased` model, which explains the better result. 

将所有内容包装在一起，我们得到我们的`compute_metrics()`函数：

```python
def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    
    return metric.compute(predictions=predictions, references=labels)
```

为了查看它在每个epoch结束时报告指标的实际应用，以下是我们如何使用`compute_metrics()`函数定义一个新`Trainer`：

```python
training_args = TrainingArguments("test-trainer", evaluation_strategy="epoch")
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
```

Note that we create a new `TrainingArguments` with its `evaluation_strategy` set to `"epoch"` and a new model，否则，我们只会继续训练我们已经训练的模型。为了启动新的训练运行，我们执行：

```python
trainer.train()
```

这一次，它将在训练损失之上报告每个epoch结束时的验证损失和指标。同样，您达到的确切准确性/F1 分数可能与我们发现的略有不同，因为模型的头部初始化是随机的，但它应该在同一范围内。

`Trainer`将在多个 GPU 或 TPU 上开箱即用，并提供许多选项，例如混合精度训练（在训练参数中使用`fp16 = True`）。我们将在第 10 章中介绍它支持的所有内容。

使用 `Trainer` API 进行微调的介绍到此结束。[第 7 章](https://huggingface.co/course/chapter7)将给出一个针对大多数常见 NLP 任务执行此操作的示例，但现在让我们看看如何在纯 PyTorch 中执行相同的操作。

✏️ 试试看！使用在第 2 节中执行的数据处理，在 GLUE SST-2 数据集上微调模型。

## 3.4 A full training

### 3.4.1 Prepare for training

### Q: DataLoader有哪些参数？

`DataLoader` 是 PyTorch 库中用于加载数据的类，通常用于训练神经网络模型。以下是部分常见的 `DataLoader` 参数：

1. `dataset`: 要加载的数据集。通常是一个 `torch.utils.data.Dataset` 类的实例。

2. `batch_size`: 每个批次中的样本数量。

3. `shuffle`: 是否在每个 epoch 开始时对数据进行洗牌（随机排序）。

4. `num_workers`: 用于加载数据的子进程数量，可以加快数据加载速度。

5. `collate_fn`: 用于将多个样本组合成一个批次的函数。通常用于处理不同长度的序列数据。

6. `pin_memory`: 是否将数据加载到 GPU 的固定内存中，可以加速数据传输。

7. `drop_last`: 是否丢弃最后一个批次中样本数不足的部分。

8. `timeout`: 数据加载的超时时间。

9. `sampler`: 用于确定样本的顺序，可以是 `torch.utils.data.Sampler` 类的实例。

10. `batch_sampler`: 用于确定批次的顺序，可以是 `torch.utils.data.BatchSampler` 类的实例。

11. `persistent_workers`: 是否在子进程之间保持数据加载器的状态，可以加快多个 epoch 的数据加载速度。

等等...

这些参数可以根据任务的需求进行调整。`DataLoader` 在训练神经网络时非常常见，它可以帮助将数据加载、处理和批处理等步骤整合在一起，使训练过程更加方便和高效。

### Q: from transformers import get_scheduler是干啥的？

`from transformers import get_scheduler` 是 Hugging Face Transformers 库中的一个函数，用于获取针对优化器的学习率调度器（learning rate scheduler）。学习率调度器在训练神经网络模型时非常有用，它可以帮助你在训练的不同阶段动态地调整学习率，从而提高训练的效果和稳定性。

调整学习率可以有助于在训练初期使用较大的学习率来更快地逼近全局最优解，然后在接近最优解时使用较小的学习率以更精细地调整参数，从而提高收敛速度和训练质量。

`get_scheduler` 函数允许你获取不同类型的学习率调度器，具体取决于训练过程中你希望使用的策略。常见的学习率调度器包括：

- `get_scheduler("constant_schedule")`：使用恒定的学习率，不进行学习率衰减。
- `get_scheduler("warmup_constant")`：在训练初期进行学习率的线性预热，然后保持恒定的学习率。
- `get_scheduler("warmup_linear")`：在训练初期进行学习率的线性预热，然后根据线性衰减进行学习率衰减。
- `get_scheduler("warmup_cosine")`：在训练初期进行学习率的线性预热，然后根据余弦函数进行学习率衰减。

此外，还有其他类型的学习率调度器可供选择，每种调度器都可以根据你的需求进行参数配置。

使用示例：

```python
from transformers import get_scheduler

scheduler = get_scheduler(
    "warmup_linear",
    optimizer,
    num_warmup_steps=500,
    num_training_steps=10000
)
```

在上述示例中，我们使用了 `get_scheduler` 函数获取了一个使用线性预热和线性衰减的学习率调度器。具体的调度器类型和参数需要根据训练需求来选择。

### Q: get_scheduler有哪些参数？

`get_scheduler` 函数接受多个参数，用于配置不同类型的学习率调度器。以下是常见的参数列表：

1. **name_or_type** (*str*): 要获取的学习率调度器的名称或类型。可以是以下字符串之一：
   - `"constant_schedule"`: 恒定学习率，不进行学习率衰减。
   - `"constant_schedule_with_warmup"`: 先进行学习率预热，然后保持恒定的学习率。
   - `"warmup_linear"`: 学习率线性预热，然后线性衰减。
   - `"warmup_cosine"`: 学习率线性预热，然后余弦衰减。
   - `"warmup_cosine_with_hard_restarts"`: 带有硬重启的余弦学习率预热和衰减。
   - `"polynomial_decay"`: 多项式学习率衰减。
   - `"linear_schedule_with_warmup"`: 线性学习率预热，然后线性衰减。
   - 或者是自定义的学习率调度器类名。
   
2. **optimizer** (*torch.optim.Optimizer*): 要应用学习率调度器的优化器。

3. **num_warmup_steps** (*int, optional*): 预热步数，即在此步数之前进行线性或余弦预热。

4. **num_training_steps** (*int*): 总的训练步数。

5. **num_cycles** (*float, optional*): 在 `"warmup_cosine_with_hard_restarts"` 调度器中，指定余弦衰减的循环次数。

6. **num_decay_steps** (*int, optional*): 在 `"polynomial_decay"` 调度器中，指定多项式衰减的步数。

7. **end_learning_rate** (*float, optional*): 在 `"polynomial_decay"` 调度器中，指定衰减后的学习率。

8. **power** (*float, optional*): 在 `"polynomial_decay"` 调度器中，指定多项式衰减的幂次。

9. **cycle_limit** (*int, optional*): 在 `"warmup_cosine_with_hard_restarts"` 调度器中，指定每个循环的最大步数。

10. **first_cycle_steps** (*int, optional*): 在 `"warmup_cosine_with_hard_restarts"` 调度器中，指定第一个循环的步数。

11. **warmup_ratio** (*float, optional*): 在 `"constant_schedule_with_warmup"` 或 `"linear_schedule_with_warmup"` 调度器中，指定预热的比例。

12. **warmup_steps** (*int, optional*): 在 `"constant_schedule_with_warmup"` 或 `"linear_schedule_with_warmup"` 调度器中，指定预热的步数。

这些参数允许你根据训练需求选择不同的学习率调度器类型，并对其行为进行配置。调度器的具体参数和行为取决于所选的调度器类型。

### 3.4.2 The training loop

### Q: 解释 from tqdm.auto import tqdm

`from tqdm.auto import tqdm` 是用于在 Python 脚本或 Jupyter Notebook 中实现进度条显示的方法之一。`tqdm` 是 "taqaddum"（阿拉伯语中的 "进展"）的缩写，它是一个用于显示循环进度的库。

`from tqdm.auto import tqdm` 中的 `auto` 表示自动选择适合当前环境的版本，可能是终端环境或 Jupyter 环境。

具体来说，`tqdm` 可以在循环中添加一个进度条，让你在执行迭代操作时实时查看进度。这对于处理大数据集或需要较长时间运行的任务非常有用，因为它能够提供实时反馈，让你了解代码执行的进展情况。

使用示例：

```python
from tqdm.auto import tqdm
import time

# 创建一个迭代范围
for i in tqdm(range(10)):
    # 模拟耗时操作
    time.sleep(1)
```

在上述示例中，`tqdm` 会在循环中显示一个进度条，每次迭代都会更新进度。这使你能够轻松地了解循环的进行情况。

在 Jupyter Notebook 环境中，使用 `tqdm.notebook` 也可以实现类似的进度条效果，但需要注意，如果你的代码在不同的环境中运行，最好使用 `from tqdm.auto import tqdm` 来保证适配不同环境。

### Q: tqdm有哪些参数？

`tqdm` 是一个功能强大的进度条库，提供了许多参数来自定义进度条的外观和行为。以下是一些常见的参数：

1. **iterable** (*iterable*): 要迭代的对象，可以是一个范围、列表、生成器等。

2. **desc** (*str, optional*): 进度条的描述，显示在进度条之前。

3. **total** (*int, optional*): 迭代对象的总长度，用于计算进度。

4. **leave** (*bool, optional*): 是否保留进度条在迭代完成后的显示，默认为 `False`，即迭代完成后进度条消失。

5. **position** (*int, optional*): 进度条显示的位置，默认为 `None`，根据屏幕大小自动选择。

6. **ncols** (*int, optional*): 进度条的宽度，以字符数为单位。

7. **mininterval** (*float, optional*): 刷新进度条的最小时间间隔，以秒为单位。

8. **maxinterval** (*float, optional*): 刷新进度条的最大时间间隔，以秒为单位。

9. **dynamic_ncols** (*bool, optional*): 是否动态调整进度条宽度以适应终端大小。

10. **position** (*int, optional*): 进度条在终端中的位置。

11. **colour** (*bool or str, optional*): 控制是否为进度条使用颜色。

12. **disable** (*bool, optional*): 是否禁用进度条，如果设置为 `True`，进度条将不会显示。

13. **unit** (*str, optional*): 迭代项的单位，显示在进度条右侧。

14. **unit_scale** (*bool, optional*): 是否自动调整单位的缩放。

15. **unit_divisor** (*int, optional*): 缩放单位的因子。

16. **bar_format** (*str, optional*): 控制进度条的显示格式。

17. **postfix** (*dict or OrderedDict, optional*): 在进度条后添加自定义文本。

18. **initial** (*int, optional*): 初始迭代数。

19. **position** (*int, optional*): 进度条在终端中的位置。

20. **dynamic_ncols** (*bool, optional*): 是否根据终端宽度动态调整进度条宽度。

等等。

这些参数可以根据你的需求来调整进度条的显示和行为，使你能够将进度条集成到你的代码中，并根据需要进行自定义。

### Q: .items()是什么意思？

`.items()` 是 Python 字典（Dictionary）对象的一个方法，用于获取字典中所有的键值对（键和对应的值）。这个方法返回一个包含键值对的迭代器，你可以用来遍历字典中的每个键值对。

示例代码：

```python
my_dict = {'a': 1, 'b': 2, 'c': 3}
print(my_dict.items())
```

最终输出结果是: dict_items([('a', 1), ('b', 2), ('c', 3)]) . 可以看出是一群tuple组成的list. 

使用 `.items()` 方法可以在循环中轻松遍历字典的键值对，这在需要处理字典中的所有数据时非常有用。

### Q: 为什么只移动v而不移动k?

在这段代码中，只移动值 `v` 而不移动键 `k`，是因为键（key）通常是标识数据的标签，它们是元数据而不是需要进行计算的实际数据。在这种情况下，通常只需要将值（例如张量）移动到指定的设备上，以便在该设备上进行模型计算。

键（标签）并不需要在设备之间移动，因为它们只是用于标识数据的标签，不直接参与模型的计算。键在字典中起到索引作用，用于访问对应的值。所以，一般情况下，只需要将值移动到指定设备上，以便在该设备上进行实际计算，而键通常不需要进行设备间的移动。

如果键也需要在设备之间移动，可以使用类似的方法，但在大多数情况下，只需要对值进行设备间的移动即可。

### Q: loss除了backward() 函数还有哪些常用的函数？

在 PyTorch 中，除了 `backward()` 函数用于计算梯度，还有一些与损失函数相关的常用函数。以下是一些常见的损失函数相关函数：

1. **`backward()`**: 用于计算张量关于梯度的反向传播。在优化过程中，通常先计算损失，然后调用此函数计算梯度。

2. **`item()`**: 将只包含一个元素的标量张量转换为 Python 数值。通常在计算损失后使用，以便获取可读性更好的损失值。

3. **`mean()` 和 `sum()`**: 这些函数用于计算张量的平均值和总和。在计算损失函数时，经常会用到这些函数。

4. **`detach()`**: 从计算图中分离出一个张量，使其不再具有梯度。通常用于生成目标标签或比较值，避免将其作为输入传递给模型。

5. **`log_softmax()` 和 `softmax()`**: 用于计算张量的对数 softmax 和 softmax。在分类问题中，这些函数通常用于计算预测概率。

6. **`sigmoid()`**: 计算张量的 sigmoid 函数值。在二分类问题中，它用于计算输出概率。

7. **`mse_loss()`**: 计算均方误差（Mean Squared Error）损失，用于回归问题。

8. **`cross_entropy_loss()`**: 计算交叉熵损失，通常用于分类问题。

9. **`l1_loss()`**: 计算 L1 损失，也称为绝对值损失，用于一些回归问题。

10. **`smooth_l1_loss()`**: 计算平滑 L1 损失，一种介于 L1 损失和 L2 损失之间的损失函数，用于一些回归问题。

这些函数是深度学习中常用的损失函数相关函数，可以根据任务的需求选择适当的函数来计算损失、梯度以及模型参数的更新。

### Q: 解释optimizer.step()

`optimizer.step()` 是在 PyTorch 中用于更新模型参数的方法。在深度学习中，优化算法通常通过调整模型的参数来最小化损失函数，从而提升模型性能。

具体来说，解释 `optimizer.step()` 如下：

- `optimizer` 是一个优化器对象，例如 Adam、SGD 等，用于执行参数优化的算法。

- `optimizer.step()` 是调用该方法来执行优化步骤，更新模型的参数。它在模型计算的反向传播（`.backward()`）之后使用，以根据梯度信息更新模型的权重。

这个操作会根据优化算法的策略，沿着梯度的反方向更新模型的参数，从而使得损失函数的值逐渐减小。不同的优化算法有不同的更新策略，例如 Adam 使用动量和学习率调整等。

示例代码：

```python
import torch
import torch.optim as optim

# 创建一个简单的模型和优化器
model = torch.nn.Linear(2, 1)  # 一个简单的线性模型
optimizer = optim.SGD(model.parameters(), lr=0.01)  # SGD 优化器

# 假设已经计算得到了损失 loss
loss = ...  # 计算得到的损失

# 执行梯度计算和参数更新
loss.backward()  # 计算梯度
optimizer.step()  # 更新模型参数
```

在上述示例中，`optimizer.step()` 被用来执行模型参数的更新，以便使损失函数逐渐减小，模型逐渐拟合训练数据。

### Q: optimizer除了step() 函数还有哪些常用的函数？

在 PyTorch 中，优化器对象除了 `step()` 方法用于更新模型参数之外，还有一些其他常用的函数来进行优化操作。以下是一些常见的优化器函数：

1. **`zero_grad()`**: 用于将模型参数的梯度归零。在每次优化步骤之前，通常会调用这个函数来清除之前的梯度信息，以便计算新的梯度。

2. **`state_dict()` 和 `load_state_dict()`**: 这两个函数用于保存和加载优化器的状态字典。状态字典包含了优化器的配置和参数，可以用于恢复训练的状态。

3. **`add_param_group()`**: 用于将额外的参数组（参数和超参数）添加到优化器中，例如在模型的不同部分使用不同的学习率。

4. **`get_lr()` 和 `set_lr()`**: 这些函数用于获取和设置当前的学习率。某些优化器支持动态调整学习率，这些函数可以用来查询和修改学习率。

5. **`state` 和 `param_groups`**: 这些属性分别返回优化器的状态字典和参数组列表。你可以使用它们来查看和修改优化器的状态和参数。

6. **`step(closure)`**: 在一些优化器中，`step()` 方法还接受一个 `closure` 参数，它是一个可选的闭包函数，用于计算损失并执行反向传播。这在某些情况下可以简化优化步骤的调用。

这些函数用于在训练过程中更精细地控制优化器的行为，管理梯度、参数以及学习率等信息。你可以根据训练任务的需求，选择适合的函数来进行优化器的操作和调整。

### Q: lr_scheduler.step()是干啥？

`lr_scheduler.step()` 是在 PyTorch 中用于调整学习率的方法。学习率调度器（Learning Rate Scheduler）用于在训练过程中自动地降低或调整学习率，以提升优化算法的效果。

具体来说，解释 `lr_scheduler.step()` 如下：

- `lr_scheduler` 是一个学习率调度器对象，例如 StepLR、ReduceLROnPlateau 等，用于根据指定策略调整学习率。

- `lr_scheduler.step()` 是调用该方法来执行学习率的调整，通常在每个训练周期结束时调用。

调用 `lr_scheduler.step()` 会根据学习率调度器的策略更新学习率，这可以根据训练过程的进展来自动调整学习率，以达到更好的训练效果。不同的学习率调度器有不同的策略，例如在一定周期内降低学习率，或者根据验证集性能自动调整学习率等。

示例代码：

```python
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

# 创建一个简单的优化器和学习率调度器
optimizer = optim.SGD(model.parameters(), lr=0.1)
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

# 在每个训练周期结束时调用学习率调度器
for epoch in range(epochs):
    train_model()
    scheduler.step()
```

在上述示例中，`scheduler.step()` 会在每个训练周期结束时调用，根据 `step_size` 和 `gamma` 参数来自动调整学习率。这样可以在训练过程中逐步减小学习率，以便更好地优化模型。

### Q: optimizer.zero_grad()是干啥？

`optimizer.zero_grad()` 是在 PyTorch 中用于清零模型参数梯度的方法。在深度学习中，模型参数的梯度在每次反向传播之后会被累积，以便进行参数更新。然而，在每次迭代之前，通常需要将这些累积的梯度清零，以避免梯度的不正确累积。

具体来说，解释 `optimizer.zero_grad()` 如下：

- `optimizer` 是一个优化器对象，例如 Adam、SGD 等，用于执行参数优化的算法。

- `zero_grad()` 是调用该方法来将模型参数的梯度清零，以便为下一次迭代做准备。

在进行反向传播之前，通常会先调用 `zero_grad()`，然后再计算梯度和执行反向传播。这样可以确保每次迭代都使用当前批次数据的梯度，而不受之前迭代的影响。

示例代码：

```python
import torch.optim as optim

# 创建一个优化器
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 在每次迭代之前清零梯度
optimizer.zero_grad()

# 执行梯度计算和反向传播
loss.backward()

# 执行参数更新
optimizer.step()
```

在上述示例中，`optimizer.zero_grad()` 用于在每次迭代之前清零模型参数的梯度。这样，在下一次迭代中，模型参数的梯度将从零开始累积。这是模型优化的常见步骤之一，以确保梯度计算的准确性。

### Q: progress_bar.update(1)是干啥？

`progress_bar.update(1)` 是在代码中更新进度条的方式。进度条用于显示迭代过程中的进度，`tqdm` 库可以创建并管理这些进度条。

在这段代码中，`progress_bar` 是通过 `tqdm` 库创建的一个进度条对象。`update(1)` 方法用于增加进度条的当前值，将进度条向前推进一步。这样做是为了在训练过程中实时地显示进度条的进展。

具体来说，`progress_bar.update(1)` 表示进度条的当前值增加 1，从而更新进度条的显示状态。这通常在每次完成一个训练批次或迭代时调用，以便用户能够实时地看到训练的进度。

综合来看，`progress_bar.update(1)` 用于更新进度条，以便实时显示模型训练的进展情况。这对于长时间的训练过程非常有用，可以提供对训练进度的直观反馈。

### 3.4.3 The evaluation loop

### Q: {'accuracy': 0.8431372549019608, 'f1': 0.8907849829351535}中的f1是什么意思？

在给定的上下文中，`f1` 是指 F1 得分，它是用于衡量二分类或多分类模型性能的一种指标。F1 得分综合了精确率（Precision）和召回率（Recall），是它们的调和平均值，用于综合考虑模型的准确性和覆盖率。

F1 得分的计算方式是：

\[ F1 = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} \]

其中，Precision 是正确预测为正类别的样本数与所有预测为正类别的样本数的比例，Recall 是正确预测为正类别的样本数与真实正类别的样本数的比例。

在二分类问题中，Precision 衡量了模型预测为正类别的样本中有多少是真正的正类别，而 Recall 衡量了模型正确识别了多少真正的正类别样本。F1 得分将这两者结合起来，适用于那些在不同 Precision 和 Recall 之间需要权衡的任务。

在多分类问题中，F1 得分可以分别计算每个类别的 Precision 和 Recall，并求取各个类别的 F1 得分的平均值，通常有宏平均（对每个类别求平均）和微平均（对每个样本的预测结果求平均）之分。

因此，在你提供的字典中，`'f1'` 表示 F1 得分，该得分为模型在某个任务中的综合性能指标，结合了模型的准确性和覆盖率。

### Q: model.eval() 是干啥？

`model.eval()` 是将深度学习模型切换到评估（evaluation）模式的方法。在评估模式下，模型的行为会发生一些变化，以便更好地适应评估任务。主要影响包括：

1. **Dropout 和 Batch Normalization的行为**：在训练模式下，Dropout 和 Batch Normalization 层会根据配置进行随机操作，有时会被禁用或减小，以防止过拟合。在评估模式下，这些层通常会被保持原样，以保证结果的一致性和可重复性。

2. **梯度计算的关闭**：在评估模式下，通常不需要计算梯度，因为在评估阶段不进行反向传播优化。通过进入评估模式，可以减少内存开销，特别是在批处理大小较大的情况下。

3. **其他一些操作的关闭**：一些模型在评估模式下还可能关闭一些与训练无关的操作，以提高性能和效率。

通常，在使用模型进行推断、验证或测试数据时，将模型切换到评估模式是一个很好的做法，因为这确保了模型的行为与在训练时保持一致。这可以减少由于模型行为变化而引起的不一致性和错误。

在代码中的这个上下文中，`model.eval()` 是用于将模型切换到评估模式，以确保在验证数据上运行模型时，模型的行为是符合预期的。

### Q: from accelerate import Accelerator是在干啥？

`from accelerate import Accelerator` 是在使用 Hugging Face Transformers 库中的 Accelerate 库，用于加速深度学习模型的训练和推理过程。

Accelerate 是 Hugging Face 推出的一个用于加速深度学习训练的库，它可以自动处理 GPU 或多 GPU、TPU 等硬件加速，并且能够简化训练循环的编写。它为训练循环的加速提供了便捷的接口，同时支持混合精度训练（Mixed Precision Training）等加速技术。

通过使用 Accelerate，你可以更轻松地利用 GPU、TPU 等硬件来提高训练速度，并且无需手动编写许多硬件和训练循环相关的代码。

以下是 Accelerate 库的一些特点：

1. **多硬件支持**：支持单 GPU、多 GPU、TPU 等多种硬件环境。

2. **混合精度训练**：支持自动的混合精度训练，将参数存储为 FP16（半精度浮点数），从而减少显存占用和加快训练速度。

3. **加速训练循环**：通过 `Accelerator` 类，你可以将模型、数据加载器等与硬件相关的操作进行封装，从而简化训练循环的编写。

4. **分布式训练**：支持在多台机器上进行分布式训练，利用多个 GPU 或 TPU 进行训练。

5. **动态批处理**：自动根据硬件资源调整批处理大小，以最大限度地利用硬件性能。

6. **自动优化器**：支持自动选择优化器，如 `torch.optim.AdamW` 和 `tf.optimizers.AdamW`。

通过使用 Accelerate，你可以更轻松地将 Hugging Face Transformers 库中的模型训练过程与硬件加速技术相结合，从而提高训练速度和效率。

### Q: 如果在单gpu上训练，是否可以用accelerator？

是的，虽然 `accelerate` 库主要用于分布式训练、多 GPU 训练和混合精度训练等加速技术，但它同样适用于单 GPU 训练。即使在单 GPU 的情况下，你仍然可以使用 `accelerator` 来简化代码并获得一些额外的优势，例如自动的混合精度训练和训练加速。

当在单 GPU 上使用 `accelerator` 时，它会为你处理混合精度的设置、训练循环的优化以及一些额外的训练加速细节。这可以让你的代码更具有可扩展性，以便将来可以轻松地切换到多 GPU 或分布式环境。

以下是一个在单 GPU 上使用 `accelerator` 的示例：

```python
from accelerate import Accelerator

# 创建加速器对象
accelerator = Accelerator()

# 准备训练环境
accelerator.prepare()

# 定义模型、优化器等
model = ...
optimizer = ...

# 在训练循环中使用 accelerator 训练模型
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = accelerator.prepare(batch)  # 处理 batch 并将其移动到设备上
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

尽管你在单 GPU 上使用 `accelerator` 可能不会获得与多 GPU 训练相同的性能提升，但它仍然可以帮助你使代码更具有可扩展性，并且在未来进行训练环境的迁移时更加方便。

### Q: 什么是混合精度训练？

混合精度训练（Mixed Precision Training）是一种深度学习训练技术，旨在通过使用不同精度的浮点数来加速训练过程，从而减少计算和内存开销。具体而言，混合精度训练使用半精度浮点数（16 位）来表示模型参数和梯度，同时使用单精度浮点数（32 位）来计算和累积梯度，以及执行优化器的参数更新。

混合精度训练的核心思想是：

1. 使用半精度浮点数表示模型参数和梯度。由于半精度浮点数只占用 16 位，相对于单精度浮点数（32 位）来说，可以减少内存开销。

2. 在前向计算和反向传播过程中使用半精度浮点数。这可以加速计算过程，尤其是在 GPU 上。

3. 在参数更新时，使用单精度浮点数执行梯度计算和参数更新。这可以避免梯度下溢问题，同时保证更新的精度。

混合精度训练的优势在于可以在不损失模型训练效果的情况下，显著减少训练时间和内存开销。由于深度学习模型的训练通常需要大量的计算和存储资源，使用混合精度可以使模型训练更加高效。

PyTorch 提供了 AMP（Automatic Mixed Precision）机制，通过自动选择合适的精度来执行计算，简化了混合精度训练的实现过程。使用混合精度需要考虑数值稳定性和数值精度问题，但合理的设置可以带来显著的性能提升。

### Q: accelerator可以自动设置gpu吗？还是需要在accelerator.prepare()中手动指定gpu?

`accelerator.prepare()` 函数可以根据系统环境自动设置 GPU，不必手动指定。当你调用 `accelerator.prepare()` 时，`accelerate` 库会自动检测系统中是否有可用的 GPU，并将模型和数据加载到 GPU 上进行训练。

如果你想手动指定 GPU，可以在 `accelerator.prepare()` 函数中传递 `device` 参数，例如 `device="cuda"` 来指定使用 GPU。如果不传递 `device` 参数，`accelerate` 库将自动检测并使用可用的 GPU，如果没有 GPU，则会使用 CPU。

示例：

```python
from accelerate import accelerator

# 自动设置 GPU
accelerator.prepare()

# 手动指定 GPU
accelerator.prepare(device="cuda")
```

总之，`accelerator.prepare()` 函数具备自动设置 GPU 的功能，但你仍然可以根据需要手动指定 GPU。

### Q: deepspeed是什么？

DeepSpeed 是一个用于深度学习训练的优化库和框架，旨在加速和扩展模型训练过程。它提供了一系列优化技术和工具，以降低内存使用、加速训练速度并支持大规模模型的训练。DeepSpeed 的设计目标之一是简化分布式和混合精度训练，使研究人员和工程师能够更轻松地在大规模训练任务中使用高性能硬件。

以下是 DeepSpeed 提供的一些主要功能和特点：

1. **分布式训练**: DeepSpeed 支持分布式训练，允许在多台机器上训练大规模模型。它提供了数据并行和模型并行两种分布式训练策略。

2. **混合精度训练**: DeepSpeed 支持混合精度训练，使用半精度浮点数（16 位）加速训练，并通过减少内存和计算需求来提高效率。

3. **内存优化**: DeepSpeed 通过使用零 Redundancy Optimizer（ZeRO）技术，将模型参数和优化器状态分成多个部分，以减少每个部分的内存需求。这有助于训练更大的模型。

4. **自动调优**: DeepSpeed 提供了自动调优功能，可以自动调整训练过程中的超参数，以获得更好的性能。

5. **优化算法**: DeepSpeed 提供了一些优化算法，如分布式 Adam 和优化器合并等，以提高训练速度和效率。

6. **支持多种框架**: DeepSpeed 可以与多个深度学习框架（如 PyTorch、TensorFlow）一起使用。

7. **易于使用**: DeepSpeed 提供了简单的 API 和配置文件，使用户能够轻松地集成和使用其中的功能。

总之，DeepSpeed 是一个强大的工具，旨在帮助深度学习从业者充分利用现代硬件并加速模型训练过程。它尤其适用于需要训练大规模模型或在分布式环境中进行训练的场景。

### Q: 要想使用accelorate, 是不是必须用命令accelerate launch train.py？

不是必须使用命令 `accelerate launch train.py` 来使用 `accelerate` 库，但这是一种方便的方法，特别是在分布式训练环境下。`accelerate` 库提供了更多的灵活性，你可以根据自己的需要在代码中集成它。

使用 `accelerate launch` 命令可以简化一些分布式训练的设置，例如自动设置 GPU、分布式参数等。然而，如果你只是在单 GPU 或者简单的训练环境下使用 `accelerate`，你可以直接在代码中使用 `accelerate` 的 API 来配置和加速训练。

以下是一些使用 `accelerate` 的方法：

1. **使用 `accelerate launch` 命令**：对于分布式训练，你可以使用 `accelerate launch` 命令来运行你的训练脚本，它会自动设置分布式训练参数和 GPU。

2. **在代码中使用 `accelerate` API**：如果你想更加灵活地配置 `accelerate`，你可以在代码中使用 API，如 `accelerator.prepare` 函数来设置训练环境、加载器、模型等。这可以适用于单 GPU 或者不需要复杂分布式设置的情况。

3. **混合使用**：你也可以将 `accelerate launch` 与代码中的 `accelerate` API 结合使用，以便在需要的地方进行更详细的配置。

综上所述，虽然使用 `accelerate launch` 命令是一个方便的选项，但不是必须的。你可以根据自己的训练需求和环境来选择合适的使用方式。

## 3.5 Fine-tuning, Check!

## 3.6 End-of-chapter quiz

### Q: 怎样理解下面这句话？ when we used AutoModelForSequenceClassification with bert-base-uncased,  we got warnings when instantiating the model. The pretrained head is not used for the sequence classification task, so it's discarded and a new  head is instantiated with random weights.

这句话的含义是：

当我们使用 `AutoModelForSequenceClassification` 配合 `bert-base-uncased` 预训练模型时，在实例化模型时会收到警告。这是因为预训练的模型头部（classification head）并没有在序列分类任务中被使用，所以它会被丢弃，而一个新的头部会被实例化并初始化为随机权重。

更详细解释如下：

- `AutoModelForSequenceClassification` 是 Hugging Face Transformers 库中的一个类，用于加载适用于序列分类任务的预训练模型。
- `bert-base-uncased` 是 BERT 模型的一个变种，其中 "uncased" 表示该模型的输入文本已经转换为小写，不区分大小写。
- 在序列分类任务中，我们通常需要对预训练模型进行微调，以适应具体的分类任务。这涉及到连接一个适当的分类头部（classification head）到预训练模型上，以便执行分类操作。
- 在这个情况下，作者使用了 `bert-base-uncased` 预训练模型，但在实例化 `AutoModelForSequenceClassification` 时，会收到警告。这是因为 `bert-base-uncased` 预训练模型已经带有一个用于分类任务的头部，但是在序列分类任务中并没有使用这个预训练的头部。
- 结果就是，为了序列分类任务，原来的预训练头部被丢弃，而一个新的头部会被实例化，并且会使用随机初始化的权重。这样，模型就可以在微调过程中学习适应序列分类任务。

总之，这句话指出了在特定情况下，使用 `AutoModelForSequenceClassification` 时可能会出现的警告，以及为了序列分类任务而丢弃预训练头部并重新实例化新头部的操作。

### Q: 什么是预训练的模型头部（classification head）？

预训练的模型头部（classification head）是指在进行预训练任务时，模型顶部的一层或多层神经网络结构，用于执行特定任务，如分类、回归等。这些头部在预训练阶段被添加到预训练模型的顶部，并通过在预训练数据上进行训练，以学习特定任务的相关信息。

在预训练过程中，模型通常被设计为解决某个“辅助任务”，这个辅助任务可以是语言模型、掩码语言模型、下一句预测等。模型的顶部会有一个相应的头部，它负责执行这个辅助任务。例如，对于 BERT 模型，预训练的任务是“掩码语言建模”，其中部分输入被随机掩盖，模型需要预测被掩盖的单词是什么。模型的顶部会包含一个掩码语言模型头部，用于预测被掩盖词的概率分布。

然而，训练好的预训练模型通常还可以用于其他任务，如分类、命名实体识别、文本生成等。这时，通常会在预训练模型的顶部添加一个新的头部，用于执行这些任务。这个新的头部会连接到预训练模型的中间或最后一层，然后通过微调来适应特定任务的数据。

因此，预训练的模型头部就是预训练模型在特定任务上添加的一个用于处理该任务的神经网络结构。它可以是一个全连接层、卷积层等，具体结构和参数取决于任务的性质。在微调阶段，这个头部会通过反向传播和优化来学习适应任务的相关特征，以便模型能够在特定任务上获得良好的性能。

### Q: 为什么预训练的模型头部没有在序列分类任务中被使用？

在某些情况下，预训练的模型头部可能没有在序列分类任务中被使用的原因可能是为了适应特定任务的需求、避免冲突或提高模型性能。以下是一些可能的原因：

1. **任务差异性**：预训练任务和序列分类任务的目标和数据特征可能不同。预训练模型的头部通常被设计为解决特定的预训练任务，例如掩码语言模型或下一句预测。这些任务可能与序列分类任务的特性不匹配，因此预训练的头部可能无法直接适应序列分类。

2. **冲突或干扰**：预训练模型的头部可能在序列分类任务中干扰模型学习任务特定的特征。有时，预训练头部可能学习了与序列分类无关的特征，这可能对序列分类的性能产生负面影响。在这种情况下，舍弃预训练头部并重新初始化新的头部可能会更有利于模型的性能。

3. **优化和微调**：微调预训练模型时，可能希望模型的顶部能够根据序列分类任务的特性进行自由优化。这意味着在微调阶段，可以使用随机初始化的头部，使模型能够更好地适应序列分类任务的数据和特征。

4. **避免过拟合**：预训练模型的头部在预训练任务上进行了训练，可能已经学习了大量的参数。在序列分类任务中，直接使用这些参数可能会导致过拟合，特别是当训练数据相对较少时。通过重新初始化一个新头部，可以减少过拟合的风险。

综上所述，预训练的模型头部可能没有在序列分类任务中被使用，是为了使模型能够更好地适应特定的任务要求，提高性能，并避免干扰和冲突。根据任务的性质和数据的特点，决定是否使用预训练头部是一个需要仔细权衡的问题。

### Q: 如果预训练头部和fine-tune任务不同，是否应该使用预训练头部？

如果预训练的头部和微调任务不同，是否应该使用预训练头部取决于任务的性质、数据的特点以及实际的需求。以下是一些考虑因素：

1. **任务相似性**：尽管预训练头部和微调任务不同，但如果两者在某些方面具有相似性，预训练头部可能仍然有用。例如，如果预训练任务和微调任务都涉及文本处理，可能存在一些重叠的特征，可以通过预训练头部传递给微调任务。

2. **数据量和多样性**：如果微调任务的训练数据非常少，或者数据在特定领域中非常不足，使用预训练头部可以为模型提供更多的先验知识，有助于提高性能。预训练头部已经在大规模数据上进行了训练，可以帮助模型更好地泛化到少量微调数据。

3. **任务复杂性**：如果微调任务比预训练任务更为复杂，例如需要更深层次的特征提取，可能需要重新设计一个更适合微调任务的头部。在这种情况下，使用预训练头部可能不太合适，因为它可能无法满足更高级的任务需求。

4. **过拟合风险**：如果预训练头部在微调任务上过拟合，导致性能下降，可能需要考虑使用不同的头部。过拟合可能发生在预训练头部已经学习了与微调任务无关的特征，但它们在微调任务中无用。

5. **实验和调试**：在实际情况下，可以进行实验来比较使用预训练头部和不同头部的性能。通过微调多个模型并评估它们的性能，可以选择最适合微调任务的方法。

综上所述，如果预训练头部和微调任务不同，是否应该使用预训练头部取决于多个因素。需要考虑任务的性质、数据的特点、模型的性能以及实际需求。在一些情况下，可能需要根据实验结果进行权衡，选择最适合的策略。

# 4. SHARING MODELS AND TOKENIZERS

## 4.1 The Hugging Face Hub

## 4.2 Using pretrained models

The only thing you need to watch out for is that **the chosen checkpoint is suitable for the task it’s going to be used for**. For example, here we are loading the `camembert-base` checkpoint in the `fill-mask` pipeline, which is completely fine. But if we were to load this checkpoint in the `text-classification` pipeline, the results would not make any sense because the head of `camembert-base` is not suitable for this task! We recommend using the task selector in the Hugging Face Hub interface in order to select the appropriate checkpoints.

## 4.3 Sharing pretrained models

### 4.3.1 Using the `push_to_hub` API

### 4.3.2 Using the `huggingface_hub` Python library

### 4.3.3 Using the web interface

### 4.3.4 Uploading the model files

## 4.4 Building a model card

### 4.4.1 Model description

### 4.4.2 Intended uses & limitations

### 4.4.3 How to use

### 4.4.4 Training data

### 4.4.5 Training procedure

### 4.4.6 Variable and metrics

### 4.4.7 Evaluation results

### 4.4.8 Example

### 4.4.9 Note

### 4.4.10 Model card metadata

## 4.5 Part 1 completed!

## 4.6 End-of-chapter quiz

# 5 THE 🤗 DATASETS LIBRARY

## 5.1 Introduction

## 5.2 What if my dataset isn't on the Hub?

### 5.2.1 Working with local and remote datasets

csv，text，json，pandas

### 5.2.2 Loading a local dataset

### 5.2.3 Loading a remote dataset

## 5.3 Time to slice and dice

### 5.3.1 Slicing and dicing our data

A good practice when doing any sort of data analysis is to **grab a small random sample** to get a quick feel for the type of data you’re working with.

### 5.3.2 Creating new columns

### 5.3.3 The `map()` method's superpowers

### 5.3.4 From `Dataset`s to `DataFrame`s and back

### 5.3.5 Creating a validation set

### 5.3.6 Saving a dataset

## 5.4 Big data? 🤗 Datasets to the rescue!

### 5.4.1 What is the Pile?

### 5.4.2 The magic of memory mapping

### 5.4.3 Streaming datasets

## 5.5 Creating your own dataset

### 5.5.1 Getting the data

### 5.5.2 Cleaning up the data

### 5.5.3 Augmenting the dataset

### 5.5.4 Uploading the dataset to the Hugging Face Hub

### 5.5.5 Creating a dataset card

## 5.6 Semantic search with FAISS

### Q: faiss-gpu是什么？

Faiss-GPU（GPU加速的Facebook AI Similarity Search）是一个用于高效相似性搜索的库，它通过利用图形处理单元（GPU）来加速向量之间的相似性搜索操作。Faiss是Facebook AI Research开发的开源库，旨在加速大规模向量集合上的相似性搜索，这在机器学习、深度学习和信息检索等领域非常有用。

Faiss-GPU 使用了GPU来加速搜索操作，因此可以在大型向量集合上更快地执行诸如K最近邻搜索（K-Nearest Neighbors，KNN）和聚类等任务。这对于许多机器学习应用程序中的特征向量检索和相似性搜索非常有用，例如图像检索、推荐系统、自然语言处理等。

Faiss-GPU通常与NVIDIA的GPU一起使用，因为NVIDIA GPU提供了高性能的并行计算能力，适用于这种类型的计算密集型任务。如果您需要在大规模数据集上进行相似性搜索并且拥有适当的GPU硬件，Faiss-GPU可以帮助您加速这些操作，提高性能和效率。它的使用通常需要一定的编程和配置技能，以便有效地集成到您的应用程序中。

### 5.6.1 Using embeddings for semantic search

### 5.6.2 Loading and preparing the dataset

### 5.6.3 Creating text embeddings

### 5.6.4 Using FAISS for efficient similarity search

## 5.7 Datasets, check!

## 5.8 End-of-chapter quiz

#  6 THE 🤗 TOKENIZERS LIBRARY

## 6.1 Introduction

## 6.2 Training a new tokenizer from an old one

### 6.2.1 Assembling a corpus

### Q: git-lfs是什么？

Git LFS（Git Large File Storage）是一个Git扩展，用于管理大型文件。它的主要目的是解决Git版本控制系统对大型二进制文件（如图像、音频、视频、数据集等）的存储和版本控制问题。通常，这些大型二进制文件占用大量存储空间，并且在每次提交和克隆操作时都需要传输，这会导致Git仓库变得庞大和不稳定。

Git LFS通过将大型文件替换为文本指针，并将实际文件存储在一个独立的存储系统中（如Git LFS服务器或云存储），有效地解决了这个问题。这些文本指针被包含在Git仓库中，而实际的大型文件则保存在外部位置，只在需要时才下载。这有助于减小Git仓库的大小，提高性能，并加速克隆和提交操作。

Git LFS的基本工作流程如下：

1. 安装Git LFS：首先，您需要安装Git LFS扩展，以便在Git仓库中使用它。

2. 指定要追踪的大型文件：通过命令行或配置文件，您可以指定要由Git LFS进行管理的大型文件扩展名或文件名模式。

3. 提交文件：将大型文件添加到Git仓库并提交。Git LFS将文件替换为文本指针，并将实际文件存储在外部存储中。

4. 下载文件：在需要时，Git LFS会自动下载大型文件的内容，使其在您的工作目录中可用。

Git LFS支持多种后端存储，包括自托管的Git LFS服务器、GitHub、GitLab等。这使得它非常适合需要管理大型文件的项目，如游戏开发、多媒体制作、数据科学和其他需要版本控制的二进制文件的领域。

。。。。。。。。。

。。。。。。。。。

。。。。。。。。。

# 7 MAIN NLP TASKS

## 7.1 Introduction

在第 [3 章](https://huggingface.co/course/chapter3)中，您了解了如何微调文本分类模型。在本章中，我们将处理以下常见的 NLP 任务：

- 令牌分类 -- 无关
- 掩码语言建模（如 BERT）
- 综述 -- 有关
- 译本 -- 有关
- 因果语言建模预训练（如 GPT-2）-- 可能有关
- 问答 -- 有关

为此，您需要利用您在第 3 章中学到的有关 API 和 🤗 Accelerate 库、🤗第 [5](https://huggingface.co/course/chapter5) 章中的 Datasets 库以及🤗[第 6 章](https://huggingface.co/course/chapter6)中的 Tokenizers 库的所有知识。我们还将把结果上传到模型中心，就像我们在第 4 章中所做的那样，所以这一[章](https://huggingface.co/course/chapter4)真的是所有东西汇集在一起的章节！`Trainer`

每个部分都可以独立阅读，并将向您展示如何使用 `Trainer` API 或您自己的训练循环（使用 🤗 Accelerate）训练模型。随意跳过任何一个部分，专注于你最感兴趣的部分：`Trainer` API 非常适合微调或训练你的模型，而不必担心幕后发生了什么，而`Accelerate`训练循环将让你更轻松地自定义任何你想要的部分。

如果你按顺序阅读这些部分，你会注意到它们有相当多的代码和散文的共同点。重复是有意为之的，目的是让你潜入（或稍后再回来）任何你感兴趣的任务，并找到一个完整的工作示例。

## 7.2 Token classification

。。。。。。。。。

。。。。。。。。。

。。。。。。。。。

## 7.3 Fine-tuning a masked language model

对于许多涉及 Transformer 模型的 NLP 应用程序，您只需从 Hugging Face Hub 中获取一个预训练模型，然后直接根据手头任务的数据对其进行微调。如果用于预训练的语料库与用于微调的语料库没有太大区别，迁移学习通常会产生良好的结果。

但是，在某些情况下，您需要先对数据上的语言模型进行微调，然后再训练特定于任务的头部。例如，如果您的数据集包含法律合同或科学文章，则像 BERT 这样的普通 Transformer 模型通常会将语料库中特定于领域的单词视为稀有标记，并且最终的性能可能不尽如人意。通过对域内数据的语言模型进行微调，您可以提高许多下游任务的性能，这意味着您通常只需执行此步骤一次！

这种在域内数据上微调预训练语言模型的过程通常称为*域适应*。它于 2018 年由 ULMFiT 推广，[ULMFiT](https://arxiv.org/abs/1801.06146) 是最早使迁移学习真正适用于 NLP 的神经架构（基于 LSTM）之一。下图显示了使用 ULMFiT 进行域适应的示例;在本节中，我们将做类似的事情，但使用 Transformer 而不是 LSTM！

![ULMFiT.](https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter7/ulmfit.svg)

在本节结束时，你将在 Hub 上有一个[掩码语言](https://huggingface.co/huggingface-course/distilbert-base-uncased-finetuned-imdb?text=This+is+a+great+[MASK].)模型，该模型可以自动完成句子，如下所示：

[网页API]

让我们开始吧！

。。。。。。。。。

。。。。。。。。。

。。。。。。。。。

## 7.4 Translation

现在让我们深入了解翻译。这是另一个序列到序列的任务，这意味着这是一个可以表述为从一个序列[到另一个序列](https://huggingface.co/course/chapter1/7)的问题。从这个意义上说，这个问题非常接近[于总结](https://huggingface.co/course/chapter7/6)，你可以将我们在这里看到的内容调整为其他序列到序列的问题，例如：

- 风格转移：创建一个模型，将以某种风格编写的文本*翻译*成另一种**风格**（例如，正式到休闲或莎士比亚英语到现代英语）
- 生成式问答：创建一个模型，该模型在给定上下文中生成问题的答案

(Comment:  这俩可能都和PCF相关)

[视频]

如果你有足够大的两种（或更多）语言的文本语料库，你可以从头开始训练一个新的翻译模型，就像我们在[因果语言建模](https://huggingface.co/course/chapter7/6)一节中所做的那样。但是，微调现有的翻译模型会更快，无论是像 mT5 或 mBART 这样的多语言模型，您希望微调到特定的语言对，还是专门用于从一种语言到另一种语言的翻译模型，您希望根据您的特定语料库进行微调。

在本节中，我们将在 [KDE4 数据集](https://huggingface.co/datasets/kde4)上微调一个经过预训练的 Marian 模型，以便从英语翻译成法语（因为很多 Hugging Face 员工会说这两种语言），该数据集是 [KDE 应用程序](https://apps.kde.org/)的本地化文件数据集。我们将使用的模型已经在从 [Opus](https://opus.nlpl.eu/) 数据集中获取的大量法语和英语文本语料库上进行了预训练，该语料库实际上包含 KDE4 数据集。但是，即使我们使用的预训练模型在预训练期间看到了这些数据，我们也会看到，在微调后，我们可以得到更好的版本。

一旦我们完成，我们将有一个模型能够进行这样的预测：

[网页API]

与前面的部分一样，你可以找到我们将使用以下代码训练并上传到 Hub 的实际模型，并[在此处](https://huggingface.co/huggingface-course/marian-finetuned-kde4-en-to-fr?text=This+plugin+allows+you+to+automatically+translate+web+pages+between+several+languages.)仔细检查其预测。

### 7.4.1 Preparing the data

要从头开始微调或训练翻译模型，我们需要一个适合该任务的数据集。如前所述，我们将在本节中使用 [KDE4 数据集](https://huggingface.co/datasets/kde4)，但您可以非常轻松地调整代码以使用您自己的数据，只要您有要翻译的两种语言的句子对。Refer back to [Chapter 5](https://huggingface.co/course/chapter5) if you need a reminder of how to load your custom data in a `Dataset`. 

#### KDE4 数据集

像往常一样，我们使用`load_dataset()`函数下载数据集：

```python
from datasets import load_dataset

raw_datasets = load_dataset("kde4", lang1="en", lang2="fr")
```

如果要使用不同的语言对，可以通过它们的代码来指定它们。该数据集共有 92 种语言可供选择;您可以通过展开[其数据集卡](https://huggingface.co/datasets/kde4)上的语言标签来查看它们。

![可用于 KDE4 数据集的语言。](https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter7/language_tags.png)

让我们看一下数据集：

```python
print(raw_datasets)
```

DatasetDict({
    train: Dataset({
        features: ['id', 'translation'],
        num_rows: 210173
    })
})

我们有 210,173 对句子，但在一个单一的拆分中，因此我们需要创建自己的验证集。正如我们[在第 5 章](https://huggingface.co/course/chapter5)中看到的，`Dataset`有一种 `train_test_split()` 方法可以帮助我们。我们将提供可重复性的种子：

```python
split_datasets = raw_datasets["train"].train_test_split(train_size=0.9, seed=20)
print(split_datasets)
```

DatasetDict({
    train: Dataset({
        features: ['id', 'translation'],
        num_rows: 189155
    })
    test: Dataset({
        features: ['id', 'translation'],
        num_rows: 21018
    })
})

(Comment:  由于这个dataset没有专门的test-set和validation-set, 只有一个大大的train, 因此需要从train中分离一部分出来作为test)

我们可以将`"test"`密钥重命名为`"validation"`：

```python
split_datasets["validation"] = split_datasets.pop("test")
```

现在让我们看一下数据集的一个元素：

```python
split_datasets["train"][1]["translation"]
```

{'en': 'Default to expanded threads',
 'fr': 'Par défaut, développer les fils de discussion'}

我们得到一本字典，里面有我们要求的两种语言的两个句子。这个充满计算机科学技术术语的数据集的一个特点是，它们都完全翻译成法语。然而，法国工程师在交谈时会留下大多数计算机科学特定的单词。例如，在这里，“线程”一词很可能出现在法语句子中，尤其是在技术对话中;但在这个数据集中，它已被翻译成更正确的“fils de discussion”。我们使用的预训练模型已经在更大的法语和英语句子语料库上进行了预训练，它采用了更简单的选择，即保持单词原样：

```python
from transformers import pipeline

model_checkpoint = "Helsinki-NLP/opus-mt-en-fr"
translator = pipeline("translation", model=model_checkpoint)
translator("Default to expanded threads")
```

[{'translation_text': 'Par défaut pour les threads élargis'}]

这种行为的另一个例子可以在“plugin”这个词中看到，它不是正式的法语单词，但大多数母语人士都会理解并且懒得翻译。 在 KDE4 数据集中，这个词被翻译成法语，翻译成更官方的“module d'extension”：

```python
split_datasets["train"][172]["translation"]
```

{'en': 'Unable to import %1 using the OFX importer plugin. This file is not the correct format.',
 'fr': "Impossible d'importer %1 en utilisant le module d'extension d'importation OFX. Ce fichier n'a pas un format correct."}

然而，我们的预训练模型坚持使用紧凑而熟悉的英语单词：

```python
translator(
    "Unable to import %1 using the OFX importer plugin. This file is not the correct format."
)
```

[{'translation_text': "Impossible d'importer %1 en utilisant le plugin d'importateur OFX. Ce fichier n'est pas le bon format."}]

看看我们微调的模型是否能抓住数据集的这些特殊性，这将是一件有趣的事情（剧透警告：它会的）。

[视频]

✏️ **该你了！**法语中经常使用的另一个英语单词是“电子邮件”。在训练数据集中查找使用此词的第一个样本。它是如何翻译的？预训练模型如何翻译相同的英语句子？

#### 处理数据

您现在应该知道了演练：所有文本都需要转换为令牌 ID 集，以便模型可以理解它们。对于此任务，我们需要对输入和目标进行标记化。我们的首要任务是创建我们的 `tokenizer` 对象。如前所述，我们将使用 Marian 英语到法语的预训练模型。如果要使用另一对语言尝试此代码，请确保调整模型检查点。[赫尔辛基-NLP](https://huggingface.co/Helsinki-NLP)组织提供了一千多种语言的模型。

```python
from transformers import AutoTokenizer

model_checkpoint = "Helsinki-NLP/opus-mt-en-fr"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, return_tensors="pt")
```

You can also replace the `model_checkpoint` with any other model you prefer from the [Hub](https://huggingface.co/models), or a local folder where you’ve saved a pretrained model and a tokenizer. 

💡 如果您使用的是多语言分词器，例如 mBART、mBART-50 或 M2M100，则需要通过设置 `tokenizer.src_lang` 和 `tokenizer.tgt_lang` 来设置输入和目标的语言代码。

我们的数据准备非常简单。只有一件事要记住: 您需要确保分词器以输出语言（此处为法语）处理目标。您可以通过将目标传递给分词器 `__call__` 方法的 `text_targets` 参数来执行此操作。

为了了解这是如何工作的，让我们处理训练集中每种语言的一个示例：

```python
en_sentence = split_datasets["train"][1]["translation"]["en"]
fr_sentence = split_datasets["train"][1]["translation"]["fr"]

inputs = tokenizer(en_sentence, text_target=fr_sentence)
print(inputs)
```

{'input_ids': [47591, 12, 9842, 19634, 9, 0], 'attention_mask': [1, 1, 1, 1, 1, 1], 'labels': [577, 5891, 2, 3184, 16, 2542, 5, 1710, 0]}

正如我们所看到的，输出包含与英语句子关联的输入 ID，而与法语句子关联的 ID 存储在`labels`字段中。如果您忘记指示您正在标记标签，它们将由输入标记器进行标记化，在 Marian 模型的情况下，这根本不会顺利进行：

```python
wrong_targets = tokenizer(fr_sentence)
print(tokenizer.convert_ids_to_tokens(wrong_targets["input_ids"]))
print(tokenizer.convert_ids_to_tokens(inputs["labels"]))
```

['▁Par', '▁dé', 'f', 'aut', ',', '▁dé', 've', 'lop', 'per', '▁les', '▁fil', 's', '▁de', '▁discussion', '</s>']
['▁Par', '▁défaut', ',', '▁développer', '▁les', '▁fils', '▁de', '▁discussion', '</s>']

正如我们所看到的，使用英语分词器来预处理法语句子会产生更多的标记，因为分词器不知道任何法语单词（除了那些也出现在英语中的单词，如“讨论”）。

由于 `inputs` 是带有我们常用键（输入 ID、注意力掩码等）的字典，因此最后一步是定义我们将应用于数据集的预处理函数：

```python
max_length = 128


def preprocess_function(examples):
    inputs = [ex["en"] for ex in examples["translation"]]
    targets = [ex["fr"] for ex in examples["translation"]]
    model_inputs = tokenizer(
        inputs, text_target=targets, max_length=max_length, truncation=True
    )
    return model_inputs
```

请注意，我们为输入和输出设置了相同的最大长度。由于我们正在处理的文本看起来很短，因此我们使用 128。

💡 如果您使用的是 T5 模型（更具体地说，`t5-xxx` 是其中一个检查点），则该模型将期望文本输入具有指示手头任务的前缀，例如 `translate: English to French:`。

⚠️ 我们不注意目标的注意力掩码，因为模型不会预料到它。相反，应设置与填充标记对应的标签为`-100`，以便在损失计算中忽略它们。由于我们正在应用动态填充，因此稍后将由我们的数据整理器完成，但如果您在此处使用填充，则应调整预处理函数以将与填充标记对应的所有标签设置为 `-100`。

现在，我们可以一次性将该预处理应用于数据集的所有拆分：

```python
tokenized_datasets = split_datasets.map(
    preprocess_function,
    batched=True,
    remove_columns=split_datasets["train"].column_names,
)
```

现在数据已经过预处理，我们准备微调我们的预训练模型了！

### 7.4.2 Fine-tuning the model with the Trainer API

`Trainer`使用 的实际代码将与以前相同，只是有一个小改动：我们在这里使用 [`Seq2SeqTrainer`](https://huggingface.co/transformers/main_classes/trainer.html#seq2seqtrainer)，它是`Trainer`的子类，它将允许我们正确处理评估，使用 `generate()` 方法来预测输入的输出。当我们谈论指标计算时，我们将更详细地探讨这一点。

首先，我们需要一个实际的模型来微调。我们将使用通常的 `AutoModel` API：

```python
from transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
```

请注意，这次我们使用的是在翻译任务上训练的模型，并且实际上已经可以使用，因此没有关于缺少权重或新初始化的权重的警告。

#### Data collation

我们需要一个数据整理器来处理动态批处理的填充。在这种情况下，我们不能像在第 [3 章](https://huggingface.co/course/chapter3)中使用 `DataCollatorWithPadding`，因为这只会填充输入（输入 ID、注意掩码和令牌类型 ID）。我们的标签也应填充到标签中遇到的最大长度。而且，如前所述，用于填充标签的填充值应该是分词器的填充值 `-100`，而不是填充标记器的填充标记，以确保在损失计算中忽略这些填充值。

这一切都是由[`DataCollatorForSeq2Seq`](https://huggingface.co/transformers/main_classes/data_collator.html#datacollatorforseq2seq)完成的。与 `DataCollatorWithPadding` 一样，它需要 `tokenizer` 用于预处理输入，但它也需要 `model` . 这是因为此数据整理器还将负责准备解码器输入 ID，这些 ID 是标签的移动版本，开头带有特殊标记。由于这种转变对于不同的架构略有不同，因此 `DataCollatorForSeq2Seq` 需要了解 `model` 对象：

```python
from transformers import DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
```

为了在几个示例上对此进行测试，我们只需在标记化训练集中的示例列表中调用它：

```python
batch = data_collator([tokenized_datasets["train"][i] for i in range(1, 3)])
print(batch.keys())
```

dict_keys(['attention_mask', 'input_ids', 'labels', 'decoder_input_ids'])

我们可以检查我们的标签是否已填充到批次的最大长度，使用 `-100`：

```python
print(batch["labels"])
```

tensor([[  577,  5891,     2,  3184,    16,  2542,     5,  1710,     0,  -100,
          -100,  -100,  -100,  -100,  -100,  -100],
        [ 1211,     3,    49,  9409,  1211,     3, 29140,   817,  3124,   817,
           550,  7032,  5821,  7907, 12649,     0]])

我们还可以看一下解码器输入 ID，看看它们是标签的移位版本：

```python
print(batch["decoder_input_ids"])
```

tensor([[59513,   577,  5891,     2,  3184,    16,  2542,     5,  1710,     0,
         59513, 59513, 59513, 59513, 59513, 59513],
        [59513,  1211,     3,    49,  9409,  1211,     3, 29140,   817,  3124,
           817,   550,  7032,  5821,  7907, 12649]])

以下是数据集中第一个和第二个元素的标签：

```python
for i in range(1, 3):
    print(tokenized_datasets["train"][i]["labels"])
```

[577, 5891, 2, 3184, 16, 2542, 5, 1710, 0]
[1211, 3, 49, 9409, 1211, 3, 29140, 817, 3124, 817, 550, 7032, 5821, 7907, 12649, 0]

我们会将`data_collator`传递给 `Seq2SeqTrainer`. 接下来，让我们看一下指标。

#### Metrics

`Seq2SeqTrainer` 添加到其超类 `Trainer` 的功能是在评估或预测期间使用 `generate()` 方法的能力。在训练期间，模型将使用带有注意力掩码的 `decoder_input_ids` ，以确保它不会在尝试预测的令牌之后使用标记，以加快训练速度。在推理过程中，我们将无法使用这些标签，因为我们没有标签，因此最好使用相同的设置来评估我们的模型。

正如我们在第 [1 章](https://huggingface.co/course/chapter1/6)中看到的，解码器通过逐个预测令牌来执行推理——这是通过 `generate()` 方法在《变形金刚》中🤗幕后实现的。如果我们设置 `predict_with_generate=True`，将允许我们使用 `Seq2SeqTrainer` 方法进行评估。

用于翻译的传统指标是 [BLEU 分数](https://en.wikipedia.org/wiki/BLEU)，在 [2002](https://aclanthology.org/P02-1040.pdf) 年 Kishore Papineni 等人的一篇文章中引入。BLEU 分数评估翻译与其标签的接近程度。它不衡量模型生成输出的可理解性或语法正确性，而是使用统计规则来确保生成输出中的所有单词也出现在目标中。此外，还有一些规则会惩罚相同单词的重复，如果它们在目标中也没有重复（以避免模型输出句子，如 `"the the the the the"`）和输出比目标中的句子短的句子（以避免模型输出句子，如 `"the"`）。

BLEU的一个弱点是它期望文本已经被标记化，这使得很难比较使用不同标记器的模型之间的分数。因此，目前对翻译模型进行基准测试的最常用指标是 [SacreBLEU](https://github.com/mjpost/sacrebleu)，它通过标准化标记化步骤来解决这个弱点（和其他弱点）。要使用此指标，我们首先需要安装 SacreBLEU 库：

```shell
!pip install sacrebleu
```

然后，我们可以像[在第 3 章](https://huggingface.co/course/chapter3)中所做的那样加载它`evaluate.load()`：

```python
import evaluate

metric = evaluate.load("sacrebleu")
```

此指标将文本作为输入和目标。**它被设计为接受几个可接受的目标，因为同一个句子通常有多个可接受的翻译——我们使用的数据集只提供一个**，但在 NLP 中，找到给出多个句子作为标签的数据集并不少见。因此，预测应该是句子列表，但参考文献应该是句子列表。

让我们尝试一个例子：

```python
predictions = [
    "This plugin lets you translate web pages between several languages automatically."
]
references = [
    [
        "This plugin allows you to automatically translate web pages between several languages."
    ]
]
metric.compute(predictions=predictions, references=references)
```

{'score': 46.750469682990165,
 'counts': [11, 6, 4, 3],
 'totals': [12, 11, 10, 9],
 'precisions': [91.67, 54.54, 40.0, 33.33],
 'bp': 0.9200444146293233,
 'sys_len': 12,
 'ref_len': 13}

这得到了 46.75 的 BLEU 分数，这是相当不错的——作为参考，“[注意力是你所需要的一切”论文](https://arxiv.org/pdf/1706.03762.pdf)中的原始 Transformer 模型在英语和法语之间的类似翻译任务中获得了 41.8 的 BLEU 分数！（有关各个指标的更多信息，如 `counts` 和 `bp` ，请参阅 [SacreBLEU 存储库](https://github.com/mjpost/sacrebleu/blob/078c440168c6adc89ba75fe6d63f0d922d42bcfe/sacrebleu/metrics/bleu.py#L74)。另一方面，如果我们尝试翻译模型中经常出现的两种糟糕的预测类型（大量重复或太短），我们将得到相当糟糕的 BLEU 分数：

```python
predictions = ["This This This This"]
references = [
    [
        "This plugin allows you to automatically translate web pages between several languages."
    ]
]
print(metric.compute(predictions=predictions, references=references))
```

{'score': 1.683602693167689,
 'counts': [1, 0, 0, 0],
 'totals': [4, 3, 2, 1],
 'precisions': [25.0, 16.67, 12.5, 12.5],
 'bp': 0.10539922456186433,
 'sys_len': 4,
 'ref_len': 13}

```python
predictions = ["This plugin"]
references = [
    [
        "This plugin allows you to automatically translate web pages between several languages."
    ]
]
print(metric.compute(predictions=predictions, references=references))
```

{'score': 0.0,
 'counts': [2, 1, 0, 0],
 'totals': [2, 1, 0, 0],
 'precisions': [100.0, 100.0, 0.0, 0.0],
 'bp': 0.004086771438464067,
 'sys_len': 2,
 'ref_len': 13}

分数可以从 0 到 100，越高越好。

为了从模型输出获取指标可以使用的文本，我们将使用`tokenizer.batch_decode()`方法。我们只需要清理标签中的所有 `-100`（分词器会自动对填充令牌执行相同的操作）：

```python
import numpy as np


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # In case the model returns more than the prediction logits
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100s in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    
    return {"bleu": result["score"]}
```

现在已经完成了，我们准备微调我们的模型了！

#### Fine-tuning the model

第一步是登录到 Hugging Face，以便能够将结果上传到模型中心。在笔记本中有一个方便的功能可以帮助您完成此操作：

```python
from huggingface_hub import notebook_login

notebook_login()
```

这将显示一个小部件，您可以在其中输入您的 Hugging Face 登录凭据。

如果您不在笔记本中工作，只需在终端中键入以下行：

```shell
huggingface-cli login
```

完成此操作后，我们可以定义我们的 `Seq2SeqTrainingArguments`. 与 `Trainer` 一样，我们使用包含更多字段的`TrainingArguments` 的子类：

```python
from transformers import Seq2SeqTrainingArguments

args = Seq2SeqTrainingArguments(
    f"marian-finetuned-kde4-en-to-fr",
    evaluation_strategy="no",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=True,
    push_to_hub=True,
)
```

(Comment:  吾人把这个东东上传到hub干啥？直接把push_to_hub设置成false就可以了)

除了通常的超参数（如学习率、纪元数、批量大小和一些权重衰减）之外，与我们在前面的部分中看到的相比，这里有一些变化：

- 我们没有设置任何定期评估，因为评估需要一段时间;我们只会在训练前和训练后评估一次模型。
- 我们设置 `fp16=True` ，这加快了现代 GPU 的训练速度。
- 如上所述，我们设置 `predict_with_generate=True`。
- 我们用 `push_to_hub=True` 在每个纪元结束时将模型上传到 Hub。

请注意，您可以使用 `hub_model_id` 参数指定要推送到的存储库的全名（特别是，您必须使用此参数推送到组织）。例如，当我们将模型推送到 [`huggingface-course` 组织](https://huggingface.co/huggingface-course)时，我们把 `hub_model_id="huggingface-course/marian-finetuned-kde4-en-to-fr"` 添加到`Seq2SeqTrainingArguments` . 默认情况下，使用的存储库将位于您的命名空间中，并以您设置的输出目录命名，因此在我们的例子中，它将是 `"sgugger/marian-finetuned-kde4-en-to-fr"`（这是我们在本节开头链接到的模型）。

💡 如果您使用的输出目录已存在，则它必须是要推送到的存储库的本地克隆。如果不是，则在定义 `Seq2SeqTrainer` 时会出现错误，并且需要设置新名称。

最后，我们只需将所有内容传递给 `Seq2SeqTrainer`：

```python
from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
```

在训练之前，我们将首先查看模型获得的分数，以仔细检查我们的微调是否使事情变得更糟。此命令将花费一些时间，因此您可以在执行时喝杯咖啡：

```python
print(trainer.evaluate(max_length=max_length))
```

{'eval_loss': 1.6964408159255981,
 'eval_bleu': 39.26865061007616,
 'eval_runtime': 965.8884,
 'eval_samples_per_second': 21.76,
 'eval_steps_per_second': 0.341}

39 分的 BLEU 分数还不错，这反映了我们的模型已经擅长将英语句子翻译成法语句子的事实。

接下来是培训，这也需要一些时间：

```python
trainer.train()
```

请注意，在训练发生时，每次保存模型时（此处为每个纪元），它都会在后台上传到 Hub。这样，如有必要，您将能够在另一台机器上恢复训练。

训练完成后，我们再次评估我们的模型——希望我们能看到 BLEU 分数有所改善！

```python
print(trainer.evaluate(max_length=max_length))
```

{'eval_loss': 0.8558505773544312,
 'eval_bleu': 52.94161337775576,
 'eval_runtime': 714.2576,
 'eval_samples_per_second': 29.426,
 'eval_steps_per_second': 0.461,
 'epoch': 3.0}

这是将近 14 个百分点的改进，这很棒。

最后，我们使用 `push_to_hub()` 方法来确保我们上传了模型的最新版本。此外，`Trainer` 还起草了包含所有评估结果的模型卡并上传。此模型卡包含元数据，可帮助模型中心为推理演示选择小组件。通常，不需要说什么，因为它可以从模型类中推断出正确的小部件，但在这种情况下，同一个模型类可以用于各种序列到序列的问题，因此我们将其指定为转换模型：

```python
trainer.push_to_hub(tags="translation", commit_message="Training complete")
```

如果要检查它，此命令将返回它刚刚执行的提交的 URL：

'https://huggingface.co/sgugger/marian-finetuned-kde4-en-to-fr/commit/3601d621e3baae2bc63d3311452535f8f58f6ef3'

在此阶段，您可以使用模型中心上的推理小组件来测试模型并与您的朋友共享。您已成功在翻译任务中微调模型 — 恭喜您！

如果您想更深入地了解训练循环，我们现在将向您展示如何使用 🤗 Accelerate 做同样的事情。

### Q: 除了AutoModelForCausalLM, AutoModelForMaskedLM, AutoModelForSequenceClassification，AutoModelForSeq2SeqLM，还有哪些常见的？

`transformers`库提供了一系列的模型类，以满足不同类型的任务。以下是一些常用的模型类：

1. `AutoModel`: 这是一个通用的类，其中不包含任何特定任务的头部结构。只返回最后层的激活（无论是否汇总）。

2. `AutoModelForPreTraining`:这个可以用于所有预训练任务（就像预训练模型在预训练过程中一样）。

3. `AutoModelForQuestionAnswering`: 这个主要用于问题回答任务。

4. `AutoModelForTokenClassification`: 这个用于分词任务，例如命名实体识别(NER)。

5. `AutoModelForMultipleChoice`: 这个用于多项选择任务。

6. `AutoModelForNextSentencePrediction`: 用于预测下一句话。

7. `AutoModelWithLMHead` (已废弃): 这个类被划分为 `AutoModelForCausalLM`, `AutoModelForMaskedLM`，和 `AutoModelForSeq2SeqLM`。

8. `AutoModelForTranslation`: 用于翻译任务。

9. `AutoModelForTextClassification`: 用于文本分类任务。

10. `AutoModelForStructuredPrediction`: 用于结构化预测任务。

这只是最常见的一些，还有很多其他模型类用于更具体或更有针对性的任务。关于更多信息，可以阅读 Hugging Face 文档的相应部分。

### Q: AutoModelForCausalLM是干啥的？

`AutoModelForCausalLM` 是 Hugging Face Transformers 库中的一个类，用于自动加载适合特定任务的预训练模型。在语言模型任务中，它用于加载适合生成式语言建模任务（如文本生成）的预训练模型。该类根据提供的模型名称自动选择适合的预训练模型，使得用户可以方便地在不同的语言模型上进行实验和应用。

### 7.4.3 A custom training loop

现在让我们看一下完整的训练循环，以便您可以轻松自定义所需的零件。它看起来很像我们在第 [2 节](https://huggingface.co/course/chapter7/2)和[第 3 章](https://huggingface.co/course/chapter3/4)中所做的。

#### 为培训做好一切准备

您现在已经多次看到所有这些内容，因此我们将很快完成代码。首先，我们将用`DataLoader`构建数据集，将数据集设置为`"torch"`格式，以便我们得到 PyTorch 张量：

```python
from torch.utils.data import DataLoader

tokenized_datasets.set_format("torch")
train_dataloader = DataLoader(
    tokenized_datasets["train"],
    shuffle=True,
    collate_fn=data_collator,
    batch_size=8,
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], collate_fn=data_collator, batch_size=8
)
```

接下来，我们重新实例化模型，以确保我们不是从之前开始继续微调，而是再次从预训练模型开始：

```python
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
```

然后我们需要一个优化器：

```python
from transformers import AdamW

optimizer = AdamW(model.parameters(), lr=2e-5)
```

一旦我们拥有所有这些对象，我们就可以将它们发送到方法中。请记住，如果要在 Colab 笔记本中对 TPU 进行训练，则需要将所有这些代码移动到训练函数中，并且该函数不应执行任何实例化 .`accelerator.prepare()``Accelerator`

```python
from accelerate import Accelerator

accelerator = Accelerator()
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)
```

现在我们已经发送了 ，我们可以使用它的长度来计算训练步骤的数量。请记住，我们应该始终在准备数据加载器后执行此操作，因为该方法将更改 .我们使用从学习率到 0 的经典线性时间表：`train_dataloader``accelerator.prepare()``DataLoader`

```python
from transformers import get_scheduler

num_train_epochs = 3
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
```

最后，要将我们的模型推送到中心，我们需要在工作文件夹中创建一个对象。如果您尚未登录，请先登录 Hugging Face Hub。我们将根据我们想要为模型提供的模型 ID 确定存储库名称（请随意将 替换为您自己的选择;它只需要包含您的用户名，这就是函数的作用）：`Repository``repo_name``get_full_repo_name()`

```python
from huggingface_hub import Repository, get_full_repo_name

model_name = "marian-finetuned-kde4-en-to-fr-accelerate"
repo_name = get_full_repo_name(model_name)
repo_name
'sgugger/marian-finetuned-kde4-en-to-fr-accelerate'
```

然后，我们可以将该存储库克隆到本地文件夹中。如果它已经存在，则此本地文件夹应该是我们正在使用的存储库的克隆：

```python
output_dir = "marian-finetuned-kde4-en-to-fr-accelerate"
repo = Repository(output_dir, clone_from=repo_name)
```

现在，我们可以通过调用该方法上传我们保存的任何内容。这将有助于我们在每个纪元结束时上传中间模型。`output_dir``repo.push_to_hub()`

#### 训练循环

现在，我们已准备好编写完整的训练循环。为了简化其评估部分，我们定义了这个函数，它接受预测和标签，并将它们转换为对象期望的字符串列表：`postprocess()``metric`

```python
def postprocess(predictions, labels):
    predictions = predictions.cpu().numpy()
    labels = labels.cpu().numpy()

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]
    return decoded_preds, decoded_labels
```

训练循环看起来很像[第 2 节](https://huggingface.co/course/chapter7/2)和[第 3 章](https://huggingface.co/course/chapter3)中的循环，但在评估部分有一些不同——所以让我们专注于这一点！

首先要注意的是，我们使用该方法来计算预测，但这是基础模型上的方法，而不是在方法中创建的包装模型 🤗 Accelerate。这就是为什么我们先解开模型，然后调用此方法。`generate()``prepare()`

第二件事是，与[令牌分类](https://huggingface.co/course/chapter7/2)一样，两个进程可能已将输入和标签填充为不同的形状，因此我们在调用该方法之前使用使预测和标签具有相同的形状。如果我们不这样做，评估要么出错，要么永远挂起。`accelerator.pad_across_processes()``gather()`

```python
from tqdm.auto import tqdm
import torch

progress_bar = tqdm(range(num_training_steps))

for epoch in range(num_train_epochs):
    # Training
    model.train()
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    # Evaluation
    model.eval()
    for batch in tqdm(eval_dataloader):
        with torch.no_grad():
            generated_tokens = accelerator.unwrap_model(model).generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=128,
            )
        labels = batch["labels"]

        # Necessary to pad predictions and labels for being gathered
        generated_tokens = accelerator.pad_across_processes(
            generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
        )
        labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)

        predictions_gathered = accelerator.gather(generated_tokens)
        labels_gathered = accelerator.gather(labels)

        decoded_preds, decoded_labels = postprocess(predictions_gathered, labels_gathered)
        metric.add_batch(predictions=decoded_preds, references=decoded_labels)

    results = metric.compute()
    print(f"epoch {epoch}, BLEU score: {results['score']:.2f}")

    # Save and upload
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
    if accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)
        repo.push_to_hub(
            commit_message=f"Training in progress epoch {epoch}", blocking=False
        )
epoch 0, BLEU score: 53.47
epoch 1, BLEU score: 54.24
epoch 2, BLEU score: 54.44
```

完成此操作后，您应该有一个模型，其结果与使用 .您可以在 [*huggingface-course/marian-finetuned-kde4-en-to-fr-accelerate*](https://huggingface.co/huggingface-course/marian-finetuned-kde4-en-to-fr-accelerate) 中查看我们使用此代码训练的那个。如果你想测试对训练循环的任何调整，你可以通过编辑上面显示的代码直接实现它们！`Seq2SeqTrainer`

### Q: 下面代码中的-100是干啥的？

在这个代码中，-100被用作一个特殊的标记，代表应当忽略的标签（target）。

在训练序列生成任务，如机器翻译或文本摘要时，模型会接收到额外的带有-100的标签。这是因为模型在对整个序列进行预测，而不仅仅是预测下一个词。在这种情况下，-100的作用就是告诉模型忽略某些特定的词语，因为这些词语在计算损失函数（计算预测值和真实值的差距）时不应被考虑。

也就是说，在这段代码中，-100被用于标注哪些位置是没有真实标签的，或者说那些单词是填充词（padding word）。这些-100标记的位置在计算损失函数和反向传播时都会被忽视。

这种做法在计算损失时非常方便，既可以使序列具有相同的长度（通过填补-100），又可以防止填充的部分对模型训练造成干扰。因为在计算损失函数的时候，会忽略这些标记为-100的位置，使它们不对损失函数的结果产生影响。

### Q: 为什么-100代表padding word? 怎么看出来的？

对于一些深度学习模型在处理变长序列数据的时候，我们常常需要将短的序列通过填充（Padding）的方式扩充到和最长的序列同样长度，以便能够在同一个批次（Batch）内同时处理。然而，填充的部分实际上并无实质性的信息，因此在后续的计算中，我们希望模型能够忽略这些填充的部分。

在 Hugging Face 的 Transformer 模型中，填充位置的标签被设置为 `-100`。当我们使用交叉熵损失函数计算预测结果和实际目标之间的差异时，损失函数会自动忽略所有的 `-100` 标签，也就是说，这部分位置的预测结果不会反馈到模型的训练中去。

我们可以从损失函数计算的相关代码中看出这一点。比如，在 PyTorch 的 `nn.CrossEntropyLoss` 中，有一个 `ignore_index` 参数默认设置为 `-100`。这个参数就是用来指示哪些标签应该被忽略在损失计算外的。

这是一个约定俗成的设定，使用 `-100` 作为填充单词的目标标签值，并不意味着 `-100` 有特别的含义，只是被用作一个特殊标记，在计算交叉熵损失时会忽略它。

### Q: s/it是什么单位？

's/it' 表示 'seconds per iteration'，即每轮迭代所需的时间（秒）。模型训练是通过迭代过程进行的，每一次迭代（在这种情况下，对应一批训练样本的处理）模型都会根据误差的反向传播稍微调整其参数。

因此，'1.08s/it' 表示模型在处理每个批次（iteration）所需的平均时间为 1.08 秒。这是评估计算效率和预估总训练时间的一个重要指标。

### 7.4.4 Using the fine-tuned model

我们已经向您展示了如何将我们在模型中心上微调的模型与推理小组件一起使用。要在 `pipeline` 中本地使用它 (Comment: 原来微调后的模型可以通过pipeline在本地使用)，我们只需要指定正确的模型标识符：

```python
from transformers import pipeline

# Replace this with your own checkpoint
model_checkpoint = "huggingface-course/marian-finetuned-kde4-en-to-fr"
translator = pipeline("translation", model=model_checkpoint)
print(translator("Default to expanded threads"))
```

[{'translation_text': 'Par défaut, développer les fils de discussion'}]

正如预期的那样，我们的预训练模型将其知识调整为我们微调的语料库，并且现在不再单独使用英语单词“threads”，而是将其翻译成法语官方版本。“插件”也是如此：

```python
print(translator(
    "Unable to import %1 using the OFX importer plugin. This file is not the correct format."
))
```

[{'translation_text': "Impossible d'importer %1 en utilisant le module externe d'importation OFX. Ce fichier n'est pas le bon format."}]

域适配的另一个很好的例子！

✏️ **该你了！**模型在带有您之前确定的单词“电子邮件”的样本上返回什么？

## 7.5 Summarization

在本节中，我们将了解如何使用 Transformer 模型将长文档压缩为摘要，这项任务称为*文本摘要*。这是最具挑战性的 NLP 任务之一，因为它需要一系列能力，例如理解长篇文章和生成连贯的文本来捕捉文档中的主要主题。但是，如果做得好，文本摘要是一个强大的工具，可以通过减轻领域专家详细阅读长文档的负担来加快各种业务流程。

虽然在[Hugging Face Hub](https://huggingface.co/models?pipeline_tag=summarization&sort=downloads)上已经存在各种用于总结的微调模型，但几乎所有这些模型都只适用于英文文档。因此，为了在本节中增加一个转折点，我们将训练一个英语和西班牙语的双语模型。在本节结束时，你将拥有一个模型，该[模型](https://huggingface.co/huggingface-course/mt5-small-finetuned-amazon-en-es)可以总结客户评论，如下所示：

正如我们将看到的，这些摘要是简洁的，因为它们是从客户在产品评论中提供的标题中学到的。让我们首先为这项任务建立一个合适的双语语料库。

### 7.5.1 Preparing a multilingual corpus

我们将使用[多语言亚马逊评论语料库](https://huggingface.co/datasets/amazon_reviews_multi)来创建我们的双语摘要器。该语料库包含六种语言的亚马逊商品评论，通常用于对多语言分类器进行基准测试。但是，由于每篇评论都附有一个简短的标题，因此我们可以将这些标题用作模型学习的目标摘要！首先，让我们从 Hugging Face Hub 下载英语和西班牙语子集：

```python
from datasets import load_dataset

spanish_dataset = load_dataset("amazon_reviews_multi", "es")
english_dataset = load_dataset("amazon_reviews_multi", "en")
print(english_dataset)
```

DatasetDict({
    train: Dataset({
        features: ['review_id', 'product_id', 'reviewer_id', 'stars', 'review_body', 'review_title', 'language', 'product_category'],
        num_rows: 200000
    })
    validation: Dataset({
        features: ['review_id', 'product_id', 'reviewer_id', 'stars', 'review_body', 'review_title', 'language', 'product_category'],
        num_rows: 5000
    })
    test: Dataset({
        features: ['review_id', 'product_id', 'reviewer_id', 'stars', 'review_body', 'review_title', 'language', 'product_category'],
        num_rows: 5000
    })
})

如您所见，对于每种语言，拆分有 200,000 条评论，每个拆分有 5,000 条评论。我们感兴趣的评论信息包含在 和 列中。让我们看几个例子，通过创建一个简单的函数，该函数使用我们[在第 5 章](https://huggingface.co/course/chapter5)中学到的技术从训练集中随机抽取样本：`train``validation``test``review_body``review_title`

```python
def show_samples(dataset, num_samples=3, seed=42):
    sample = dataset["train"].shuffle(seed=seed).select(range(num_samples))
    for example in sample:
        print(f"\n'>> Title: {example['review_title']}'")
        print(f"'>> Review: {example['review_body']}'")


show_samples(english_dataset)
```

'>> Title: Worked in front position, not rear'
'>> Review: 3 stars because these are not rear brakes as stated in the item description. At least the mount adapter only worked on the front fork of the bike that I got it for.'

'>> Title: meh'
'>> Review: Does it’s job and it’s gorgeous but mine is falling apart, I had to basically put it together again with hot glue'

'>> Title: Can\'t beat these for the money'
'>> Review: Bought this for handling miscellaneous aircraft parts and hanger "stuff" that I needed to organize; it really fit the bill. The unit arrived quickly, was well packaged and arrived intact (always a good sign). There are five wall mounts-- three on the top and two on the bottom. I wanted to mount it on the wall, so all I had to do was to remove the top two layers of plastic drawers, as well as the bottom corner drawers, place it when I wanted and mark it; I then used some of the new plastic screw in wall anchors (the 50 pound variety) and it easily mounted to the wall. Some have remarked that they wanted dividers for the drawers, and that they made those. Good idea. My application was that I needed something that I can see the contents at about eye level, so I wanted the fuller-sized drawers. I also like that these are the new plastic that doesn\'t get brittle and split like my older plastic drawers did. I like the all-plastic construction. It\'s heavy duty enough to hold metal parts, but being made of plastic it\'s not as heavy as a metal frame, so you can easily mount it to the wall and still load it up with heavy stuff, or light stuff. No problem there. For the money, you can\'t beat it. Best one of these I\'ve bought to date-- and I\'ve been using some version of these for over forty years.'

✏️ **试试看！**更改命令中的随机种子以浏览语料库中的其他评论。如果您是讲西班牙语的人，请查看其中的一些评论，看看标题是否也是合理的摘要。`Dataset.shuffle()``spanish_dataset`

这个样本显示了人们通常在网上找到的评论的多样性，从正面到负面（以及介于两者之间的一切！虽然带有“meh”标题的示例信息量不大，但其他标题看起来像是对评论本身的体面总结。在所有 400,000 条评论上训练摘要模型在单个 GPU 上花费的时间太长，因此我们将专注于为单个产品领域生成摘要。为了了解我们可以选择哪些域，让我们转换为 a 并计算每个产品类别的评论数量：`english_dataset``pandas.DataFrame`

```
english_dataset.set_format("pandas")
english_df = english_dataset["train"][:]
# Show counts for top 20 products
english_df["product_category"].value_counts()[:20]
home                      17679
apparel                   15951
wireless                  15717
other                     13418
beauty                    12091
drugstore                 11730
kitchen                   10382
toy                        8745
sports                     8277
automotive                 7506
lawn_and_garden            7327
home_improvement           7136
pet_products               7082
digital_ebook_purchase     6749
pc                         6401
electronics                6186
office_product             5521
shoes                      5197
grocery                    4730
book                       3756
Name: product_category, dtype: int64
```

英语数据集中最受欢迎的产品是关于家居用品、服装和无线电子产品的。不过，为了坚持亚马逊的主题，让我们专注于总结书评——毕竟，这就是公司成立的基础！我们可以看到两个符合要求的产品类别（ 和 ），因此让我们仅针对这些产品过滤两种语言的数据集。正如我们在第 [5 章](https://huggingface.co/course/chapter5)中看到的，该函数允许我们非常有效地对数据集进行切片，因此我们可以定义一个简单的函数来执行此操作：`book``digital_ebook_purchase``Dataset.filter()`

```python
def filter_books(example):
    return (
        example["product_category"] == "book"
        or example["product_category"] == "digital_ebook_purchase"
    )
```

现在，当我们将此函数应用于 和 时，结果将只包含涉及书籍类别的那些行。在应用过滤器之前，让我们将 的格式从 切换回 ：`english_dataset``spanish_dataset``english_dataset``"pandas"``"arrow"`

```
english_dataset.reset_format()
```

然后，我们可以应用过滤功能，作为健全性检查，让我们检查一个评论样本，看看它们是否确实是关于书籍的：

```python
spanish_books = spanish_dataset.filter(filter_books)
english_books = english_dataset.filter(filter_books)
show_samples(english_books)
```

'>> Title: I\'m dissapointed.'
'>> Review: I guess I had higher expectations for this book from the reviews. I really thought I\'d at least like it. The plot idea was great. I loved Ash but, it just didnt go anywhere. Most of the book was about their radio show and talking to callers. I wanted the author to dig deeper so we could really get to know the characters. All we know about Grace is that she is attractive looking, Latino and is kind of a brat. I\'m dissapointed.'

'>> Title: Good art, good price, poor design'
'>> Review: I had gotten the DC Vintage calendar the past two years, but it was on backorder forever this year and I saw they had shrunk the dimensions for no good reason. This one has good art choices but the design has the fold going through the picture, so it\'s less aesthetically pleasing, especially if you want to keep a picture to hang. For the price, a good calendar'

'>> Title: Helpful'
'>> Review: Nearly all the tips useful and. I consider myself an intermediate to advanced user of OneNote. I would highly recommend.'

好的，我们可以看到这些评论并不是严格意义上的书籍，可能指的是日历和电子应用程序（如 OneNote）之类的东西。尽管如此，该领域似乎更适合训练摘要模型。在我们研究适合此任务的各种模型之前，我们还有最后一点数据准备工作要做：将英语和西班牙语评论合并为一个对象。🤗 数据集提供了一个方便的函数，顾名思义，它将两个对象堆叠在一起。因此，为了创建我们的双语数据集，我们将遍历每个拆分，连接该拆分的数据集，并对结果进行打牌，以确保我们的模型不会过度拟合到单一语言：`DatasetDict``concatenate_datasets()``Dataset`

```
from datasets import concatenate_datasets, DatasetDict

books_dataset = DatasetDict()

for split in english_books.keys():
    books_dataset[split] = concatenate_datasets(
        [english_books[split], spanish_books[split]]
    )
    books_dataset[split] = books_dataset[split].shuffle(seed=42)

# Peek at a few examples
show_samples(books_dataset)
'>> Title: Easy to follow!!!!'
'>> Review: I loved The dash diet weight loss Solution. Never hungry. I would recommend this diet. Also the menus are well rounded. Try it. Has lots of the information need thanks.'

'>> Title: PARCIALMENTE DAÑADO'
'>> Review: Me llegó el día que tocaba, junto a otros libros que pedí, pero la caja llegó en mal estado lo cual dañó las esquinas de los libros porque venían sin protección (forro).'

'>> Title: no lo he podido descargar'
'>> Review: igual que el anterior'
```

这当然看起来像是英语和西班牙语评论的混合体！现在我们有了一个训练语料库，最后要检查的是评论中的单词分布及其标题。这对于摘要任务尤为重要，因为数据中的简短参考摘要可能会使模型偏向于在生成的摘要中仅输出一个或两个单词。下图显示了单词分布，我们可以看到标题严重偏向于只有 1-2 个单词：

![Word count distributions for the review titles and texts.](https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter7/review-lengths.svg)

为了解决这个问题，我们将过滤掉标题非常短的示例，以便我们的模型可以生成更有趣的摘要。由于我们正在处理英语和西班牙语文本，因此我们可以使用粗略的启发式方法将标题拆分为空格，然后使用我们可靠的方法，如下所示：`Dataset.filter()`

```
books_dataset = books_dataset.filter(lambda x: len(x["review_title"].split()) > 2)
```

现在我们已经准备好了语料库，让我们来看看一些可能的 Transformer 模型，人们可以对其进行微调！

### 7.5.2 Models for text summarization

如果你仔细想想，**文本摘要是一种类似于机器翻译的任务：我们有一个像评论一样的文本正文，我们想把它“翻译”成一个较短的版本，以捕捉输入的显着特征** （Comment:  没错，吾人也是这么想的。而且吾人认为PCF也类似于这样的任务，因此说不定可以采用类似的架构）。因此，大多数用于汇总的 Transformer 模型都采用了我们[在第 1 章](https://huggingface.co/course/chapter1)中首次遇到的编码器-解码器架构，尽管也有一些例外，例如 GPT 系列模型，它们也可用于在小样本设置中进行汇总。下表列出了一些常用的预训练模型，可以对这些模型进行微调以进行汇总。

| 变压器型号                                                   | 描述                                                         | 多种语言？ |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ---------- |
| [GPT-2的](https://huggingface.co/gpt2-xl)                    | 虽然被训练为自动回归语言模型，但您可以通过附加“TL;DR“在输入文本的末尾。 | ❌          |
| [飞马座](https://huggingface.co/google/pegasus-large)        | 使用预训练目标来预测多句子文本中的屏蔽句子。这个预训练目标比普通的语言建模更接近于总结，并且在流行的基准测试中得分很高。 | ❌          |
| [T5系列](https://huggingface.co/t5-base)                     | 通用的 Transformer 架构，可在文本转文本框架中制定所有任务;例如，用于总结文档的模型的输入格式为 。`summarize: ARTICLE` | ❌          |
| [mT5型](https://huggingface.co/google/mt5-base)              | T5 的多语言版本，在多语言 Common Crawl 语料库 （mC4） 上进行预训练，涵盖 101 种语言。 | ✅          |
| [巴特](https://huggingface.co/facebook/bart-base)            | 一种新颖的 Transformer 架构，具有编码器和解码器堆栈，经过训练以重建损坏的输入，结合了 BERT 和 GPT-2 的预训练方案。 | ❌          |
| [mBART-50型](https://huggingface.co/facebook/mbart-large-50) | BART 的多语言版本，预训练了 50 种语言。                      | ✅          |

从这张表中可以看出，大多数用于总结的 Transformer 模型（实际上大多数 NLP 任务）都是单语的。如果您的任务是使用英语或德语等“高资源”语言，那就太好了，但对于世界各地使用的数千种其他语言来说就不那么重要了。幸运的是，有一类多语言的 Transformer 模型，如 mT5 和 mBART，可以派上用场。这些模型是使用语言建模进行预训练的，但有一个转折点：它们不是在一种语言的语料库上进行训练，而是同时在 50 多种语言的文本上进行联合训练！

我们将重点介绍 mT5，这是一个基于 T5 的有趣架构，在文本转文本框架中进行了预训练。在 T5 中，每个 NLP 任务都是根据提示前缀来表述的，例如，该前缀使模型使生成的文本适应提示。如下图所示，这使得 T5 非常通用，因为您可以使用单个模型解决许多任务！`summarize:`

![Different tasks performed by the T5 architecture.](https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter7/t5.svg)

mT5 不使用前缀，但与 T5 一样具有多功能性，并具有多语言的优势。现在我们已经选择了一个模型，让我们看一下如何准备训练数据。

✏️ **试试看！**完成本节后，通过使用相同的技术微调 mT5 与 mBART 相比如何。为了获得奖励积分，您还可以尝试仅对英文评论进行微调 T5。由于 T5 具有特殊的前缀提示，因此您需要在下面的预处理步骤中将 prepend 到输入示例之前。`summarize:`

### 7.5.3 Preprocessing the data

我们的下一个任务是对我们的评论及其标题进行标记和编码。像往常一样，我们首先加载与预训练模型检查点关联的分词器。我们将用作检查点，以便我们可以在合理的时间内对模型进行微调：`mt5-small`

```python
from transformers import AutoTokenizer

model_checkpoint = "google/mt5-small"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
```

💡 在 NLP 项目的早期阶段，一个好的做法是在一小块数据样本上训练一类“小”模型。这使您可以更快地调试和迭代端到端工作流。一旦您对结果充满信心，您始终可以通过简单地更改模型检查点来扩展模型！

让我们在一个小例子中测试 mT5 分词器：

```python
inputs = tokenizer("I loved reading the Hunger Games!")
print(inputs)
```

{'input_ids': [336, 259, 28387, 11807, 287, 62893, 295, 12507, 1], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1]}

在这里，我们可以看到熟悉的，以及我们在第 [3 章](https://huggingface.co/course/chapter3)的第一次微调实验中遇到的。让我们用分词器的函数解码这些输入 ID，看看我们正在处理哪种分词器：`input_ids``attention_mask``convert_ids_to_tokens()`

```python
tokenizer.convert_ids_to_tokens(inputs.input_ids)
```

['▁I', '▁', 'loved', '▁reading', '▁the', '▁Hung', 'er', '▁Games', '</s>']

特殊的 Unicode 字符和序列末尾标记表明我们正在处理 SentencePiece 标记器，它基于[第 6 章](https://huggingface.co/course/chapter6)中讨论的 Unigram 分割算法。Unigram 对于多语言语料库特别有用，因为它允许 SentencePiece 与口音、标点符号以及许多语言（如日语）没有空格字符这一事实无关。`▁``</s>`

为了标记我们的语料库，我们必须处理与摘要相关的微妙之处：因为我们的标签也是文本，所以它们可能会超过模型的最大上下文大小。这意味着我们需要对评论及其标题应用截断，以确保我们不会将过长的输入传递到我们的模型中。Transformers 中的🤗分词器提供了一个漂亮的参数，允许您将标签与输入并行标记化。以下是如何处理 mT5 的输入和目标的示例：`text_target`

```python
max_input_length = 512
max_target_length = 30


def preprocess_function(examples):
    model_inputs = tokenizer(
        examples["review_body"],
        max_length=max_input_length,
        truncation=True,
    )
    labels = tokenizer(
        examples["review_title"], max_length=max_target_length, truncation=True
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
```

让我们通过此代码来了解正在发生的事情。我们做的第一件事是定义 和 的值，它设置了我们的评论和标题可以有多长的上限。由于评价正文通常比标题大得多，因此我们相应地调整了这些值。`max_input_length``max_target_length`

有了 ，那么使用我们在本课程中广泛使用的便捷函数对整个语料库进行标记就很简单了：`preprocess_function()``Dataset.map()`

```python
tokenized_datasets = books_dataset.map(preprocess_function, batched=True)
```

现在语料库已经过预处理，让我们来看看一些通常用于汇总的指标。正如我们将看到的，在衡量机器生成文本的质量方面，没有灵丹妙药。

💡 您可能已经注意到，我们在上面的函数中使用了。这将以 1,000 个（默认值）为批次对示例进行编码，并允许您利用 Transformer 中🤗快速分词器的多线程功能。在可能的情况下，尝试使用以充分利用您的预处理！`batched=True``Dataset.map()``batched=True`

### 7.5.4 Metrics for text summarization

与我们在本课程中介绍的大多数其他任务相比，衡量摘要或翻译等文本生成任务的性能并不那么简单。例如，给定像“我喜欢阅读《饥饿游戏》”这样的评论，有多个有效的摘要，例如“我喜欢《饥饿游戏》”或“《饥饿游戏》是一本很棒的书”。显然，在生成的摘要和标签之间应用某种精确匹配并不是一个好的解决方案——即使是人类在这样的指标下也会表现不佳，因为我们都有自己的写作风格。

总结一下，最常用的指标之一是 [ROUGE 分数](https://en.wikipedia.org/wiki/ROUGE_(metric))（Recall-Oriented Understudy for Gisting Evaluation 的缩写）。该指标背后的基本思想是**将生成的摘要与通常由人类创建的一组参考摘要进行比较**。为了更准确地说，假设我们想比较以下两个摘要：

```python
generated_summary = "I absolutely loved reading the Hunger Games"
reference_summary = "I loved reading the Hunger Games"
```

比较它们的一种方法是**计算重叠单词的数量**，在这种情况下为 6。然而，这有点粗糙，所以ROUGE是基于计算重叠的*精确度*和*召回*率分数。（Comment:  这个ROUGE分数指标吾人在PCF中也可以用。）

🙋 如果这是您第一次听说精确度和召回率，请不要担心——我们将一起通过一些明确的例子来说明一切。这些指标通常在分类任务中遇到，因此，如果您想了解如何在该上下文中定义精确率和召回率，我们建议您查看[指南](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html)。`scikit-learn`

对于 ROUGE，召回率衡量生成的参考摘要捕获了多少参考摘要。如果我们只是比较单词，召回率可以根据以下公式计算：Rec一个ll=Number of overl一个pp我ng wordsTot一个l number of words 我n reference summ一个ry召回=参考文献摘要中的总字数重叠字数

对于我们上面的简单示例，这个公式给出了 6/6 = 1 的完美召回;即，参考摘要中的所有单词均由模型生成。这听起来可能很棒，但想象一下，如果我们生成的摘要是“我真的很喜欢整夜阅读饥饿游戏”。这也可以完美地回忆起来，但可以说是一个更糟糕的总结，因为它很冗长。为了处理这些情况，我们还计算了精度，在ROUGE上下文中，该精度衡量生成的摘要中有多少是相关的：Prec我s我on=Number of overl一个pp我ng wordsTot一个l number of words 我n gener一个ted summ一个ry精度=生成的摘要中的总字数重叠字数

将其应用于我们的详细摘要，得出的精度为 6/10 = 0.6，这比我们较短的摘要获得的 6/7 = 0.86 的精度要差得多。在实践中，通常同时计算精确率和召回率，然后报告 F1 分数（精确率和召回率的调和平均值）。我们可以通过首先安装软件包在 Datasets 中🤗轻松完成此操作：`rouge_score`

```shell
!pip install rouge_score
```

然后加载 ROUGE 指标，如下所示：

```python
import evaluate

rouge_score = evaluate.load("rouge")
```

然后，我们可以使用该函数一次计算所有指标：`rouge_score.compute()`

```python
scores = rouge_score.compute(
    predictions=[generated_summary], references=[reference_summary]
)
print(scores)
```

{'rouge1': AggregateScore(low=Score(precision=0.86, recall=1.0, fmeasure=0.92), mid=Score(precision=0.86, recall=1.0, fmeasure=0.92), high=Score(precision=0.86, recall=1.0, fmeasure=0.92)),
 'rouge2': AggregateScore(low=Score(precision=0.67, recall=0.8, fmeasure=0.73), mid=Score(precision=0.67, recall=0.8, fmeasure=0.73), high=Score(precision=0.67, recall=0.8, fmeasure=0.73)),
 'rougeL': AggregateScore(low=Score(precision=0.86, recall=1.0, fmeasure=0.92), mid=Score(precision=0.86, recall=1.0, fmeasure=0.92), high=Score(precision=0.86, recall=1.0, fmeasure=0.92)),
 'rougeLsum': AggregateScore(low=Score(precision=0.86, recall=1.0, fmeasure=0.92), mid=Score(precision=0.86, recall=1.0, fmeasure=0.92), high=Score(precision=0.86, recall=1.0, fmeasure=0.92))}

哇，输出中有很多信息——这是什么意思？首先，🤗数据集实际上计算精确率、召回率和 F1 分数的置信区间;这些是您可以在此处看到的 、 和 属性。此外，在比较生成的摘要和参考摘要时，🤗数据集会根据不同类型的文本粒度计算各种 ROUGE 分数。变体是单字母的重叠——这只是一种说单词重叠的奇特方式，正是我们上面讨论的指标。为了验证这一点，让我们提取分数的值：`low``mid``high``rouge1``mid`

```python
print(scores["rouge1"].mid)
```

Score(precision=0.86, recall=1.0, fmeasure=0.92)

太好了，精确度和召回率数字匹配！那么其他的ROUGE分数呢？ 测量二元组之间的重叠（想想单词对的重叠），同时通过在生成的和参考摘要中查找最长的公共子字符串来测量最长的匹配单词序列。中的“总和”是指该指标是在整个摘要上计算的，而计算为单个句子的平均值。`rouge2``rougeL``rougeLsum``rougeLsum``rougeL`

✏️ **试试看！**创建您自己的生成和参考摘要示例，并查看生成的 ROUGE 分数是否与基于精确率和召回率公式的手动计算一致。为了获得奖励，将文本拆分为二元组，并比较指标的精确度和召回率。`rouge2`

我们将使用这些 ROUGE 分数来跟踪我们模型的性能，但在这样做之前，让我们做一些每个优秀的 NLP 从业者都应该做的事情：创建一个强大而简单的基线！

#### Creating a strong baseline

文本摘要的常见基线是简单地采用文章的前三句话，通常称为 *lead-3* 基线。我们可以使用句号来跟踪句子边界，但这在“U.S.”或“U.N.”等首字母缩略词上会失败——所以我们将使用该库，其中包括一个更好的算法来处理这些情况。您可以使用以下方法安装软件包：`nltk``pip`

```shell
!pip install nltk
```

然后下载标点符号规则：

```python
import nltk

nltk.download("punkt")
```

接下来，我们从中导入句子分词器，并创建一个简单的函数来提取评论中的前三个句子。文本摘要的惯例是用换行符分隔每个摘要，因此让我们也包含此内容并在训练示例中对其进行测试：`nltk`

```python
from nltk.tokenize import sent_tokenize


def three_sentence_summary(text):
    return "\n".join(sent_tokenize(text)[:3])


print(three_sentence_summary(books_dataset["train"][1]["review_body"]))
```

'I grew up reading Koontz, and years ago, I stopped,convinced i had "outgrown" him.'
'Still,when a friend was looking for something suspenseful too read, I suggested Koontz.'
'She found Strangers.'

这似乎有效，所以现在让我们实现一个函数，从数据集中提取这些“摘要”并计算基线的 ROUGE 分数：

```python
def evaluate_baseline(dataset, metric):
    summaries = [three_sentence_summary(text) for text in dataset["review_body"]]
    return metric.compute(predictions=summaries, references=dataset["review_title"])
```

然后，我们可以使用这个函数来计算验证集上的ROUGE分数，并使用Pandas对它们进行一些美化：

```
import pandas as pd

score = evaluate_baseline(books_dataset["validation"], rouge_score)
rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
rouge_dict = dict((rn, round(score[rn].mid.fmeasure * 100, 2)) for rn in rouge_names)
rouge_dict
{'rouge1': 16.74, 'rouge2': 8.83, 'rougeL': 15.6, 'rougeLsum': 15.96}
```

我们可以看到分数明显低于其他分数;这可能反映了一个事实，即综述标题通常很简洁，因此 lead-3 基线过于冗长。现在我们有了一个很好的基线，让我们把注意力转向微调 mT5！`rouge2`

### 7.5.5 Fine-tuning mT5 with the Trainer API

微调汇总模型与我们在本章中介绍的其他任务非常相似。我们需要做的第一件事是从检查点加载预训练的模型。由于汇总是一个序列到序列的任务，我们可以用类加载模型，该类将自动下载并缓存权重：`mt5-small``AutoModelForSeq2SeqLM`

```python
from transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
```

💡 如果您想知道为什么在下游任务上看不到任何关于微调模型的警告，那是因为对于序列到序列任务，我们保留了网络的所有权重。将其与[第 3 章](https://huggingface.co/course/chapter3)中的文本分类模型进行比较，其中预训练模型的头部被随机初始化的网络替换。

接下来我们需要做的是登录 Hugging Face Hub。如果在笔记本中运行此代码，可以使用以下实用工具函数执行此操作：

```python
from huggingface_hub import notebook_login

notebook_login()
```

这将显示一个小部件，您可以在其中输入凭据。或者，您可以在终端中运行此命令并登录：

```shell
huggingface-cli login
```

我们需要生成摘要，以便在训练期间计算 ROUGE 分数。幸运的是，🤗变形金刚提供了可以自动为我们做到这一点的专用类和类！为了了解其工作原理，让我们首先为实验定义超参数和其他参数：`Seq2SeqTrainingArguments``Seq2SeqTrainer`

```python
from transformers import Seq2SeqTrainingArguments

batch_size = 8
num_train_epochs = 8
# Show the training loss with every epoch
logging_steps = len(tokenized_datasets["train"]) // batch_size
model_name = model_checkpoint.split("/")[-1]

args = Seq2SeqTrainingArguments(
    output_dir=f"{model_name}-finetuned-amazon-en-es",
    evaluation_strategy="epoch",
    learning_rate=5.6e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=num_train_epochs,
    predict_with_generate=True,
    logging_steps=logging_steps,
    push_to_hub=True,
)
```

在这里，参数被设置为指示我们应该在评估期间生成摘要，以便我们可以计算每个时期的 ROUGE 分数。如[第 1 章](https://huggingface.co/course/chapter1)所述，解码器通过逐个预测令牌来执行推理，这是通过模型的方法实现的。设置告诉使用该方法进行评估。我们还调整了一些默认的超参数，如学习率、epoch 数和权重衰减，并且我们设置了在训练期间最多只保存 3 个检查点的选项——这是因为即使是 mT5 的“小”版本也会使用大约 1 GB 的硬盘空间，我们可以通过限制保存的副本数量来节省一些空间。`predict_with_generate``generate()``predict_with_generate=True``Seq2SeqTrainer``save_total_limit`

该参数将允许我们在训练后将模型推送到 Hub;您可以在用户配置文件下找到该存储库，该存储库位于 定义的位置。请注意，您可以使用参数指定要推送到的存储库的名称（特别是，您必须使用此参数推送到组织）。例如，当我们将模型推送到 [`huggingface-course` 组织](https://huggingface.co/huggingface-course)时，我们添加了 .`push_to_hub=True``output_dir``hub_model_id``hub_model_id="huggingface-course/mt5-finetuned-amazon-en-es"``Seq2SeqTrainingArguments`

接下来我们需要做的是为训练器提供一个函数，以便我们可以在训练期间评估我们的模型。对于摘要来说，这比简单地调用模型的预测要复杂一些，因为我们需要先将输出和标签*解码*为文本，然后才能计算 ROUGE 分数。以下函数正是这样做的，并且还利用函数 from 用换行符分隔摘要句子：`compute_metrics()``rouge_score.compute()``sent_tokenize()``nltk`

```python
import numpy as np


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Decode generated summaries into text
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    # Decode reference summaries into text
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # ROUGE expects a newline after each sentence
    decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(sent_tokenize(label.strip())) for label in decoded_labels]
    # Compute ROUGE scores
    result = rouge_score.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    # Extract the median scores
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    return {k: round(v, 4) for k, v in result.items()}
```

接下来，我们需要为我们的序列到序列任务定义一个数据整理器。由于 mT5 是编码器-解码器转换器模型，因此准备批次的一个微妙之处在于，在解码过程中，我们需要将标签向右移动一个。这是为了确保解码器只看到以前的地面实况标签，而看不到当前或未来的地面实况标签，这对模型来说很容易记住。这类似于在[因果语言建模](https://huggingface.co/course/chapter7/6)等任务中将蒙面的自我注意力应用于输入的方式。

幸运的是，🤗 Transformers 提供了一个整理器，可以动态地为我们填充输入和标签。要实例化此整理器，我们只需要提供 和 ：`DataCollatorForSeq2Seq``tokenizer``model`

```python
from transformers import DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
```

让我们看看这个整理器在输入一小批示例时会产生什么。首先，我们需要删除带有字符串的列，因为整理器不知道如何填充这些元素：

```python
tokenized_datasets = tokenized_datasets.remove_columns(
    books_dataset["train"].column_names
)
```

由于整理器需要一个 s 列表，其中每个 s 代表数据集中的单个示例，因此我们还需要在将数据传递给数据整理器之前将数据整理成预期的格式：`dict``dict`

```python
features = [tokenized_datasets["train"][i] for i in range(2)]
data_collator(features)
```

{'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]), 'input_ids': tensor([[  1494,    259,   8622,    390,    259,    262,   2316,   3435,    955,
            772,    281,    772,   1617,    263,    305,  14701,    260,   1385,
           3031,    259,  24146,    332,   1037,    259,  43906,    305,    336,
            260,      1,      0,      0,      0,      0,      0,      0],
        [   259,  27531,  13483,    259,   7505,    260, 112240,  15192,    305,
          53198,    276,    259,  74060,    263,    260,    459,  25640,    776,
           2119,    336,    259,   2220,    259,  18896,    288,   4906,    288,
           1037,   3931,    260,   7083, 101476,   1143,    260,      1]]), 'labels': tensor([[ 7483,   259,  2364, 15695,     1,  -100],
        [  259, 27531, 13483,   259,  7505,     1]]), 'decoder_input_ids': tensor([[    0,  7483,   259,  2364, 15695,     1],
        [    0,   259, 27531, 13483,   259,  7505]])}

这里要注意的主要事情是，第一个示例比第二个示例长，因此第二个示例的 and 在右侧填充了一个标记（其 ID 是 ）。同样，我们可以看到 已经用 s 填充了，以确保填充标记被 loss 函数忽略。最后，我们可以看到一个新标签，它通过在第一个条目中插入一个标记来将标签向右移动。`input_ids``attention_mask``[PAD]``0``labels``-100``decoder_input_ids``[PAD]`

我们终于拥有了训练所需的所有成分！现在，我们只需要使用标准参数实例化训练器：

```python
from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
```

并启动我们的培训运行：

```python
trainer.train()
```

在训练期间，您应该会看到训练损失减少，ROUGE 分数随着每个时期的增加而增加。训练完成后，您可以通过运行以下命令查看最终的 ROUGE 分数：`Trainer.evaluate()`

```python
trainer.evaluate()
```

{'eval_loss': 3.028524398803711,
 'eval_rouge1': 16.9728,
 'eval_rouge2': 8.2969,
 'eval_rougeL': 16.8366,
 'eval_rougeLsum': 16.851,
 'eval_gen_len': 10.1597,
 'eval_runtime': 6.1054,
 'eval_samples_per_second': 38.982,
 'eval_steps_per_second': 4.914}

从分数中，我们可以看到我们的模型轻松超越了我们的 lead-3 基线——很好！最后要做的是将模型权重推送到中心，如下所示：

```
trainer.push_to_hub(commit_message="Training complete", tags="summarization")
'https://huggingface.co/huggingface-course/mt5-finetuned-amazon-en-es/commit/aa0536b829b28e73e1e4b94b8a5aacec420d40e0'
```

这会将检查点和配置文件保存到 ，然后再将所有文件上传到 Hub。通过指定参数，我们还确保 Hub 上的小组件将用于摘要管道，而不是与 mT5 架构关联的默认文本生成（有关模型标记的详细信息，请参阅 [🤗 Hub 文档](https://huggingface.co/docs/hub/main#how-is-a-models-type-of-inference-api-and-widget-determined)）。输出是 Git 提交哈希的 URL，因此您可以轻松查看对模型存储库所做的更改！`output_dir``tags``trainer.push_to_hub()`

在结束本节之前，让我们来看看如何使用 Accelerate 提供的🤗低级功能来微调 mT5。

### 7.5.6 Fine-tuning mT5 with 🤗 Accelerate

使用 🤗 Accelerate 微调我们的模型与我们[在第 3 章](https://huggingface.co/course/chapter3)中遇到的文本分类示例非常相似。主要区别在于需要在训练期间明确生成我们的摘要，并定义我们如何计算 ROUGE 分数（回想一下，我们负责生成）。让我们来看看如何在 Accelerate 中🤗实现这两个要求！`Seq2SeqTrainer`

#### Preparing everything for training

我们需要做的第一件事是为每个拆分创建一个。由于 PyTorch 数据加载器需要批量张量，因此我们需要在数据集中将格式设置为：`DataLoader``"torch"`

```python
tokenized_datasets.set_format("torch")
```

现在我们已经有了仅由张量组成的数据集，接下来要做的就是再次实例化。为此，我们需要提供模型的新版本，因此让我们再次从缓存中加载它：`DataCollatorForSeq2Seq`

```python
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
```

然后，我们可以实例化数据整理器，并使用它来定义我们的数据加载器：

```python
from torch.utils.data import DataLoader

batch_size = 8
train_dataloader = DataLoader(
    tokenized_datasets["train"],
    shuffle=True,
    collate_fn=data_collator,
    batch_size=batch_size,
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], collate_fn=data_collator, batch_size=batch_size
)
```

接下来要做的是定义我们要使用的优化器。与其他示例一样，我们将使用 ，它适用于大多数问题：`AdamW`

```python
from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=2e-5)
```

最后，我们将模型、优化器和数据加载器提供给该方法：`accelerator.prepare()`

```python
from accelerate import Accelerator

accelerator = Accelerator()
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)
```

🚨 如果您在 TPU 上进行训练，则需要将上述所有代码移动到专用的训练函数中。有关详细信息，请参阅[第 3 章](https://huggingface.co/course/chapter3)。

现在我们已经准备好了对象，还有三件事要做：

- 定义学习率计划。
- 实现对摘要进行后处理以进行评估的功能。
- 在 Hub 上创建一个存储库，我们可以将模型推送到该存储库。

对于学习速率计划，我们将使用前面部分中的标准线性计划：

```python
from transformers import get_scheduler

num_train_epochs = 10
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
```

对于后处理，我们需要一个函数，将生成的摘要拆分为用换行符分隔的句子。这是 ROUGE 指标期望的格式，我们可以通过以下代码片段来实现这一点：

```python
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # ROUGE expects a newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels
```

如果您还记得我们是如何定义 .`compute_metrics()``Seq2SeqTrainer`

最后，我们需要在 Hugging Face Hub 上创建一个模型存储库。为此，我们可以使用适当标题🤗的 Hub 库。我们只需要为我们的存储库定义一个名称，并且该库具有将存储库 ID 与用户配置文件组合在一起的实用函数：

```python
from huggingface_hub import get_full_repo_name

model_name = "test-bert-finetuned-squad-accelerate"
repo_name = get_full_repo_name(model_name)
repo_name
'lewtun/mt5-finetuned-amazon-en-es-accelerate'
```

现在，我们可以使用此存储库名称将本地版本克隆到将存储训练工件的结果目录：

```python
from huggingface_hub import Repository

output_dir = "results-mt5-finetuned-squad-accelerate"
repo = Repository(output_dir, clone_from=repo_name)
```

这将允许我们通过在训练期间调用该方法将工件推送回中心！现在让我们通过写出训练循环来结束我们的分析。`repo.push_to_hub()`

#### Training loop

用于总结的训练循环与我们遇到的其他🤗加速示例非常相似，大致分为四个主要步骤：

1. 通过遍历每个 epoch 的所有示例来训练模型。`train_dataloader`
2. 在每个纪元结束时生成模型摘要，方法是首先生成标记，然后将它们（和参考摘要）解码为文本。
3. 使用我们之前看到的相同技术计算 ROUGE 分数。
4. 保存检查点并将所有内容推送到中心。在这里，我们依赖于对象的漂亮参数，以便我们可以*异步*地推送每个纪元的检查点。这使我们能够继续训练，而不必等待与 GB 大小的模型相关的有点慢的上传！`blocking=False``Repository`

可以在以下代码块中看到这些步骤：

```python
from tqdm.auto import tqdm
import torch
import numpy as np

progress_bar = tqdm(range(num_training_steps))

for epoch in range(num_train_epochs):
    # Training
    model.train()
    for step, batch in enumerate(train_dataloader):
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    # Evaluation
    model.eval()
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            generated_tokens = accelerator.unwrap_model(model).generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )

            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )
            labels = batch["labels"]

            # If we did not pad to max length, we need to pad the labels too
            labels = accelerator.pad_across_processes(
                batch["labels"], dim=1, pad_index=tokenizer.pad_token_id
            )

            generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
            labels = accelerator.gather(labels).cpu().numpy()

            # Replace -100 in the labels as we can't decode them
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]
            decoded_preds = tokenizer.batch_decode(
                generated_tokens, skip_special_tokens=True
            )
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            decoded_preds, decoded_labels = postprocess_text(
                decoded_preds, decoded_labels
            )

            rouge_score.add_batch(predictions=decoded_preds, references=decoded_labels)

    # Compute metrics
    result = rouge_score.compute()
    # Extract the median ROUGE scores
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    result = {k: round(v, 4) for k, v in result.items()}
    print(f"Epoch {epoch}:", result)

    # Save and upload
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
    if accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)
        repo.push_to_hub(
            commit_message=f"Training in progress epoch {epoch}", blocking=False
        )
Epoch 0: {'rouge1': 5.6351, 'rouge2': 1.1625, 'rougeL': 5.4866, 'rougeLsum': 5.5005}
Epoch 1: {'rouge1': 9.8646, 'rouge2': 3.4106, 'rougeL': 9.9439, 'rougeLsum': 9.9306}
Epoch 2: {'rouge1': 11.0872, 'rouge2': 3.3273, 'rougeL': 11.0508, 'rougeLsum': 10.9468}
Epoch 3: {'rouge1': 11.8587, 'rouge2': 4.8167, 'rougeL': 11.7986, 'rougeLsum': 11.7518}
Epoch 4: {'rouge1': 12.9842, 'rouge2': 5.5887, 'rougeL': 12.7546, 'rougeLsum': 12.7029}
Epoch 5: {'rouge1': 13.4628, 'rouge2': 6.4598, 'rougeL': 13.312, 'rougeLsum': 13.2913}
Epoch 6: {'rouge1': 12.9131, 'rouge2': 5.8914, 'rougeL': 12.6896, 'rougeLsum': 12.5701}
Epoch 7: {'rouge1': 13.3079, 'rouge2': 6.2994, 'rougeL': 13.1536, 'rougeLsum': 13.1194}
Epoch 8: {'rouge1': 13.96, 'rouge2': 6.5998, 'rougeL': 13.9123, 'rougeLsum': 13.7744}
Epoch 9: {'rouge1': 14.1192, 'rouge2': 7.0059, 'rougeL': 14.1172, 'rougeLsum': 13.9509}
```

就是这样！运行此操作后，您将获得一个模型和结果，这些模型和结果与我们使用 .`Trainer`

### 7.5.7 Using your fine-tuned model

将模型推送到 Hub 后，您可以通过推理小部件或对象来使用它，如下所示：`pipeline`

```python
from transformers import pipeline

hub_model_id = "huggingface-course/mt5-small-finetuned-amazon-en-es"
summarizer = pipeline("summarization", model=hub_model_id)
```

我们可以将测试集（模型尚未看到的）中的一些示例提供给我们的管道，以了解摘要的质量。首先，让我们实现一个简单的函数，将评论、标题和生成的摘要一起显示：

```python
def print_summary(idx):
    review = books_dataset["test"][idx]["review_body"]
    title = books_dataset["test"][idx]["review_title"]
    summary = summarizer(books_dataset["test"][idx]["review_body"])[0]["summary_text"]
    print(f"'>>> Review: {review}'")
    print(f"\n'>>> Title: {title}'")
    print(f"\n'>>> Summary: {summary}'")
```

让我们看一下我们得到的一个英语示例：

```python
print_summary(100)
```

'>>> Review: Nothing special at all about this product... the book is too small and stiff and hard to write in. The huge sticker on the back doesn’t come off and looks super tacky. I would not purchase this again. I could have just bought a journal from the dollar store and it would be basically the same thing. It’s also really expensive for what it is.'

'>>> Title: Not impressed at all... buy something else'

'>>> Summary: Nothing special at all about this product'

这还不错！我们可以看到，我们的模型实际上已经能够通过用新词来增强评论的部分内容来执行*抽象*总结。也许我们模型最酷的方面是它是双语的，因此我们还可以生成西班牙语评论的摘要：

```python
print_summary(0)
```

'>>> Review: Es una trilogia que se hace muy facil de leer. Me ha gustado, no me esperaba el final para nada'

'>>> Title: Buena literatura para adolescentes'

'>>> Summary: Muy facil de leer'

摘要翻译成英文的“非常容易阅读”，在这种情况下，我们可以看到它是直接从评论中提取的。尽管如此，这显示了 mT5 模型的多功能性，并让您体验了处理多语言语料库的感觉！

接下来，我们将把注意力转向一个稍微复杂的任务：从头开始训练语言模型。

## 7.6 Training a causal language model from scratch

到目前为止，我们主要使用预训练模型，并通过重用预训练中的权重来针对新的用例对其进行微调。正如我们在第 [1 章](https://huggingface.co/course/chapter1)中所看到的，这通常被称为*迁移学习*，这是一种非常成功的策略，可以将 Transformer 模型应用于标记数据稀疏的大多数实际用例。在本章中，我们将采用不同的方法，从头开始训练一个全新的模型。如果您有大量数据，并且与用于可用模型的预训练数据有很大不同，这是一个很好的方法。但是，它也需要更多的计算资源来预训练语言模型，而不仅仅是微调现有模型。训练新模型有意义的示例包括由音符、DNA 等分子序列或编程语言组成的数据集。后者最近获得了牵引力，这要归功于 TabNine 和 GitHub 的 Copilot 等工具，这些工具由 OpenAI 的 Codex 模型提供支持，可以生成长序列的代码。这种文本生成任务最好使用自动回归或因果语言模型（如 GPT-2）来解决。

在本节中，我们将构建代码生成模型的缩小版本：我们将重点关注单行完成，而不是使用 Python 代码的子集完成完整的函数或类。在 Python 中处理数据时，您经常接触 Python 数据科学堆栈，该堆栈由 `matplotlib`、`seaborn` 、`pandas` 和 `scikit-learn` 库组成。使用这些框架时，通常需要查找特定的命令，因此，如果我们可以使用模型来为我们完成这些调用，那就太好了。

在第 [6 章](https://huggingface.co/course/chapter6)中，我们创建了一个高效的分词器来处理 Python 源代码，但我们仍然需要一个大规模的数据集来预训练模型。在这里，我们将分词器应用于派生自 GitHub 存储库的 Python 代码语料库。然后，我们将使用 API 和 🤗 Accelerate 来训练模型。让我们开始吧！`Trainer`

这实际上是在展示使用本节中所示的代码训练并上传到 Hub 的模型。你可以[在这里](https://huggingface.co/huggingface-course/codeparrot-ds?text=plt.imshow()找到它。请注意，由于在文本生成中发生了一些随机化，因此您可能会得到略有不同的结果。

### 7.6.1 Gathering the data

Python 代码可以从 GitHub 等代码存储库中大量获得，我们可以使用它通过抓取每个 Python 存储库来创建数据集。这是[《变形金刚》教科书](https://learning.oreilly.com/library/view/natural-language-processing/9781098136789/)中采用的预训练大型 GPT-2 模型的方法。使用一个大约180 GB的GitHub转储，其中包含大约2000万个Python文件称为`codeparrot`，作者建立了一个数据集，然后在[Hugging Face Hub](https://huggingface.co/datasets/transformersbook/codeparrot)上共享。

（Comment 1: 这个文件太大了，吾人先在colab上测试一番，再考虑要不要下载到本地。=> 卧槽，这个速度是真慢呀）

然而，在完整的语料库上进行训练既耗时又耗费计算，我们只需要与 Python 数据科学堆栈相关的数据集子集。因此，让我们首先筛选 `codeparrot` 包含此堆栈中任何库的所有文件的数据集。由于数据集的大小，我们希望避免下载它;相反，我们将使用流式处理功能来动态过滤它。为了帮助我们使用前面提到的库筛选代码示例，我们将使用以下函数：

```python
def any_keyword_in_string(string, keywords):
    for keyword in keywords:
        if keyword in string:
            return True
    return False
```

让我们在两个示例中对其进行测试：

```python
filters = ["pandas", "sklearn", "matplotlib", "seaborn"]
example_1 = "import numpy as np"
example_2 = "import pandas as pd"

print(
    any_keyword_in_string(example_1, filters), any_keyword_in_string(example_2, filters)
)
```

False True

我们可以使用它来创建一个函数，该函数将流式传输数据集并过滤我们想要的元素：

```python
from collections import defaultdict
from tqdm import tqdm
from datasets import Dataset


def filter_streaming_dataset(dataset, filters):
    filtered_dict = defaultdict(list)
    total = 0
    for sample in tqdm(iter(dataset)):
        total += 1
        if any_keyword_in_string(sample["content"], filters):
            for k, v in sample.items():
                filtered_dict[k].append(v)
    print(f"{len(filtered_dict['content'])/total:.2%} of data after filtering.")
    return Dataset.from_dict(filtered_dict)
```

然后我们可以简单地将这个函数应用于流数据集：

```python
# This cell will take a very long time to execute, so you should skip it and go to
# the next one!
from datasets import load_dataset

split = "train"  # "valid"
filters = ["pandas", "sklearn", "matplotlib", "seaborn"]

data = load_dataset(f"transformersbook/codeparrot-{split}", split=split, streaming=True)
filtered_data = filter_streaming_dataset(data, filters)
3.26% of data after filtering.
```

这给我们留下了大约 3% 的原始数据集，这仍然相当可观——生成的数据集为 6 GB，由 600,000 个 Python 脚本组成！

过滤整个数据集可能需要 2-3 小时，具体取决于您的机器和带宽。如果您不想自己经历这个漫长的过程，我们在 Hub 上提供过滤后的数据集供您下载：

```python
from datasets import load_dataset, DatasetDict

ds_train = load_dataset("huggingface-course/codeparrot-ds-train", split="train")
ds_valid = load_dataset("huggingface-course/codeparrot-ds-valid", split="validation")

raw_datasets = DatasetDict(
    {
        "train": ds_train,  # .shuffle().select(range(50000)),
        "valid": ds_valid,  # .shuffle().select(range(500))
    }
)

print(raw_datasets)
```

DatasetDict({
    train: Dataset({
        features: ['repo_name', 'path', 'copies', 'size', 'content', 'license'],
        num_rows: 606720
    })
    valid: Dataset({
        features: ['repo_name', 'path', 'copies', 'size', 'content', 'license'],
        num_rows: 3322
    })
})

预训练语言模型需要一段时间。我们建议您首先通过取消注释上面的两行来对数据样本运行训练循环，并确保训练成功完成并存储模型。没有什么比训练运行在最后一步失败更令人沮丧的了，因为您忘记创建文件夹或因为训练循环结束时有拼写错误！

让我们看一下数据集中的一个示例。我们只显示每个字段的前 200 个字符：

```python
for key in raw_datasets["train"][0]:
    print(f"{key.upper()}: {raw_datasets['train'][0][key][:200]}")
```

'REPO_NAME: kmike/scikit-learn'
'PATH: sklearn/utils/__init__.py'
'COPIES: 3'
'SIZE: 10094'
'''CONTENT: """
The :mod:`sklearn.utils` module includes various utilites.
"""

from collections import Sequence

import numpy as np
from scipy.sparse import issparse
import warnings

from .murmurhash import murm
LICENSE: bsd-3-clause'''

我们可以看到 `content` 字段包含我们希望模型训练的代码。现在我们有了一个数据集，我们需要准备文本，使它们采用适合预训练的格式。

### 7.6.2 Preparing the dataset

第一步是标记数据，这样我们就可以用它来训练。由于我们的目标主要是自动完成短函数调用，因此我们可以保持上下文大小相对较小。这样做的好处是，我们可以更快地训练模型，并且需要的内存要少得多。如果应用程序必须具有更多上下文（例如，如果希望模型基于具有函数定义的文件编写单元测试），请确保增加该数字，但也要记住，这会占用更大的 GPU 内存。现在，让我们将上下文大小固定为 128 个令牌，而不是 GPT-2 或 GPT-3 中分别使用的 1,024 或 2,048 个令牌。

大多数文档包含超过 128 个标记，因此只需将输入截断到最大长度即可消除我们数据集的很大一部分。相反，我们将使用该选项对整个输入进行标记化，并将其拆分为几个块，就像我们[在第 6 章](https://huggingface.co/course/chapter6/4)中所做的那样。我们还将使用该选项自动返回每个创建块的长度。通常，最后一个块会小于上下文大小，我们将删除这些块以避免填充问题;我们真的不需要它们，因为我们有大量的数据。`return_overflowing_tokens``return_length`

![将大文本分成几块。](https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter7/chunking_texts.svg)

让我们通过查看前两个示例来确切地了解其工作原理：

```python
from transformers import AutoTokenizer

context_length = 128
tokenizer = AutoTokenizer.from_pretrained("huggingface-course/code-search-net-tokenizer")

outputs = tokenizer(
    raw_datasets["train"][:2]["content"],
    truncation=True,
    max_length=context_length,
    return_overflowing_tokens=True,
    return_length=True,
)

print(f"Input IDs length: {len(outputs['input_ids'])}")
print(f"Input chunk lengths: {(outputs['length'])}")
print(f"Chunk mapping: {outputs['overflow_to_sample_mapping']}")
```

Input IDs length: 34
Input chunk lengths: [128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 117, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 41]
Chunk mapping: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

我们可以看到，从这两个示例中，我们总共得到了 34 个段。查看块长度，我们可以看到两个文档末尾的块都少于 128 个标记（分别为 117 个和 41 个）。这些只占我们总块的一小部分，因此我们可以安全地将它们扔掉。通过该字段，我们还可以重建哪些块属于哪些输入样本。`overflow_to_sample_mapping`

通过此操作，我们在数据集中使用🤗了该函数的一个方便功能，即它不需要一对一的地图;正如我们在第 [3 节](https://huggingface.co/course/chapter7/3)中看到的，我们可以创建比输入批处理更多或更少的元素的批次。这在执行数据增强或数据筛选等更改元素数量的操作时非常有用。在我们的例子中，当将每个元素标记化为指定上下文大小的块时，我们从每个文档创建许多样本。我们只需要确保删除现有列，因为它们的大小冲突。如果我们想保留它们，我们可以适当地重复它们并在调用中返回它们：`Dataset.map()``Dataset.map()`

```python
def tokenize(element):
    outputs = tokenizer(
        element["content"],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
    )
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length == context_length:
            input_batch.append(input_ids)
    return {"input_ids": input_batch}


tokenized_datasets = raw_datasets.map(
    tokenize, batched=True, remove_columns=raw_datasets["train"].column_names
)
tokenized_datasets
DatasetDict({
    train: Dataset({
        features: ['input_ids'],
        num_rows: 16702061
    })
    valid: Dataset({
        features: ['input_ids'],
        num_rows: 93164
    })
})
```

我们现在有 1670 万个示例，每个示例有 128 个代币，总共相当于约 21 亿个代币。作为参考，OpenAI 的 GPT-3 和 Codex 模型分别在 300 亿个和 1000 亿个代币上进行训练，其中 Codex 模型是从 GPT-3 检查点初始化的。我们在本节中的目标不是与这些模型竞争，这些模型可以生成长而连贯的文本，而是创建一个缩小的版本，为数据科学家提供快速的自动完成功能。

现在我们已经准备好了数据集，让我们设置模型！

✏️ **试试看！**删除所有小于上下文大小的块在这里并不是一个大问题，因为我们使用的是小上下文窗口。随着上下文大小的增加（或者，如果您有一个短文档的语料库），被丢弃的块的比例也会增加。准备数据的更有效方法是将所有标记化样本联接到一个批处理中，中间有一个标记，然后对串联序列执行分块。作为练习，修改函数以利用该方法。请注意，您需要从分词器中设置和删除其他参数，以获取令牌 ID 的完整序列。`eos_token_id``tokenize()``truncation=False`

### 7.6.3 Initializing a new model

我们的第一步是全新初始化一个 GPT-2 模型。我们将为我们的模型使用与小型 GPT-2 模型相同的配置，因此我们加载预训练的配置，确保分词器大小与模型词汇大小匹配，并传递 `bos` 和 `eos`（序列的开始和结束）令牌 ID：

```python
from transformers import AutoTokenizer, GPT2LMHeadModel, AutoConfig

config = AutoConfig.from_pretrained(
    "gpt2",
    vocab_size=len(tokenizer),
    n_ctx=context_length,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)
```

（Comment:  看来即使是从头训练一个新模型，也可以直接用GPT2的分词器）

通过该配置，我们可以加载一个新模型。请注意，这是我们第一次不使用 `from_pretrained()` 函数，因为我们实际上是在自己初始化一个模型：

```python
model = GPT2LMHeadModel(config)
model_size = sum(t.numel() for t in model.parameters())
print(f"GPT-2 size: {model_size/1000**2:.1f}M parameters")
GPT-2 size: 124.2M parameters
```

我们的模型有 124M 个参数，我们必须调整这些参数。在开始训练之前，我们需要设置一个数据整理器来负责创建批处理。我们可以使用专为语言建模而设计的 `DataCollatorForLanguageModeling` 整理器（顾名思义）。除了堆叠和填充批处理外，它还负责创建语言模型标签——在因果语言建模中，输入也用作标签（只是移动了一个元素），并且此数据整理器在训练期间动态创建它们，因此我们不需要复制 `input_ids` .

请注意，`DataCollatorForLanguageModeling` 支持掩码语言建模 （MLM） 和因果语言建模 （CLM）。默认情况下，它为 MLM 准备数据，但我们可以通过设置 `mlm=False` 参数来切换到 CLM：

```python
from transformers import DataCollatorForLanguageModeling

tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
```

让我们看一个例子：

```python
out = data_collator([tokenized_datasets["train"][i] for i in range(5)])
for key in out:
    print(f"{key} shape: {out[key].shape}")
```

input_ids shape: torch.Size([5, 128])
attention_mask shape: torch.Size([5, 128])
labels shape: torch.Size([5, 128])

我们可以看到，这些示例已经堆叠在一起，并且所有张量都具有相同的形状。

⚠️ 在模型内部移动输入和标签以使其对齐，因此数据整理器只需复制输入即可创建标签。

现在，我们已经做好了实际训练模型的所有准备工作——毕竟这还不算什么工作！在开始训练之前，我们应该登录 Hugging Face。如果您在笔记本中工作，则可以使用以下实用程序函数执行此操作：

```python
from huggingface_hub import notebook_login

notebook_login()
```

这将显示一个小部件，您可以在其中输入您的 Hugging Face 登录凭据。

如果您不在笔记本中工作，只需在终端中键入以下行：

```shell
huggingface-cli login
```

剩下要做的就是配置训练参数并启动 .我们将使用余弦学习速率计划，并进行一些预热，有效批处理大小为 256 （ * ）。当单个批处理无法放入内存时，使用梯度累积，并通过多次向前/向后传递以增量方式建立梯度。当我们使用 🤗 Accelerate 创建训练循环时，我们将看到这一点。`Trainer``per_device_train_batch_size``gradient_accumulation_steps`

```python
from transformers import Trainer, TrainingArguments

args = TrainingArguments(
    output_dir="codeparrot-ds",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    evaluation_strategy="steps",
    eval_steps=5_000,
    logging_steps=5_000,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    weight_decay=0.1,
    warmup_steps=1_000,
    lr_scheduler_type="cosine",
    learning_rate=5e-4,
    save_steps=5_000,
    fp16=True,
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["valid"],
)
```

现在我们可以开始 `Trainer` 并等待训练完成。根据您是在完整训练集还是训练集的子集上运行它，这将分别需要 20 小时或 2 小时，所以请喝几杯咖啡和一本好书来阅读！

```python
trainer.train()
```

训练完成后，我们可以将模型和分词器推送到 Hub：

```
trainer.push_to_hub()
```

✏️ **试试看！**除了从原始文本到训练 GPT-2 之外，我们只花了大约 30 行代码。用你自己的数据集试试看，看看你是否能得到好的结果！`TrainingArguments`

💡 如果您有权访问具有多个 GPU 的计算机，请尝试在那里运行代码。自动管理多台机器，这可以大大加快训练速度。`Trainer`

### 7.6.4 Code generation with a pipeline

现在是关键时刻：让我们看看经过训练的模型的实际效果如何！我们可以在日志中看到损失稳步下降，但为了测试模型，让我们看看它在某些提示上的效果如何。为此，我们将模型包装在文本生成中，如果有的话，我们会将其放在 GPU 上以进行快速生成：`pipeline`

```python
import torch
from transformers import pipeline

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
pipe = pipeline(
    "text-generation", model="huggingface-course/codeparrot-ds", device=device
)
```

（Comment:  看来生成任务也可以指定CPU）

让我们从创建散点图的简单任务开始：

```python
txt = """\
# create some data
x = np.random.randn(100)
y = np.random.randn(100)

# create scatter plot with x, y
"""
print(pipe(txt, num_return_sequences=1)[0]["generated_text"])
# create some data
x = np.random.randn(100)
y = np.random.randn(100)

# create scatter plot with x, y
plt.scatter(x, y)

# create scatter
```

结果看起来是正确的。它也适用于手术吗？让我们看看我们是否可以从两个数组创建一个：`pandas``DataFrame`

```python
txt = """\
# create some data
x = np.random.randn(100)
y = np.random.randn(100)

# create dataframe from x and y
"""
print(pipe(txt, num_return_sequences=1)[0]["generated_text"])
# create some data
x = np.random.randn(100)
y = np.random.randn(100)

# create dataframe from x and y
df = pd.DataFrame({'x': x, 'y': y})
df.insert(0,'x', x)
for
```

很好，这是正确的答案——尽管它随后再次插入了列。由于生成的令牌数量有限，因此以下循环被切断。让我们看看我们是否可以做一些更复杂的事情，让模型帮助我们使用这个操作：`x``for``groupby`

```python
txt = """\
# dataframe with profession, income and name
df = pd.DataFrame({'profession': x, 'income':y, 'name': z})

# calculate the mean income per profession
"""
print(pipe(txt, num_return_sequences=1)[0]["generated_text"])
# dataframe with profession, income and name
df = pd.DataFrame({'profession': x, 'income':y, 'name': z})

# calculate the mean income per profession
profession = df.groupby(['profession']).mean()

# compute the
```

不错;这是正确的方法。最后，让我们看看我们是否也可以使用它来设置一个随机森林模型：`scikit-learn`

```python
txt = """
# import random forest regressor from scikit-learn
from sklearn.ensemble import RandomForestRegressor

# fit random forest model with 300 estimators on X, y:
"""
print(pipe(txt, num_return_sequences=1)[0]["generated_text"])
# import random forest regressor from scikit-learn
from sklearn.ensemble import RandomForestRegressor

# fit random forest model with 300 estimators on X, y:
rf = RandomForestRegressor(n_estimators=300, random_state=random_state, max_depth=3)
rf.fit(X, y)
rf
```

从这几个例子来看，该模型似乎已经学习了 Python 数据科学堆栈的一些语法（当然，在将模型部署到现实世界中之前，我们需要对其进行更彻底的评估）。然而，有时它需要对模型训练进行更多定制，以实现给定用例所需的性能。例如，如果我们想动态更新批处理大小或有一个条件训练循环来即时跳过不良示例，该怎么办？一种选择是将 子类化并添加必要的更改，但有时从头开始编写训练循环会更简单。这就是 🤗 Accelerate 的用武之地。`Trainer`

### 7.6.5 使用 Accelerate 进行🤗训练

我们已经了解了如何使用 训练模型，这可以允许一些自定义。但是，有时我们想要完全控制训练循环，或者我们想进行一些奇特的更改。在这种情况下🤗，Accelerate 是一个不错的选择，在本节中，我们将介绍使用它来训练模型的步骤。为了让事情变得更有趣，我们还将在训练循环中添加一个转折。`Trainer`

```python
keytoken_ids = []
for keyword in [
    "plt",
    "pd",
    "sk",
    "fit",
    "predict",
    " plt",
    " pd",
    " sk",
    " fit",
    " predict",
    "testtest",
]:
    ids = tokenizer([keyword]).input_ids[0]
    if len(ids) == 1:
        keytoken_ids.append(ids[0])
    else:
        print(f"Keyword has not single token: {keyword}")
```

'Keyword has not single token: testtest'

太好了，这似乎很好用！现在，我们可以编写一个自定义损失函数，该函数将输入序列、logit 和我们刚刚选择的密钥标记作为输入。首先，我们需要对齐对数和输入：向右移动一个的输入序列形成标签，因为下一个标记是当前标记的标签。我们可以通过从输入序列的第二个标记开始标签来实现这一点，因为模型无论如何都不会对第一个标记进行预测。然后我们切断最后一个 logit，因为我们没有遵循完整输入序列的令牌的标签。有了它，我们可以计算每个样本的损失，并计算每个样本中所有关键字的出现次数。最后，我们使用出现次数作为权重来计算所有样本的加权平均值。由于我们不想丢弃所有没有关键字的样本，因此我们在权重中添加 1：

```python
from torch.nn import CrossEntropyLoss
import torch


def keytoken_weighted_loss(inputs, logits, keytoken_ids, alpha=1.0):
    # Shift so that tokens < n predict n
    shift_labels = inputs[..., 1:].contiguous()
    shift_logits = logits[..., :-1, :].contiguous()
    # Calculate per-token loss
    loss_fct = CrossEntropyLoss(reduce=False)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    # Resize and average loss per sample
    loss_per_sample = loss.view(shift_logits.size(0), shift_logits.size(1)).mean(axis=1)
    # Calculate and scale weighting
    weights = torch.stack([(inputs == kt).float() for kt in keytoken_ids]).sum(
        axis=[0, 2]
    )
    weights = alpha * (1.0 + weights)
    # Calculate weighted average
    weighted_loss = (loss_per_sample * weights).mean()
    return weighted_loss
```

在开始训练这个令人敬畏的新损失函数之前，我们需要准备一些东西：

- 我们需要数据加载器来批量加载数据。
- 我们需要设置权重衰减参数。
- 我们时不时地想要计算，因此将评估代码包装在函数中是有意义的。

让我们从数据加载器开始。我们只需要将数据集的格式设置为 ，然后就可以将其传递给具有适当批量大小的 PyTorch：`"torch"``DataLoader`

```python
from torch.utils.data.dataloader import DataLoader

tokenized_dataset.set_format("torch")
train_dataloader = DataLoader(tokenized_dataset["train"], batch_size=32, shuffle=True)
eval_dataloader = DataLoader(tokenized_dataset["valid"], batch_size=32)
```

接下来，我们对参数进行分组，以便优化器知道哪些参数将获得额外的权重衰减。通常，所有偏差和 LayerNorm 权重项都不受此限制;以下是我们执行此操作的方法：

```python
weight_decay = 0.1


def get_grouped_params(model, no_decay=["bias", "LayerNorm.weight"]):
    params_with_wd, params_without_wd = [], []
    for n, p in model.named_parameters():
        if any(nd in n for nd in no_decay):
            params_without_wd.append(p)
        else:
            params_with_wd.append(p)
    return [
        {"params": params_with_wd, "weight_decay": weight_decay},
        {"params": params_without_wd, "weight_decay": 0.0},
    ]
```

由于我们希望在训练期间定期在验证集上评估模型，因此我们也为此编写一个函数。它只是通过评估数据加载器运行，并收集跨进程的所有损失：

```python
def evaluate():
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(batch["input_ids"], labels=batch["input_ids"])

        losses.append(accelerator.gather(outputs.loss))
    loss = torch.mean(torch.cat(losses))
    try:
        perplexity = torch.exp(loss)
    except OverflowError:
        perplexity = float("inf")
    return loss.item(), perplexity.item()
```

有了这个功能，我们可以定期报告丢失和[困惑](https://huggingface.co/course/chapter7/3)。接下来，我们重新定义模型，以确保再次从头开始训练：`evaluate()`

```python
model = GPT2LMHeadModel(config)
```

然后，我们可以定义优化器，使用之前的函数来拆分权重衰减的参数：

```python
from torch.optim import AdamW

optimizer = AdamW(get_grouped_params(model), lr=5e-4)
```

现在，让我们准备模型、优化器和数据加载器，以便我们可以开始训练：

```python
from accelerate import Accelerator

accelerator = Accelerator(fp16=True)

model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)
```

🚨 如果您在 TPU 上进行训练，则需要将从上面单元格开始的所有代码移动到专用的训练函数中。有关详细信息，请参阅[第 3 章](https://huggingface.co/course/chapter3)。

现在我们已经发送了 ，我们可以使用它的长度来计算训练步骤的数量。请记住，我们应该始终在准备数据加载器后执行此操作，因为该方法会更改其长度。我们使用从学习率到 0 的经典线性时间表：`train_dataloader``accelerator.prepare()`

```python
from transformers import get_scheduler

num_train_epochs = 1
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=1_000,
    num_training_steps=num_training_steps,
)
```

最后，要将我们的模型推送到中心，我们需要在工作文件夹中创建一个对象。如果您尚未登录，请先登录 Hugging Face Hub。我们将根据我们想要为模型提供的模型 ID 确定存储库名称（请随意将 替换为您自己的选择;它只需要包含您的用户名，这就是函数的作用）：`Repository``repo_name``get_full_repo_name()`

```python
from huggingface_hub import Repository, get_full_repo_name

model_name = "codeparrot-ds-accelerate"
repo_name = get_full_repo_name(model_name)
repo_name
'sgugger/codeparrot-ds-accelerate'
```

然后，我们可以将该存储库克隆到本地文件夹中。如果它已经存在，则此本地文件夹应该是我们正在使用的存储库的现有克隆：

```python
output_dir = "codeparrot-ds-accelerate"
repo = Repository(output_dir, clone_from=repo_name)
```

现在，我们可以通过调用该方法上传我们保存的任何内容。这将有助于我们在每个纪元结束时上传中间模型。`output_dir``repo.push_to_hub()`

在训练之前，让我们运行一个快速测试，看看评估函数是否正常工作：

```python
evaluate()
(10.934126853942871, 56057.14453125)
```

这些是非常高的损失和困惑值，但这并不奇怪，因为我们还没有训练模型。这样一来，我们就准备好了编写训练脚本的核心部分：训练循环。在训练循环中，我们遍历数据加载器并将批处理传递给模型。有了logits，我们就可以评估我们的自定义损失函数。我们按梯度累积步骤的数量来缩放损失，以便在聚合更多步骤时不会产生更大的损失。在优化之前，我们还会裁剪渐变以获得更好的收敛效果。最后，每隔几个步骤，我们就会使用新函数评估评估集上的模型：`evaluate()`

```python
from tqdm.notebook import tqdm

gradient_accumulation_steps = 8
eval_steps = 5_000

model.train()
completed_steps = 0
for epoch in range(num_train_epochs):
    for step, batch in tqdm(
        enumerate(train_dataloader, start=1), total=num_training_steps
    ):
        logits = model(batch["input_ids"]).logits
        loss = keytoken_weighted_loss(batch["input_ids"], logits, keytoken_ids)
        if step % 100 == 0:
            accelerator.print(
                {
                    "samples": step * samples_per_step,
                    "steps": completed_steps,
                    "loss/train": loss.item() * gradient_accumulation_steps,
                }
            )
        loss = loss / gradient_accumulation_steps
        accelerator.backward(loss)
        if step % gradient_accumulation_steps == 0:
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            completed_steps += 1
        if (step % (eval_steps * gradient_accumulation_steps)) == 0:
            eval_loss, perplexity = evaluate()
            accelerator.print({"loss/eval": eval_loss, "perplexity": perplexity})
            model.train()
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
            if accelerator.is_main_process:
                tokenizer.save_pretrained(output_dir)
                repo.push_to_hub(
                    commit_message=f"Training in progress step {step}", blocking=False
                )
```

就是这样——您现在拥有了自己的因果语言模型（如 GPT-2）的自定义训练循环，您可以根据需要进一步自定义。

✏️ **试试看！**您可以根据自己的用例创建自己的自定义损失函数，也可以在训练循环中添加另一个自定义步骤。

✏️ **试试看！**在进行长时间的训练实验时，最好使用TensorBoard或Weights & Biases等工具记录重要指标。将适当的日志记录添加到训练循环中，以便您始终可以检查训练的进展情况。





