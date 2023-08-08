# https://huggingface.co/learn/nlp-course/chapter2/2?fw=pt
# Comment: 将端到端pipeline分解为各个步骤。

import torch
# Q: huggingface AutoModel是干啥的？
# `AutoModel` 是 Hugging Face Transformers 库中的一个类，它是一个方便的工具，用于根据指定的模型名称或 checkpoint 来自动选择和加载相应的预训练模型。
# 在深度学习中，使用预训练模型是一种常见的方式，可以为各种自然语言处理（NLP）任务提供强大的基础。
# 然而，Hugging Face Transformers 库提供了许多不同类型的预训练模型（例如 BERT、GPT、RoBERTa 等），而每个模型可能有多个不同的变体和配置。
# 为了方便用户加载和使用不同模型，`AutoModel` 类允许用户通过一个简单的 API 来加载预训练模型，而无需手动选择特定的模型或配置。
# `AutoModel` 的主要作用是根据给定的模型名称或 checkpoint 来自动选择和加载对应的预训练模型。这样，你只需要指定模型名称或 checkpoint，`AutoModel` 就会自动选择和加载与之对应的预训练模型。
from transformers import AutoModel
# Q: huggingface AutoModelForSequenceClassification是干啥的？
# `AutoModelForSequenceClassification` 是 Hugging Face Transformers 库中的一个类，它是用于序列分类任务的预训练模型的自动加载工具。
# 在自然语言处理（NLP）中，序列分类任务是指将输入的文本序列（例如句子、段落）分类到预定义的类别或标签中的任务。
# `AutoModelForSequenceClassification` 类的作用是根据指定的模型名称或 checkpoint 来自动选择和加载适用于序列分类任务的预训练模型。
# 它是 `AutoModel` 类的一个子类，在加载模型的同时，它会自动设置模型的输出层，以适应序列分类任务的特定需求。
from transformers import AutoModelForSequenceClassification
# Q: huggingface AutoTokenizer是干啥的？
# `AutoTokenizer` 是 Hugging Face Transformers 库中的一个类，它是一个自动加载预训练模型的分词器（Tokenizer）的工具类。
# 在自然语言处理（NLP）中，分词器用于将输入文本（句子、段落等）拆分成单词或子词的序列，以便机器学习模型能够处理和理解文本。
# `AutoTokenizer` 的主要作用是根据给定的模型名称或 checkpoint 来自动选择和加载对应的预训练模型的分词器。
# 这个类是 `AutoModel` 和 `AutoModelForSequenceClassification` 类的分词器版本，它使得在加载预训练模型和分词器时变得非常方便和灵活。
# 使用 `AutoTokenizer` 有以下几个优点：
# 1. 自动选择模型：无需手动指定模型名称，`AutoTokenizer` 会根据提供的模型名称自动选择和加载对应的分词器。
# 2. 多种模型支持：`AutoTokenizer` 支持加载各种不同的预训练模型的分词器，如 BERT、GPT、RoBERTa 等。
# 3. 方便的代码迁移：如果你在代码中使用了 `AutoTokenizer` 来加载分词器，当你改变模型时，只需更改模型名称，而不需要修改其他代码。
from transformers import AutoTokenizer
from transformers import pipeline

'''
classifier = pipeline("sentiment-analysis")
result = classifier(
    [
        "I've been waiting for a HuggingFace course my whole life.",
        "I hate this so much!",
    ]
)
print(result)
'''

# Section 1: Preprocessing with a tokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
# padding是自动填充，truncation=true是超过最大长度截断，return_tensors是返回pytorch格式
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
print(inputs)

# 奇怪的是，怎么输出向量的维数和输入单词的数量不一样？这因为是一些单词(比如huggingface)会被分词器分成两个

# Section 2: Going through the model
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModel.from_pretrained(checkpoint)
# print(model)

# Section 3: A high-dimensional vector?
# Q: model(**inputs)为什么inputs前面有**?
# 在 Python 中，双星号（`**`）用于解包字典（或关键字参数）并将其作为关键字参数传递给函数。
# 在 `model(**inputs)` 中，`**inputs` 表示将 `inputs` 这个字典中的键值对解包，并将其中的键作为关键字参数名，将对应的值作为关键字参数值传递给函数 `model`。
# 在 Hugging Face Transformers 库中，`model` 是一个预训练模型的实例，可以接受一组特定的输入参数，例如 `input_ids`、`attention_mask` 等。
# 而 `inputs` 是一个字典，其中包含了这些输入参数，键为参数名，值为相应的张量数据。
# 通过使用 `**inputs`，我们可以将字典 `inputs` 中的键值对解包，并将其传递给 `model` 函数作为关键字参数。
# 这样做的好处是，可以根据需要轻松地添加或修改输入参数，而不需要显式地逐个传递每个参数。
outputs = model(**inputs)
# Q: outputs.last_hidden_state是在logits前一层吗？
# 是的，`outputs.last_hidden_state` 在 `logits` 前一层。
# 在 Transformers 模型中，`outputs.last_hidden_state` 包含了模型的中间表示，也称为上下文向量（contextual embeddings）或特征向量（feature embeddings）。
# 它是模型在输入序列上进行处理后的最终输出，记录了输入序列中每个位置（单词或子词）的编码表示。
# `outputs.last_hidden_state` 的维度是 `[batch_size, sequence_length, hidden_size]`，
# 其中 `batch_size` 表示批量大小，`sequence_length` 表示序列的长度，`hidden_size` 表示隐藏层的大小（即每个位置的特征向量维度）。
# `logits` 则是在 `outputs.last_hidden_state` 基础上通过模型的最后一层或特定输出层得到的预测输出。
# 对于分类任务，`logits` 是模型在每个类别上的分数或得分，表示输入属于每个类别的概率。
# `logits` 的维度通常是 `[batch_size, num_classes]`，其中 `num_classes` 表示分类任务中的类别数。
# 在文本分类任务中，通常会使用 `outputs.last_hidden_state` 来获取输入序列的表示，然后通过一些后续的操作（例如平均池化、最大池化等）来获得整个序列的固定表示。
# 然后，这个固定表示可以作为输入传递给分类层，生成最终的 `logits`，并通过 softmax 函数获得分类概率。
print(outputs.last_hidden_state.shape)

# Section 4: Model heads: Making sense out of numbers
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model(**inputs)
print(outputs.logits.shape)

# Section 5: Postprocessing the output
# Q: 什么是logits?
# 在机器学习和深度学习中，Logits 是指模型的输出层（或最后一层）的原始预测结果，尚未经过概率化转换的值。Logits 是一个向量，其中的每个元素表示模型对于不同类别的预测得分或原始输出值。
# Logits 的值通常不具有直接的概率解释，它们可以是任意实数，可以是正数、负数或零。在分类问题中，对于每个类别，logits 的数值越高，表示模型对该类别的预测置信度越高；而数值越低，则表示模型对该类别的预测置信度越低。
# 为了将 logits 转换为概率分布，常见的做法是使用 softmax 函数，将 logits 映射为概率值。
# Softmax 函数将每个 logits 值转换为一个介于 0 到 1 之间的概率，并且所有类别的概率之和为 1。这样可以更直观地解释模型的输出，得到每个类别的概率预测。
# 总结起来，logits 是指模型输出层的原始预测结果，表示模型对不同类别的预测得分或原始输出值，尚未经过概率化转换。通过应用 softmax 函数，可以将 logits 转换为概率分布，提供类别预测的概率信息。
print(outputs.logits)
# Q: predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)中dim=-1是什么意思？
# 在 PyTorch 中，`dim=-1` 表示在张量的最后一个维度上进行操作。在 `torch.nn.functional.softmax` 函数中，`dim` 参数用于指定 softmax 操作应该沿着哪个维度进行计算。
# 考虑一个张量 `x`，它的维度是 `[a, b, c, ...]`，其中 `a`, `b`, `c`, ... 是张量的维度大小。
# 如果我们使用 `dim=-1`，那么 softmax 操作会在最后一个维度上进行计算。例如，对于维度为 `[batch_size, num_classes]` 的张量，
# 使用 `dim=-1` 表示 softmax 操作会在 `num_classes` 这个维度上进行计算，即对每个样本的类别分数进行 softmax。
# 在代码中，`outputs.logits` 是模型在分类任务上的原始预测输出，是一个维度为 `[batch_size, num_classes]` 的张量。
# 通过 `torch.nn.functional.softmax(outputs.logits, dim=-1)`，我们对每个样本的类别分数进行 softmax，得到分类概率分布。
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)

# Q: model.config.id2label是什么意思？
# 在 Hugging Face Transformers 中，`model.config.id2label` 是一个词典（字典），用于将模型输出的类别标签 ID（整数值）映射回对应的类别标签（字符串标签）。
# 在文本分类任务中，通常模型的输出是一个张量，每个元素代表一个类别的得分或概率。为了将这些得分或概率转换为对应的类别标签，我们需要一个映射关系，将类别标签的 ID 映射回实际的类别名称。
# `model.config.id2label` 就是用来构建这种映射关系的词典。它的键是类别标签的 ID（整数值），值是对应的类别名称（字符串标签）。
print(model.config.id2label)








