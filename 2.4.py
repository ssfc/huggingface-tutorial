from transformers import AutoTokenizer
# Q: BertTokenizer是干啥的？
# `BertTokenizer` 的主要功能有：
# 1. 分词：将输入文本分割成一个个单词或子词，构成模型输入的标记（tokens）序列。
# 2. 添加特殊标记：在输入文本的开头和结尾添加 `[CLS]` 和 `[SEP]` 等特殊标记，用于标识句子的开始和结束，以及分隔多个句子。
# 3. 编码：将分词后的标记序列转换为模型输入的张量形式，包括转换为对应的整数 ID 和添加对齐用的填充标记。
from transformers import BertTokenizer

# Section 1: Word-based
'''
tokenized_text = "Jim Henson was a puppeteer".split()
print(tokenized_text)
'''
# Section 2: Character-based

# Section 3: Subword tokenization

# Section 4: And more!

# Section 5: Loading and saving
# tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
# print(tokenizer)

# 这行代码使用 Hugging Face Transformers 库中的 `AutoTokenizer` 类来加载预训练的 BERT 模型的分词器（Tokenizer）。
# `AutoTokenizer` 是一个方便的工具，用于根据模型名称自动选择合适的分词器。
# 具体来说，`AutoTokenizer.from_pretrained("bert-base-cased")` 的意思是：
# 1. `"bert-base-cased"` 是一个预训练的 BERT 模型的名称。
# 在 Hugging Face Transformers 库中，模型名称通常由两部分组成，第一部分表示模型的基本结构和配置，第二部分表示模型的大小和是否区分大小写。
# 在这个例子中，`"bert-base-cased"` 表示一个基本的 BERT 模型，且模型在预训练时区分了大小写。
# 2. `AutoTokenizer.from_pretrained("bert-base-cased")` 通过指定模型名称，自动选择并加载与该模型对应的分词器。
# 这样，我们就可以直接使用 `tokenizer` 对象来进行文本分词，将文本转换为模型可以接受的输入格式。
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
print(tokenizer)

result = tokenizer("Using a Transformer network is simple")
print(result)
# 通常情况下，当我们加载预训练的模型和分词器时，每次都要从 Hugging Face 模型库中下载并加载，这可能会花费一些时间和网络资源。
# 为了避免重复下载和加载，我们可以使用 `tokenizer.save_pretrained()` 将分词器保存到本地文件夹中，然后在以后的使用中，
# 只需要从本地文件夹加载分词器即可，这样会更快捷方便。
tokenizer.save_pretrained("directory_on_my_computer")

# Section 6: Encoding

# Section 7: Tokenization
# Q: AutoTokenizer是干啥的？
# `AutoTokenizer` 是 Hugging Face Transformers 库中的一个类，它是一个自动加载预训练模型的分词器（Tokenizer）的工具类。
# 在自然语言处理（NLP）中，分词器用于将输入文本（句子、段落等）拆分成单词或子词的序列，以便机器学习模型能够处理和理解文本。
# `AutoTokenizer` 的主要作用是根据给定的模型名称或 checkpoint 来自动选择和加载对应的预训练模型的分词器。
# 这样，你可以通过一个简单的 API 调用来加载不同模型的分词器，而不需要手动指定特定模型的分词器。
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

# sequence = "Using a Transformer network is simple"
# tokenize object code
# sequence = "*146*171*5*171*1*38*171*23*38*171*23*38*171*23*17*5*"
# tokens = tokenizer.tokenize(sequence)

# tokenize source code
sequence = "left(rotate(tree, var));<br>var = right(var);<br>right(var) = left(var);<br>if (left(var) == parent(root(" \
           "tree)) ){<br>    parent(left(var)) = var }<br>parent(var) = parent(var);<br>if (parent(var) == parent(" \
           "root(tree)) ){<br>   root(tree) = var}<br>if (var == left(parent(var))){<br>    left(parent(var)) = var " \
           "}<br>else(){ right(parent(var)) = var;<br>    left(var) = var}<br>parent(var) = var;<br>"
tokens = tokenizer.tokenize(sequence)

print("tokens: ", tokens)

# Section 8: From tokens to input IDs
ids = tokenizer.convert_tokens_to_ids(tokens)

print(ids)

raw_inputs = "Using a Transformer network is simple"
inputs = tokenizer(raw_inputs)
print(inputs)
# Comment: 输出是不一样的，头部尾部多了俩元素

# 2.4.9 Decoding
'''
decoded_string = tokenizer.decode([7993, 170, 11303, 1200, 2443, 1110, 3014])
print(decoded_string)
'''






