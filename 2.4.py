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

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
print(tokenizer)

result = tokenizer("Using a Transformer network is simple")
print(result)
tokenizer.save_pretrained("directory_on_my_computer")

# Section 6: Encoding

# Section 7: Tokenization
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






