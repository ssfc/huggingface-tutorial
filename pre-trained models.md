# GPT

默认存储路径：/home/ssfc/.cache/huggingface



## distilbert/distilgpt2

https://huggingface.co/distilbert/distilgpt2

DistilGPT2 is an English-language model pre-trained with the supervision of the 124 million parameter version of GPT-2. DistilGPT2, which has 82 million parameters. 

DistilGPT2本地存储空间340M。



Comment:  这样看来distilgpt2并不比gpt2小多少嘛。(2024年2月9日)



## openai-community/gpt2

https://huggingface.co/openai-community/gpt2

This is the **smallest** version of GPT-2, with 124M parameters.

GPT2本地存储空间526M。

124/82 = 1.512

526/340 = 1.547

参数量的比值和本地存储空间的比值还挺接近。(2024年2月9日)



## openai-community/gpt2-medium

https://huggingface.co/openai-community/gpt2-medium

GPT-2 Medium is the **355M parameter** version of GPT-2. 

gpt2-medium本地存储空间1.5G。

355/124 = 2.863

1.5/0.526 = 2.852

这个参数量的比值和本地存储空间的比值也挺接近。(2024年2月9日)



## openai-community/gpt2-large

https://huggingface.co/openai-community/gpt2-large

GPT-2 Large is the **774M parameter** version of GPT-2. 

gpt2-medium本地存储空间3.1G。(2024年2月9日)



## openai-community/gpt2-xl

https://huggingface.co/openai-community/gpt2-xl

GPT-2 XL is the **1.5B parameter** version of GPT-2. 

gpt2-xl本地存储空间6G。(2024年2月9日)

Comment:  模型越大，推理时间也会越长。(2024年2月9日)



## bigcode/starcoder

https://huggingface.co/bigcode/starcoder

https://www.bilibili.com/video/BV1mm4y1x72Q/?spm_id_from=333.337.search-card.all.click&vd_source=3ef4175721f926fbf390a069da19b0ca

代码预训练模型。

Comment:  原来评测大模型可以用huggingface evaluate库。(2024年2月9日)



### Q: 如果我想fine-tune的一个续写小说的模型，那么我fine-tune的dataset应该是prompt-output这种结构，还是一大段连在一起的文本结构？

你的场景是想用GPT-2微调出适合**小说续写**的模型，这时候你的**数据集**格式建议如下：

#### 1. **续写任务建议用 prompt-output 结构**

- **原因**：  
  你希望模型学会“接着给定开头继续写故事”，这实际上是一个**条件文本生成**（conditional generation）问题。
- **数据结构**：  
  - 每一条样本由`prompt`（故事片段开头，或用户输入的提示）和`output`（后续应生成的部分）组成。
  - CSV文件例子：
    | prompt           | output                |
    | ---------------- | --------------------- |
    | Once upon a time | there was a little... |
    | In a dark forest | a wolf started to...  |

  你现有的预处理和Trainer代码，**天然适配这种结构**，且可以控制只计算output部分的loss。

#### 2. **“一大段连在一起文本”适用场景：无条件生成或者想学语感**

- 这种格式（比如把整本小说长文本直接拼一起）用于“语言建模”任务，让模型泛化出整体写作风格、用语、结构，但模型再生成时**不是根据特定prompt续写**，而是“自由发挥”。
- 这种训练方式类似于GPT/GPT-2预训练，只能让模型生成“自洽的下一句”，但不擅长“读懂开头接着写”。

#### 3. **你的实际需求对比**

- 你希望模型根据“给定开头”生成故事后续，应使用**prompt-output**结构。
- 这样模型学会“遇到prompt就输出对应output”，推理时只输入prompt，即可完成续写任务。

#### 结论

- **建议：用prompt-output数据结构进行微调**
- 你目前的dataset处理和代码已经是最佳实践。

#### **补充说明：**

- 这种格式对想做“对话生成”、“故事接龙”、“摘要”等任务都适用：让模型明确“输入/条件→输出”关系。
- 若以后想让模型不只会续写，还会模仿写作风格，可以先用长文本“无监督预训练”，再用prompt-output进行微调，这样兼顾泛化和任务性。

### Q: 如果我想fine-tune的一个思维链推理的模型，那么我fine-tune的dataset应该是什么样子的？

你想要微调一个**具备“思维链（Chain-of-Thought, CoT）推理能力**”的模型，关键在于**数据集样例需要逐步展示思考过程**，而不是只给出问题和答案。这样模型才能学习到中间推理步骤，在推理任务中更具可解释性和表现力。

### 1. 数据字段设计

和你现在的数据结构相同，可以继续用 `prompt`（输入/问题） 和 `output`（模型要学的完整解答）。**区别在于 output 需要包含推理详细步骤**。

#### CSV 示例
| prompt                                          | output                                                       |
| ----------------------------------------------- | ------------------------------------------------------------ |
| 5个人吃10个苹果，每个人吃多少？                 | 首先，10个苹果要分给5个人。10除以5等于2。所以每个人能分到2个苹果。答案：2。 |
| 桌上有12支笔，分给4个同学，每人能分到几支？     | 总共有12支笔，要分给4个人。用12除以4，得到3。所以每人能分到3支。答案：3。 |
| 一列火车有8节车厢，每节有12个人，一共有多少人？ | 火车有8节车厢，每节有12个人。先算8乘以12等于96，所以一共有96个人。答案：96。 |

#### 英文 Chain-of-Thought Example
| prompt                                                       | output                                                       |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| If Lily has 3 bags, each with 4 apples, how many apples does she have in total? | Lily has 3 bags. Each bag contains 4 apples. So, 3 times 4 is 12. Therefore, Lily has 12 apples. Answer: 12. |

### 2. 格式建议（包含思维链步骤）
1. **推理过程明确分步**，可以使用“首先…然后…所以…”等语言指引
2. **最后再写结论**，如“答案：12”或“Final answer: 12”

#### 推荐编写方式（中英皆可）
```csv
prompt,output
小明有12根棒棒糖，他要均分给3个朋友，每人能分到几根？,"小明一共有12根棒棒糖。需要把12根棒棒糖分给3个朋友。用12除以3，结果是4。所以每人能分到4根棒棒糖。答案：4。"
Mary has 4 boxes. Each box has 7 candies. How many candies in total?,"There are 4 boxes. Each box has 7 candies. 4 times 7 equals 28. Therefore, Mary has 28 candies. Answer: 28."
```

### 3. 数据集规模建议
- CoT能力对样本细致度要求较高，建议**每条样本都详细罗列步骤**
- 数据量至少几千条更好（但哪怕几百条也可以起步微调）

### 4. 代码准备实现
只需要保证 data.csv 的 `output` 字段已经按上面格式编写，**你现有的微调代码可直接用**，无需修改主流程。

### 5. 进阶建议
- 题目类型和难度可多样，但保持每个 `output` 都适合一步步推理
- 也可以人工造一些错例，再反例强化

---

**总结：**
你的 fine-tune 数据集应该保证：
- `prompt` 给出完整问题
- `output` 包含详细的**推理步骤+最终答案**

这样的数据才能训练出“会讲思维链”的 GPT2 微调模型。

如需英文、高质量公开数据样例推荐，或自动化批量制备脚本，可随时告知！
