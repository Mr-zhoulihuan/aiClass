# Hugging Face 入门指南

Hugging Face 是当前**全球最大的机器学习社区和平台**，集模型托管、数据集、AI 应用部署、开源库于一体，被称为"机器学习的 GitHub"。

---

## 一、平台概述

### 1.1 核心服务

| 服务 | 说明 | 网址 |
|------|------|------|
| **Model Hub**（模型库） | 托管 100 万+ 开源模型，涵盖 NLP、CV、音频、多模态 | huggingface.co/models |
| **Datasets**（数据集） | 托管 5 万+ 公开数据集，与 Hugging Face 库直接集成 | huggingface.co/datasets |
| **Spaces**（应用空间） | 免费部署 Gradio/Streamlit/静态 AI 应用 | huggingface.co/spaces |
| **Docs**（文档） | 各库的官方文档、教程、API 参考 | huggingface.co/docs |

### 1.2 为什么重要？

- **开源核心库**：Transformers、Diffusers、Datasets、PEFT、TRL 等
- **跨框架支持**：兼容 PyTorch、TensorFlow、JAX
- **预训练模型即用**：几行代码加载 BERT、GPT、LLaMA、Stable Diffusion 等模型
- **社区驱动**：500 万+ 注册用户，企业用户包括 Google、Meta、Microsoft
- **企业级服务**：提供 Inference API、AutoTrain、SafeCoder 等商业产品

---

## 二、核心库详解

### 2.1 Transformers（核心库）

安装：
```bash
pip install transformers
```

**几行代码加载预训练模型：**

```python
from transformers import pipeline

# 情感分析
classifier = pipeline("sentiment-analysis")
result = classifier("Hugging Face 太棒了！")
print(result)
# [{'label': 'POSITIVE', 'score': 0.999}]

# 文本生成
generator = pipeline("text-generation", model="gpt2")
result = generator("AI 的未来是", max_length=50)
print(result[0]["generated_text"])

# 图像分类
classifier = pipeline("image-classification", model="google/vit-base-patch16-224")
result = classifier("cat.jpg")
print(result)
```

| 常用 Pipeline | 说明 |
|---------------|------|
| `"sentiment-analysis"` | 情感分析 |
| `"text-generation"` | 文本生成 |
| `"fill-mask"` | 掩码填充（BERT 类模型） |
| `"ner"` | 命名实体识别 |
| `"question-answering"` | 问答 |
| `"summarization"` | 文本摘要 |
| `"translation"` | 翻译 |
| `"image-classification"` | 图像分类 |
| `"text-to-speech"` | 语音合成 |
| `"zero-shot-classification"` | 零样本分类 |

### 2.2 Datasets（数据集库）

```bash
pip install datasets
```

```python
from datasets import load_dataset

# 加载数据集（自动下载缓存）
dataset = load_dataset("imdb", split="train")
print(dataset[0])
# {'text': '...', 'label': 0}

# 数据处理
dataset = dataset.filter(lambda x: x["label"] == 1)  # 筛选
dataset = dataset.map(lambda x: {"length": len(x["text"])})  # 增加列
dataset = dataset.train_test_split(test_size=0.2)  # 分割

# Tokenize
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
dataset = dataset.map(lambda x: tokenizer(x["text"], truncation=True, padding=True), batched=True)
```

### 2.3 Diffusers（扩散模型库）

```bash
pip install diffusers
```

```python
from diffusers import StableDiffusionPipeline
import torch

# 文生图
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

image = pipe("一只可爱的猫，在阳光下看书", num_inference_steps=50).images[0]
image.save("cat_reading.png")
```

### 2.4 其他重要库

| 库 | 说明 | 安装 |
|----|------|------|
| **PEFT** | 高效微调方法（LoRA、Prefix Tuning 等） | `pip install peft` |
| **TRL** | Transformer 强化学习（RLHF、DPO） | `pip install trl` |
| **Tokenizers** | 高性能 Tokenizer | `pip install tokenizers` |
| **Accelerate** | 分布式训练简化 | `pip install accelerate` |
| **Optimum** | 硬件加速优化（ONNX、OpenVINO） | `pip install optimum` |
| **Gradio** | 快速构建 ML 演示界面 | `pip install gradio` |

---

## 三、模型使用详解

### 3.1 查找模型

访问 [huggingface.co/models](https://huggingface.co/models) 按以下维度筛选：

- **任务**（NLP、CV、Audio、Multimodal）
- **框架**（PyTorch、TensorFlow、ONNX）
- **语言**（中文、英文、多语言）
- **许可证**（MIT、Apache、LLaMA、CC）
- **流行度**（下载量、点赞数）

### 3.2 加载方式

```python
# 方式一：使用 AutoClass（推荐）
from transformers import AutoModel, AutoTokenizer

model_name = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

# 方式二：使用 pipeline（最简单）
from transformers import pipeline
pipe = pipeline("text-generation", model=model_name)

# 方式三：使用特定模型类
from transformers import BertModel, BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
model = BertModel.from_pretrained("bert-base-chinese")
```

### 3.3 模型配置文件与下载路径

下载的模型默认缓存到：

- **Windows**：`C:\Users\<用户名>\.cache\huggingface\hub`
- **Linux/macOS**：`~/.cache/huggingface/hub`

可通过环境变量更改：
```bash
export HF_HOME=/path/to/cache  # Linux/macOS
# 或 $env:HF_HOME="D:\cache"  # Windows PowerShell
```

---

## 四、模型微调（Fine-tuning）

### 4.1 完整微调流程

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

# 1. 加载数据
dataset = load_dataset("imdb", split="train[:1000]")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding=True)

dataset = dataset.map(tokenize, batched=True)

# 2. 加载模型
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=2
)

# 3. 配置训练参数
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    logging_dir="./logs",
)

# 4. 训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()

# 5. 保存模型
model.save_pretrained("./my-finetuned-model")
tokenizer.save_pretrained("./my-finetuned-model")
```

### 4.2 LoRA 高效微调（PEFT）

```bash
pip install peft bitsandbytes
```

```python
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained("bigscience/mt0-small")

lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=8,           # LoRA 秩
    lora_alpha=32, # 缩放参数
    lora_dropout=0.1,
    target_modules=["q", "v"],  # 目标模块
)

model = get_peft_model(model, lora_config)
print(f"可训练参数：{model.print_trainable_parameters()}")
# 通常只训练原始参数的 0.1%~1%
```

---

## 五、Hugging Face Spaces 部署

### 5.1 快速部署

1. 登录 [huggingface.co](https://huggingface.co) → New Space
2. 选择 SDK：
   - **Gradio** — Python 交互式应用
   - **Streamlit** — Python 数据应用
   - **Docker** — 自定义容器
   - **Static** — HTML/CSS/JS
3. 选择空间可见性：Public / Private
4. 上传代码或关联 GitHub 仓库

### 5.2 配置文件

`requirements.txt`：
```
gradio
transformers
torch
```

`app.py` 示例（情感分析演示）：
```python
import gradio as gr
from transformers import pipeline

classifier = pipeline("sentiment-analysis")

def classify(text):
    result = classifier(text)
    return f"{result[0]['label']}（置信度：{result[0]['score']:.2%}）"

demo = gr.Interface(
    fn=classify,
    inputs=gr.Textbox(label="输入文本", placeholder="输入一句英文..."),
    outputs=gr.Textbox(label="分析结果"),
    title="情感分析演示",
)
demo.launch()
```

### 5.3 使用 GPU

在 Spaces 设置中启用 GPU：
```
Settings → Space Hardware → T4 Small / T4 Medium / A10G
```

> 免费账户有每月 GPU 使用时长限制，超额后需升级或等待下月重置。

---

## 六、常用命令行工具（huggingface-cli）

```bash
# 登录
huggingface-cli login

# 下载模型
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir ./model

# 上传模型
huggingface-cli upload my-username/my-model ./model

# 查看缓存
huggingface-cli scan-cache

# 删除缓存
huggingface-cli delete-cache
```

---

## 七、国内镜像加速

由于网络原因，国内访问 Hugging Face 可能较慢，推荐使用镜像站：

```python
import os

# 方式一：设置环境变量
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 方式二：命令行设置
# export HF_ENDPOINT=https://hf-mirror.com  # Linux/macOS
# $env:HF_ENDPOINT="https://hf-mirror.com"  # Windows PowerShell

# 方式三：下载时指定镜像
from huggingface_hub import snapshot_download
snapshot_download(repo_id="Qwen/Qwen2.5-7B", endpoint="https://hf-mirror.com")
```

国内常用镜像站：

| 镜像站 | 地址 |
|--------|------|
| **HF Mirror** | https://hf-mirror.com |
| **阿里云** | https://hf-mirror.com |
| **ModelScope** | https://modelscope.cn（阿里，国内最全替代） |

---

## 八、常用模型推荐

### 8.1 NLP 类

| 模型 | 参数量 | 说明 |
|------|--------|------|
| `bert-base-chinese` | 110M | 中文 BERT，经典的编码器模型 |
| `Qwen/Qwen2.5-7B-Instruct` | 7B | 通义千问，中文对话首选 |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | 7B | DeepSeek 推理模型 |
| `meta-llama/Llama-3.1-8B-Instruct` | 8B | Meta 开源最强对话模型 |
| `THUDM/chatglm3-6b` | 6B | 智谱 ChatGLM，中文对话优化 |

### 8.2 多模态类

| 模型 | 说明 |
|------|------|
| `Qwen/Qwen2.5-VL-7B-Instruct` | 通义千问多模态模型 |
| `openai/clip-vit-base-patch32` | 图文匹配模型 |
| `llava-hf/llava-v1.6-mistral-7b-hf` | LLaVA 多模态对话 |

### 8.3 图像类

| 模型 | 说明 |
|------|------|
| `runwayml/stable-diffusion-v1-5` | 文生图经典模型 |
| `stabilityai/stable-diffusion-xl-base-1.0` | SDXL，高质量图像生成 |
| `black-forest-labs/FLUX.1-dev` | Flux，当前最强开源文生图 |

### 8.4 语音类

| 模型 | 说明 |
|------|------|
| `openai/whisper-large-v3` | 语音识别（多语言） |
| `suno/bark` | 文本转语音 |

### 8.5 中文专用

| 模型 | 说明 |
|------|------|
| `Qwen/Qwen2.5-7B-Instruct` | 当前最强中文开源模型之一 |
| `THUDM/chatglm3-6b` | 智谱对话模型 |
| `BAAI/bge-large-zh-v1.5` | 中文向量嵌入模型（RAG 常用） |
| `shibing624/text2vec-base-chinese` | 中文语义相似度 |

---

## 九、与 ModelScope 对比

| 对比维度 | Hugging Face | ModelScope（魔搭社区） |
|----------|-------------|----------------------|
| **运营方** | Hugging Face（美国） | 阿里云（中国） |
| **模型数量** | 100 万+ | 1 万+ |
| **下载速度（国内）** | 慢（需镜像加速） | **快** |
| **社区活跃度** | 全球最大 | 国内领先，快速增长 |
| **生态工具** | Transformers、Diffusers 等 | ModelScope Library |
| **Spaces 替代** | Hugging Face Spaces | 魔搭 Notebook + 应用部署 |
| **免费 GPU** | 有限（Spaces） | 有免费算力资源 |

> **建议**：国际项目优先 Hugging Face，国内项目可同时使用 ModelScope 加速下载。

---

## 十、常见问题

| 问题 | 解决方案 |
|------|----------|
| **下载模型速度极慢** | 使用国内镜像：`HF_ENDPOINT=https://hf-mirror.com` |
| **磁盘空间不足** | 清理缓存：`huggingface-cli delete-cache` |
| **GPU 内存不足（OOM）** | 使用 `model.half()` 或加载量化版本（4bit/8bit） |
| **模型加载报错** | 检查 `trust_remote_code=True` 参数 |
| **登录失败** | 使用 `huggingface-cli login --token hf_xxxxxxxxx` |
| **Spaces 部署失败** | 检查 `requirements.txt` 是否完整，查看构建日志 |

---

*最后更新：2025 年*
