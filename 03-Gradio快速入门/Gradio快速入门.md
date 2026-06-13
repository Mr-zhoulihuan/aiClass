# Gradio 快速入门

Gradio 是一个 Python 库，让你**几行代码就能为机器学习模型创建 Web 交互界面**。无需前端知识，即可快速构建和分享 AI 演示应用。

---

## 一、安装

```bash
pip install gradio
```

验证安装：

```python
import gradio as gr
print(gr.__version__)
```

---

## 二、第一个应用：Hello World

```python
import gradio as gr

def greet(name):
    return f"你好，{name}！"

demo = gr.Interface(fn=greet, inputs="text", outputs="text")
demo.launch()
```

运行后访问 `http://127.0.0.1:7860`，即可看到输入框和输出框。

---

## 三、核心组件

### 3.1 Interface（最常用）

`gr.Interface` 是 Gradio 最简洁的封装，适合快速原型：

```python
gr.Interface(
    fn=your_function,     # 处理函数
    inputs=...,           # 输入组件
    outputs=...,          # 输出组件
    title="标题",
    description="描述",
    examples=[["示例1"], ["示例2"]],  # 示例输入
).launch()
```

### 3.2 常用输入/输出组件

| 组件 | 说明 | 示例 |
|------|------|------|
| `gr.Textbox()` | 文本输入/输出 | 问答、翻译 |
| `gr.Number()` | 数字输入 | 计算器参数 |
| `gr.Slider()` | 滑动条 | 调节参数 |
| `gr.Image()` | 图像输入/输出 | 图片分类、生成 |
| `gr.Audio()` | 音频输入/输出 | 语音识别、合成 |
| `gr.Video()` | 视频输入/输出 | 视频处理 |
| `gr.File()` | 文件上传/下载 | 批量处理 |
| `gr.Dataframe()` | 表格数据 | 数据分析 |
| `gr.Dropdown()` | 下拉选择 | 选项配置 |
| `gr.Radio()` | 单选按钮 | 模式选择 |
| `gr.Checkbox()` | 复选框 | 开关选项 |

### 3.3 Blocks（灵活布局）

需要复杂布局时用 `gr.Blocks`，比 Interface 更灵活：

```python
import gradio as gr

def image_classify(img):
    return f"处理完成，图片尺寸：{img.shape}"

with gr.Blocks(title="图片处理工具") as demo:
    gr.Markdown("# 图片处理演示")

    with gr.Row():
        with gr.Column():
            img_input = gr.Image(label="上传图片")
            btn = gr.Button("开始处理")

        with gr.Column():
            output = gr.Textbox(label="处理结果")

    btn.click(fn=image_classify, inputs=img_input, outputs=output)

demo.launch()
```

---

## 四、完整示例：图像分类演示

```python
import gradio as gr
import torch
from PIL import Image
import requests

# 使用预训练模型（需安装 torchvision）
model = torch.hub.load("pytorch/vision:v0.10.0", "resnet18", pretrained=True)
model.eval()

# 加载 ImageNet 类别标签
LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
labels = requests.get(LABELS_URL).text.strip().split("\n")

def classify_image(img):
    from torchvision import transforms
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = outputs.max(1)
    return f"预测结果：{labels[predicted.item()]}"

demo = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="图像分类演示（ResNet18）",
    description="上传一张图片，模型将预测它属于哪个类别。",
    examples=[["cat.jpg"]],  # 需准备示例图片
)
demo.launch()
```

---

## 五、与 LLM 结合

### 5.1 调用 OpenAI API 的聊天界面

```python
import gradio as gr
from openai import OpenAI

client = OpenAI(api_key="your-api-key")

def chat(message, history):
    messages = [{"role": "system", "content": "你是一个有用的助手。"}]
    for user_msg, bot_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": bot_msg})
    messages.append({"role": "user", "content": message})

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    return response.choices[0].message.content

demo = gr.ChatInterface(
    fn=chat,
    title="AI 聊天机器人",
    description="基于 OpenAI GPT-4o 的聊天演示",
)
demo.launch()
```

### 5.2 流式输出（打字机效果）

```python
import gradio as gr
from openai import OpenAI

client = OpenAI(api_key="your-api-key")

def stream_chat(message, history):
    messages = [{"role": "system", "content": "你是一个有用的助手。"}]
    for user_msg, bot_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": bot_msg})
    messages.append({"role": "user", "content": message})

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        stream=True
    )

    partial = ""
    for chunk in response:
        if chunk.choices[0].delta.content:
            partial += chunk.choices[0].delta.content
            yield partial  # 使用 yield 实现流式输出

demo = gr.ChatInterface(fn=stream_chat, title="流式 AI 聊天")
demo.launch()
```

> **流式输出**需要函数用 `yield` 替代 `return`，Gradio 会自动处理流式展示。

### 5.3 调用本地模型（Ollama）

```python
import gradio as gr
import requests
import json

def chat_ollama(message, history):
    messages = [{"role": "system", "content": "你是一个有用的助手。"}]
    for user_msg, bot_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": bot_msg})
    messages.append({"role": "user", "content": message})

    response = requests.post(
        "http://localhost:11434/api/chat",
        json={"model": "qwen2.5:7b", "messages": messages, "stream": True},
        stream=True
    )

    partial = ""
    for line in response.iter_lines():
        if line:
            data = json.loads(line)
            if data.get("done"):
                break
            partial += data["message"]["content"]
            yield partial

demo = gr.ChatInterface(fn=chat_ollama, title="本地 AI 聊天（Ollama）")
demo.launch()
```

---

## 六、高级功能

### 6.1 标签页（Tab）

```python
with gr.Blocks() as demo:
    with gr.Tab("图像处理"):
        gr.Markdown("上传图片进行处理")
        # 图像处理组件

    with gr.Tab("文本处理"):
        gr.Markdown("输入文本进行处理")
        # 文本处理组件
```

### 6.2 状态管理（State）

Gradio 用 `gr.State()` 维护用户会话级别的状态：

```python
def chat_with_memory(message, history):
    # history 由 Gradio 自动维护
    return f"历史共 {len(history)} 轮对话。你说：{message}"

demo = gr.ChatInterface(fn=chat_with_memory)
```

### 6.3 进度条

```python
import time

def slow_process():
    for i in range(10):
        time.sleep(0.5)
        yield i / 10  # 0.0 ~ 1.0

gr.Interface(fn=slow_process, inputs=None, outputs=gr.Textbox()).launch()
```

### 6.4 认证与分享

```python
# 设置用户名密码
demo.launch(auth=("admin", "password123"))

# 生成公网分享链接（需安装 pyngrok）
demo.launch(share=True)
```

**share=True** 会生成一个可公开访问的临时链接（通过 Gradio 的代理服务器），有效期 72 小时。

### 6.5 队列与并发

```python
demo.queue(max_size=10)  # 限制队列长度，防止过载
demo.launch()
```

---

## 七、部署方式

| 方式 | 说明 | 适用场景 |
|------|------|----------|
| **Gradio Cloud** | 官方托管，`demo.launch(share=True)` | 快速分享演示 |
| **Hugging Face Spaces** | 免费托管 Gradio 应用 | 社区分享、开源项目 |
| **自建服务器** | 用 `demo.launch(server_name="0.0.0.0")` | 企业内部部署 |
| **Docker 部署** | 容器化部署，环境隔离 | 生产环境 |
| **Kubernetes** | 容器编排，自动伸缩 | 高并发场景 |

### 部署到 Hugging Face Spaces

1. 登录 [huggingface.co](https://huggingface.co)
2. 创建 New Space，选择 **Gradio** SDK
3. 将代码上传为 `app.py`，添加 `requirements.txt`
4. Spaces 自动构建并运行

---

## 八、常用技巧与 FAQ

### 8.1 安装问题

**Q：运行后无法访问 `localhost:7860`？**
A：尝试 `demo.launch(server_port=7861)` 换个端口，或检查防火墙。

**Q：`share=True` 报错？**
A：需要安装 `pip install pyngrok`。

### 8.2 性能优化

```python
# 启用队列和并发
demo.queue(max_size=20, concurrency_count=5)

# 控制批处理大小
demo.queue(batch_size=8)

# 设置超时（秒）
demo.queue(default_concurrency_limit=5)
```

### 8.3 样式定制

```python
# 使用自定义 CSS
demo = gr.Interface(..., css="footer {display: none !important;}")

# 设置主题
demo = gr.Interface(..., theme="soft")  # 可选：default、soft、monochrome
```

### 8.4 常用快捷键

- `Ctrl + Enter`：提交输入
- `Clear` 按钮：清除当前输入
- `Flag` 按钮：标记特定结果（调试用）

---

## 九、完整项目模板

```python
import gradio as gr

def process_text(text, option, use_advanced):
    if use_advanced:
        result = f"[高级模式] 处理结果：{text}（选项：{option}）"
    else:
        result = f"处理结果：{text}（选项：{option}）"
    return result

with gr.Blocks(title="AI 工具箱", theme="soft") as demo:
    gr.Markdown("# 🤖 AI 工具箱")

    with gr.Tab("文本处理"):
        with gr.Row():
            with gr.Column():
                text_input = gr.Textbox(label="输入文本", placeholder="请输入...")
                option = gr.Dropdown(choices=["翻译", "总结", "改写"], label="操作类型")
                advanced = gr.Checkbox(label="高级模式")
                submit_btn = gr.Button("提交", variant="primary")
            with gr.Column():
                text_output = gr.Textbox(label="输出结果")

        submit_btn.click(
            fn=process_text,
            inputs=[text_input, option, advanced],
            outputs=text_output
        )

    with gr.Tab("关于"):
        gr.Markdown("""
        ## 关于此应用

        这是一个基于 Gradio 构建的 AI 工具箱演示。

        - **框架**：Gradio 5.x
        - **版本**：1.0.0
        """)

if __name__ == "__main__":
    demo.launch(server_port=7860)
```

---

## 十、学习资源

| 资源 | 地址 |
|------|------|
| **官方文档** | [gradio.app/docs](https://gradio.app/docs) |
| **官方示例** | [gradio.app/demos](https://gradio.app/demos) |
| **Hugging Face Spaces** | [huggingface.co/spaces](https://huggingface.co/spaces) |
| **GitHub** | [github.com/gradio-app/gradio](https://github.com/gradio-app/gradio) |

---

*最后更新：2025 年*
