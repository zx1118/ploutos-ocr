# 版式理解模型集成指南

本文档介绍如何集成 **PaddleStructure** 或 **DocLayout-YOLO** 来提升 OCR 模块的版式分析能力。

## 方案对比

| 特性 | PaddleStructure | DocLayout-YOLO |
|------|-----------------|----------------|
| **准确率** | 高（表格结构识别强） | 高（布局检测快） |
| **速度** | 中等 (~1-2s/页) | 快 (~0.1s/页) |
| **GPU 需求** | 可选 | 推荐 |
| **表格提取** | ✅ 原生 HTML 输出 | 需配合 OCR |
| **安装复杂度** | 低 | 中等 |
| **模型大小** | ~1.5GB | ~300MB |

**推荐**：优先使用 **PaddleStructure**（与现有 PaddleOCR 兼容性好，表格识别强）

---

## 方案 A：PaddleStructure 安装

### 1. 安装依赖

```bash
# CPU 版本
pip install paddlepaddle paddleocr

# GPU 版本（CUDA 11.x）
pip install paddlepaddle-gpu paddleocr

# GPU 版本（CUDA 12.x）
pip install paddlepaddle-gpu==2.6.0 -f https://www.paddlepaddle.org.cn/whl/linux/cudnnin/stable.html
```

### 2. 验证安装

```python
from paddleocr import PPStructure

engine = PPStructure(use_gpu=False, lang="ch")
result = engine("test_invoice.png")
print(result)
```

### 3. 配置 OCR Worker

编辑 `.env` 或环境变量：

```bash
# 使用 PaddleStructure
OCR_LAYOUT_ENGINE=paddle-structure
OCR_ENABLE_TABLE_STRUCTURE=true

# 如有 GPU
OCR_USE_GPU=true
```

### 4. 功能特性

- **布局检测**：自动识别 text, table, figure, title, list, equation
- **表格结构**：输出 HTML 格式表格，支持合并单元格
- **阅读顺序**：自动确定文档阅读顺序

---

## 方案 B：DocLayout-YOLO 安装

### 1. 安装依赖

```bash
# 方式一：从 pip 安装
pip install doclayout-yolo

# 方式二：从源码安装
git clone https://github.com/opendatalab/DocLayout-YOLO
cd DocLayout-YOLO
pip install -e .
```

### 2. 下载模型

从 [GitHub Releases](https://github.com/opendatalab/DocLayout-YOLO/releases) 下载模型：

```bash
# 创建模型目录
mkdir -p models

# 下载 DocStructBench 模型（推荐）
wget -O models/doclayout_yolo_docstructbench.pt \
  https://github.com/opendatalab/DocLayout-YOLO/releases/download/v0.0.1/doclayout_yolo_docstructbench.pt
```

### 3. 验证安装

```python
from doclayout_yolo import YOLOv10

model = YOLOv10("models/doclayout_yolo_docstructbench.pt")
result = model("test_invoice.png")
print(result)
```

### 4. 配置 OCR Worker

```bash
# 使用 DocLayout-YOLO
OCR_LAYOUT_ENGINE=doclayout-yolo
OCR_DOCLAYOUT_MODEL_PATH=models/doclayout_yolo_docstructbench.pt

# 使用 GPU（推荐）
OCR_USE_GPU=true
```

### 5. 功能特性

- **快速检测**：基于 YOLO 架构，GPU 上可达 10+ FPS
- **多类检测**：title, text, figure, table, list, equation 等
- **轻量模型**：~300MB

---

## 自动选择模式

默认配置为 `auto`，系统会自动检测并使用最佳可用引擎：

```bash
OCR_LAYOUT_ENGINE=auto  # 默认
```

优先级：
1. PaddleStructure（如已安装）
2. DocLayout-YOLO（如模型存在）
3. Rule-based（始终可用）

---

## API 使用示例

### Python 代码

```python
from ocr_worker.ocr.layout_analyzer import (
    analyze_document_layout,
    LayoutEngine,
    get_available_engines,
)

# 查看可用引擎
print(get_available_engines())
# ['rule-based', 'paddle-structure', 'doclayout-yolo']

# 使用 PaddleStructure
result = analyze_document_layout(
    image_path="invoice.png",
    engine=LayoutEngine.PADDLE_STRUCTURE,
    use_gpu=True,
)

# 获取表格区域
tables = result.get_table_regions()
for table in tables:
    print(f"Table at {table.bbox}, confidence: {table.confidence}")
    print(f"HTML: {table.table_structure}")

# 使用 DocLayout-YOLO（更快）
result = analyze_document_layout(
    image_path="invoice.png",
    engine=LayoutEngine.DOCLAYOUT_YOLO,
    use_gpu=True,
)
```

### 直接使用引擎

```python
# PaddleStructure
from ocr_worker.ocr.paddle_structure import PaddleStructureEngine

engine = PaddleStructureEngine(use_gpu=True, lang="ch")
result = engine.analyze("invoice.png")
tables = engine.extract_tables("invoice.png")

# DocLayout-YOLO
from ocr_worker.ocr.doclayout_yolo import DocLayoutYOLOEngine

engine = DocLayoutYOLOEngine(device="cuda")
result = engine.analyze("invoice.png")
tables = engine.detect_tables("invoice.png")
```

---

## 性能优化建议

### GPU 加速

```bash
# PaddleStructure GPU
pip install paddlepaddle-gpu
export OCR_USE_GPU=true

# DocLayout-YOLO GPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
export OCR_USE_GPU=true
```

### 批量处理

```python
# PaddleStructure 支持批量
engine = PaddleStructureEngine(use_gpu=True)
for image_path in image_paths:
    result = engine.analyze(image_path)
```

### 模型预热

```python
# 首次调用会加载模型，后续调用更快
engine = PaddleStructureEngine(use_gpu=True)
engine._init_engine()  # 预热
```

---

## 常见问题

### Q: PaddleStructure 安装失败？

```bash
# 确保 Python 版本
python --version  # 推荐 3.8-3.11

# 使用清华源
pip install paddleocr -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Q: DocLayout-YOLO 模型加载慢？

首次加载需下载模型权重，后续会使用缓存。

### Q: GPU 不可用？

```python
# 检查 PaddlePaddle GPU
import paddle
print(paddle.is_compiled_with_cuda())  # True
print(paddle.device.cuda.device_count())  # >= 1

# 检查 PyTorch GPU
import torch
print(torch.cuda.is_available())  # True
```

---

## 参考链接

- [PaddleStructure 文档](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/ppstructure/README.md)
- [DocLayout-YOLO GitHub](https://github.com/opendatalab/DocLayout-YOLO)
- [MinerU 项目](https://github.com/opendatalab/MinerU)（使用了 DocLayout-YOLO）
