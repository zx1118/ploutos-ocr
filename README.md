# Ploutos OCR Worker

基于 PaddleOCR 的 OCR 服务，使用 Redis Stream 作为任务队列。

## 架构

```
┌─────────────────┐     Redis Stream     ┌─────────────────┐
│   JeecgBoot     │ ──── XADD ────────▶ │   OCR Worker    │
│   (Java)        │ ◀─── HTTP回调 ───── │   (Python)      │
└─────────────────┘                      └─────────────────┘
```

## 组件

- **Redis Client**: 通用 Redis 操作封装
- **Stream Client**: Redis Stream 专用操作（Consumer Group, XREADGROUP, XAUTOCLAIM）
- **OCR Engine**: PaddleOCR 封装，支持文本和结构化识别
- **Post Processor**: 规则后处理引擎
- **Consumer**: Redis Stream 消费者
- **API**: FastAPI 服务（可选）

## 安装

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

## 配置

复制 `env.template` 为 `.env` 并修改配置：

```bash
cp env.template .env
```

主要配置项：

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| REDIS_HOST | Redis 主机 | 127.0.0.1 |
| REDIS_PORT | Redis 端口 | 6379 |
| OCR_STREAM_KEY | 任务队列 Key | ocr:tasks |
| OCR_DLQ_KEY | 死信队列 Key | ocr:tasks:dlq |
| OCR_USE_GPU | 是否使用 GPU | false |
| JEECG_BASE_URL | JeecgBoot 地址 | http://localhost:8080/jeecg-boot |

## 运行

### 运行 Worker（消费者）

```bash
python -m ocr_worker worker
```

### 运行 API 服务

```bash
python -m ocr_worker api --port 8100
```

### 同时运行 Worker 和 API

```bash
python -m ocr_worker both
```

## API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/v1/ocr/text` | POST | 文本 OCR |
| `/v1/ocr/structure` | POST | 结构化 OCR |
| `/v1/ocr/upload` | POST | 上传文件 OCR |
| `/admin/stats` | GET | 队列统计 |
| `/admin/dlq` | GET | 死信队列 |
| `/admin/rules` | GET | 规则列表 |

## 目录结构

```
ploutos-ocr/
├── ocr_worker/
│   ├── __init__.py
│   ├── config.py           # 配置管理
│   ├── main.py             # 入口
│   ├── consumer.py         # Stream 消费者
│   ├── callback.py         # HTTP 回调
│   ├── logging.py          # 日志配置
│   ├── redis/              # Redis 封装
│   │   ├── __init__.py
│   │   ├── pool.py         # 连接池
│   │   ├── client.py       # 通用客户端
│   │   └── stream.py       # Stream 客户端
│   ├── ocr/                # OCR 引擎
│   │   ├── __init__.py
│   │   ├── engine.py       # PaddleOCR 封装
│   │   ├── result.py       # 结果数据结构
│   │   └── structure.py    # 结构提取
│   ├── processor/          # 后处理
│   │   ├── __init__.py
│   │   ├── rule.py         # 规则定义
│   │   ├── engine.py       # 规则引擎
│   │   └── processor.py    # 处理器
│   └── api/                # FastAPI
│       ├── __init__.py
│       └── main.py         # API 服务
├── requirements.txt
├── pyproject.toml
├── env.template
└── README.md
```

## Docker 部署

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY . .

# 运行 Worker
CMD ["python", "-m", "ocr_worker", "worker"]
```

## License

MIT

