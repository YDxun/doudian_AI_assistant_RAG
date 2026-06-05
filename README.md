# AI 客服助手 (Biocomma AI Customer Service)

基于 **RAG（检索增强生成）** 架构的多格式文档智能客服系统，支持 PDF、DOCX、XLSX、TXT、MD 等多种文件格式的解析、索引和智能问答。

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                       前端 (React 18)                        │
│  聊天界面 · SSE 流式响应 · Markdown 渲染 · 文件上传/附件     │
└──────────────────────────┬──────────────────────────────────┘
                           │ REST API + SSE
┌──────────────────────────▼──────────────────────────────────┐
│                    后端 (FastAPI)                            │
│  ├─ 文件解析管线    │  ├─ 向量索引管线  │  ├─ RAG 对话管线   │
│  └─ 知识库管理      │  └─ 跨库检索      │  └─ 批量导入       │
└──────┬───────────────┴────────┬──────────┴────────┬─────────┘
       │                        │                   │
  ┌────▼─────┐           ┌──────▼──────┐     ┌──────▼──────┐
  │ MinerU    │           │ FAISS       │     │ DeepSeek    │
  │ 文档解析  │           │ 向量检索     │     │ Chat API    │
  │ (PDF)     │           │ (IP/L2距离)  │     │             │
  └──────────┘           └─────────────┘     └─────────────┘
```

---

## 技术栈

| 层级 | 技术 |
|------|------|
| **后端** | Python 3.10+ · FastAPI · uvicorn |
| **前端** | React 18 · TypeScript · Vite · Tailwind CSS · shadcn/ui |
| **Embedding** | Qwen3-VL-Embedding-2B (本地，last-token-pooling) · FAISS |
| **LLM** | DeepSeek Chat (云端 API) |
| **PDF 处理** | MinerU (pipeline 模式) · PyMuPDF (页面渲染) |
| **DOCX** | python-docx (段落/表格/图片提取) |
| **XLSX** | openpyxl (逐行读取，表头+数据拼装) |
| **OCR (附件)** | PaddleOCR (聊天附件图片识别) |

---

## 支持的文件格式

| 格式 | 解析方式 | 切分策略 |
|------|---------|---------|
| **PDF** | MinerU (pipeline) · 布局分析 · OCR · 表格识别 → Markdown | RecursiveCharacterTextSplitter (800 chars) |
| **DOCX** | python-docx (标题/段落/表格/图片提取) → Markdown | RecursiveCharacterTextSplitter (600 chars) |
| **XLSX** | openpyxl (逐行读取，表头+数据拼装) → Markdown | 每行作为一个 chunk |
| **TXT** | 直接读取 (支持 utf-8/gbk/gb2312) | MarkdownHeaderTextSplitter (按 # / ## 切分) |
| **MD** | 直接读取 | MarkdownHeaderTextSplitter (按 # / ## 切分) |

---

## 快速开始

### 1. 环境准备

- **Python 3.10+**（推荐虚拟环境）
- **Node.js 18+**（前端）
- **MinerU**：PDF 文档解析引擎
- **本地 Embedding 模型**：`Qwen3-VL-Embedding-2B/`
- **DeepSeek API Key**

### 2. MinerU 安装与配置

```powershell
# 进入后端目录
cd backend

# 激活虚拟环境
venv\Scripts\activate

# 安装 MinerU (CPU 模式)
uv pip install -U "mineru[all]"
```

> 以上命令安装的是 CPU 模式。如需 GPU 加速，请参考 [MinerU 官方文档](https://github.com/opendatalab/MinerU)。

#### 下载模型文件

```powershell
# 使用 ModelScope 源下载模型
mineru-models-download -s modelscope -m all
```

> 模型文件约 3-5GB，下载完成后自动生成配置文件 `mineru.json`。

#### 验证安装

```powershell
mineru -p "your_pdf_file.pdf" -o "./output_test" -b pipeline
```

### 3. 后端启动

```powershell
cd backend

# 创建虚拟环境（如未创建）
python -m venv venv
venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 检查 .env 配置（需包含以下变量）：
#   DEEPSEEK_API_KEY=sk-xxx
#   OPENAI_API_KEY=sk-xxx
#   OPENAI_BASE_URL=https://ai.devtool.tech/proxy/v1

# 启动服务
uvicorn app:app --host 0.0.0.0 --port 8001 --reload
```

后端运行在 `http://localhost:8001`。

### 4. 前端启动

```powershell
cd frontend

# 安装依赖
npm install

# 启动开发服务器
npm run dev
```

前端运行在 `http://localhost:3000`，自动连接后端 `http://localhost:8001`。

如需修改后端地址，创建 `frontend/.env.local`：

```env
VITE_API_BASE_URL=http://localhost:8001/api/v1
```

### 5. 验证

```powershell
# 健康检查
curl http://localhost:8001/api/v1/health
# 返回: {"ok": true, "version": "1.0.0", "knowledge_bases": 0}
```

---

## 使用指南

### 方式一：Web 界面单文件上传

1. 在聊天界面点击附件按钮，上传 PDF/DOCX/XLSX/TXT/MD 文件
2. 系统自动进行文件解析 → 索引构建
3. 在聊天框提问，系统检索知识库并生成带引用的回答

### 方式二：统一知识库批量导入（服务端 API）

将文件放入 `backend/data/kb_import/` 目录，然后通过 API 触发批量导入：

```powershell
# 将文件放入导入目录
copy *.pdf backend\data\kb_import\
copy *.docx backend\data\kb_import\

# 通过 API 触发导入
curl -X POST http://localhost:8001/api/v1/unified-kb/import
```

导入完成后，通过聊天接口的 `useUnifiedKB: true` 参数使用统一知识库问答。

### 方式三：独立导入脚本（无需启动后端）

```powershell
# 将文件放入 kb_import 目录
copy *.pdf backend\data\kb_import\

# 运行独立导入脚本
python standalone_kb_importer.py
```

脚本使用 `Qwen3-VL-Embedding-2B` 模型生成多模态向量（文本+图片），直接写入 FAISS 索引文件。

---

## API 接口

### 统一知识库

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/v1/unified-kb/import` | POST | 从 `data/kb_import/` 批量导入文件到统一知识库 |
| `/api/v1/unified-kb/import/status` | GET | 查询导入进度 |
| `/api/v1/unified-kb/search` | POST | 在统一知识库中搜索 |
| `/api/v1/unified-kb/info` | GET | 获取统一知识库信息（文件数、块数等） |

### 单文件知识库

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/v1/knowledge-base/list` | GET | 列出所有已建索引的知识库 |
| `/api/v1/knowledge-base/{file_id}` | DELETE | 删除指定知识库 |
| `/api/v1/knowledge-base/search` | POST | 跨知识库搜索 |

### 文件操作

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/v1/pdf/upload` | POST | 上传文件（PDF/DOCX/XLSX/TXT/MD） |
| `/api/v1/pdf/parse` | POST | 触发文件解析 |
| `/api/v1/pdf/status` | GET | 查询解析状态 |
| `/api/v1/pdf/page` | GET | 获取 PDF 页面图片 |
| `/api/v1/pdf/images` | GET | 获取解析后的图片 |
| `/api/v1/files/upload` | POST | 上传聊天附件（自动 OCR 提取文本） |
| `/api/v1/files/batch-import` | POST | 从 `data/batch_import/` 批量导入 |

### 索引

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/v1/index/build` | POST | 为指定文件构建 FAISS 向量索引 |
| `/api/v1/index/search` | POST | 搜索指定文件索引 |

### 对话

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/v1/chat` | POST | SSE 流式问答（支持统一知识库/单文件/跨库检索） |
| `/api/v1/chat/clear` | POST | 清除对话历史 |

### 健康检查

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/v1/health` | GET | 服务状态 + 知识库数量 |

---

## 核心流程

### 文件解析管线

```
上传文件 → save_upload() → run_full_parse_pipeline()
                                │
            ┌───────────────────┼───────────────────┐
            │                   │                   │
       [PDF文件]           [DOCX文件]           [XLSX文件]
            │                   │                   │
      MinerU pipeline     python-docx           openpyxl
      布局分析+OCR        标题/段落/表格        表头+数据拼装
      表格识别+图片提取    图片提取
            │                   │                   │
            └───────────────────┼───────────────────┘
                                │
                          output.md
```

### RAG 问答管线

```
用户提问
    │
    ├── useUnifiedKB=true → 检索统一知识库 (FAISS IP)
    ├── pdfFileId 指定     → 检索单个文件索引 (FAISS L2)
    ├── knowledgeBaseIds   → 跨库检索
    └── 无指定             → 优先统一知识库，否则跨库检索
    │
    ├── Qwen3-VL-Embedding-2B 编码查询
    ├── FAISS 相似度检索 Top-K
    ├── 评分判定 (L2: top1 ≤ 0.45 或 mean3 ≤ 0.60)
    ├── LLM Grader 复核相关性（评分不达标时）
    │
    ├── with_context → 组装上下文 → DeepSeek Chat 生成
    └── no_context   → 直接调用 DeepSeek Chat + system prompt
    │
    └── SSE 流式返回: citation → token → done
```

### 统一知识库批量导入管线

```
data/kb_import/ 目录
    │
    ├── 扫描所有文件
    ├── 逐个处理：
    │   ├── 检查文件哈希 → 是否已导入？
    │   ├── 保存文件 → MinerU/DOCX/XLSX 解析 → output.md
    │   ├── 文本分块（根据文件类型选择策略）
    │   ├── MD5 块级去重
    │   ├── 向量化 → 添加到统一 FAISS 索引
    │   └── 导入成功 → 移动到 archive/
    │
    └── 实时进度: {status, total, processed, success, failed, current_file}
```

---

## 目录结构

```
MRAG/
├── backend/
│   ├── app.py                       # FastAPI 主入口
│   ├── .env                         # API Key 配置
│   ├── requirements.txt             # Python 依赖
│   ├── services/
│   │   ├── file_service.py          # 多格式文件解析 (PDF/DOCX/XLSX/TXT/MD)
│   │   ├── index_service.py         # FAISS 索引构建与检索（含统一知识库）
│   │   └── rag_service.py           # RAG 对话 + OCR + 多库检索 + LLM 生成
│   └── data/
│       ├── {fileId}/                # 单文件工作目录
│       │   ├── original.{ext}       # 原始文件
│       │   ├── output.md            # 解析后的 Markdown
│       │   ├── index_faiss/         # FAISS 向量索引
│       │   ├── pages/original/      # 原图 PNG
│       │   ├── pages/parsed/        # 叠框可视化图
│       │   └── images/              # 提取的图片
│       ├── kb_import/               # 统一知识库导入目录
│       │   └── archive/             # 已导入文件归档
│       ├── unified_knowledge_base/  # 统一知识库索引
│       │   ├── index.faiss          # FAISS 索引文件
│       │   └── metadata.json        # 文档元数据
│       └── chat_uploads/            # 聊天附件临时目录
├── frontend/
│   ├── src/
│   │   ├── App.tsx                  # 主布局
│   │   ├── components/
│   │   │   ├── ChatInterface.tsx    # 聊天界面（SSE 流式 + 文件上传）
│   │   │   ├── Header.tsx           # 顶部导航
│   │   │   ├── MarkdownRenderer.tsx # Markdown 渲染
│   │   │   └── ui/                  # shadcn/ui 组件
│   │   └── services/api.ts         # API 服务层
│   ├── package.json
│   └── vite.config.ts
├── Qwen3-VL-Embedding-2B/           # 本地 Embedding 模型
├── tesseract_ydx/                   # Tesseract OCR 引擎
└── standalone_kb_importer.py        # 独立知识库导入脚本
```

---

## 模型说明

| 模型 | 运行位置 | 作用 |
|------|---------|------|
| **Qwen3-VL-Embedding-2B** | 本地 | 文本向量化 (last-token-pooling, IP 距离) |
| **DeepSeek Chat** | 云端 API | 对话生成 + 相关性判定 |
| **MinerU 模型** | 本地 | PDF 文档解析（布局分析、OCR、表格识别） |
| **PaddleOCR** | 本地 | 聊天附件图片 OCR 识别 |

> Qwen3-VL-Embedding-2B 是多模态模型，`standalone_kb_importer.py` 使用其图片+文本联合向量化能力。后端 API 服务仅使用其文本向量化能力。

---

## 配置说明

### .env 文件 (`backend/.env`)

```env
# DeepSeek LLM API Key
DEEPSEEK_API_KEY=sk-xxx

# OpenAI 兼容 API（用于代理）
OPENAI_API_KEY=sk-xxx
OPENAI_BASE_URL=https://ai.devtool.tech/proxy/v1
```

### 前端环境变量 (`frontend/.env.local`)

```env
VITE_API_BASE_URL=http://localhost:8001/api/v1
```

---

## 常见问题

### Q: 后端启动报错 "ModuleNotFoundError"

确认已激活虚拟环境并执行 `pip install -r requirements.txt`。

### Q: PDF 解析很慢

MinerU 首次运行会自动加载模型，后续会缓存。使用 CPU 时每页约需 5-15 秒，GPU 可显著加速。

### Q: 索引构建很慢

Qwen3-VL-Embedding-2B 模型首次加载需要初始化时间，有 GPU 会更快。

### Q: 如何查看知识库列表

访问 `http://localhost:8001/api/v1/knowledge-base/list` 或调用健康检查接口。

### Q: 如何删除知识库

`DELETE /api/v1/knowledge-base/{file_id}` 或直接删除 `data/{file_id}/` 目录。
