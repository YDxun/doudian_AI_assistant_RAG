# AI 客服助手 (Biocomma AI Customer Service)

基于 **RAG（检索增强生成）** 架构的多格式文档智能问答系统，支持 PDF、DOCX、XLSX、TXT、MD 文件的解析、索引和流式对话。

## 系统架构

```
前端 (React + TypeScript)  ──REST/SSE──▶  后端 (FastAPI)
                                               │
                          ┌────────────────────┼────────────────────┐
                          ▼                    ▼                    ▼
                     文件解析              向量检索              LLM 生成
                   (MinerU/DOCX         (Qwen3-VL +           (DeepSeek)
                    /XLSX/TXT)           FAISS)
```

## 技术栈

| 层级 | 技术 |
|------|------|
| 后端 | Python 3.10+ · FastAPI · uvicorn |
| 前端 | React 18 · TypeScript · Vite · Tailwind CSS |
| Embedding | Qwen3-VL-Embedding-2B (本地) · FAISS |
| LLM | DeepSeek Chat (云端 API) |
| 文档解析 | MinerU (PDF) · python-docx (DOCX) · openpyxl (XLSX) |
| OCR | PaddleOCR (附件图片) |

## 目录结构

```
MRAG/
├── backend/
│   ├── app.py                     # FastAPI 入口
│   ├── requirements.txt
│   ├── services/
│   │   ├── file_service.py        # 多格式文件解析
│   │   ├── index_service.py       # FAISS 索引构建与检索
│   │   └── rag_service.py         # RAG 对话 + OCR + LLM 生成
│   └── data/
│       ├── {fileId}/              # 单文件工作目录 (output.md, index_faiss/, images/)
│       ├── kb_import/             # 统一知识库导入目录 (处理后移至 archive/)
│       ├── unified_knowledge_base/ # 统一知识库 (index.faiss + metadata.json)
│       └── chat_uploads/          # 聊天附件临时目录
├── frontend/                      # React 前端
├── Qwen3-VL-Embedding-2B/         # 本地 Embedding 模型
└── standalone_kb_importer.py      # 独立导入脚本
```

## 快速开始

### 1. 环境准备

- Python 3.10+ / Node.js 18+
- [MinerU](https://github.com/opendatalab/MinerU) 文档解析引擎（需下载模型 `mineru-models-download -s modelscope -m all`）
- 本地 Embedding 模型目录 `Qwen3-VL-Embedding-2B/`
- DeepSeek API Key

### 2. 后端

```powershell
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# 配置 backend/.env
# DEEPSEEK_API_KEY=sk-xxx

uvicorn app:app --host 0.0.0.0 --port 8001 --reload
```

### 3. 前端

```powershell
cd frontend
npm install
npm run dev
# 默认连接 http://localhost:8001
```

### 4. 验证

```powershell
curl http://localhost:8001/api/v1/health
# {"ok": true, "version": "1.0.0", "knowledge_bases": 0}
```

## 使用方式

### Web 界面上传

聊天界面直接上传 PDF/DOCX/XLSX/TXT/MD 文件，系统自动解析并构建索引后即可提问。

### 批量导入（统一知识库）

将文件放入 `backend/data/kb_import/`，通过 API 触发导入：

```powershell
curl -X POST http://localhost:8001/api/v1/unified-kb/import
```

聊天时设置 `useUnifiedKB: true` 即可使用统一知识库。

### 独立脚本导入（无需后端）

```powershell
# 将文件放入 backend/data/kb_import/ 后运行
python standalone_kb_importer.py
```

## API 接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/v1/chat` | POST | **SSE 流式问答**（支持统一知识库/单文件/跨库检索） |
| `/api/v1/chat/clear` | POST | 清除对话历史 |
| `/api/v1/pdf/upload` | POST | 上传文件 |
| `/api/v1/pdf/parse` | POST | 触发解析 |
| `/api/v1/pdf/status` | GET | 解析进度 |
| `/api/v1/pdf/page` | GET | PDF 页面图片 |
| `/api/v1/pdf/images` | GET | 解析提取的图片 |
| `/api/v1/files/upload` | POST | 上传聊天附件（自动 OCR） |
| `/api/v1/index/build` | POST | 构建 FAISS 索引 |
| `/api/v1/index/search` | POST | 单文件索引搜索 |
| `/api/v1/knowledge-base/list` | GET | 列出所有知识库 |
| `/api/v1/knowledge-base/search` | POST | 跨库搜索 |
| `/api/v1/knowledge-base/{id}` | DELETE | 删除知识库 |
| `/api/v1/unified-kb/import` | POST | 批量导入到统一知识库 |
| `/api/v1/unified-kb/import/status` | GET | 导入进度 |
| `/api/v1/unified-kb/search` | POST | 统一知识库搜索 |
| `/api/v1/unified-kb/info` | GET | 统一知识库信息 |
| `/api/v1/files/batch-import` | POST | 从 batch_import/ 批量导入 |
| `/api/v1/health` | GET | 健康检查 |

## 核心流程

```
用户提问
  │
  ├── 检测简单提问（问候/身份/COA等）→ 跳过检索
  ├── useUnifiedKB  → 检索统一知识库
  ├── pdfFileId     → 检索单文件索引（无则回退全库）
  ├── kbIds         → 跨库检索
  └── 默认          → 优先统一知识库，否则全库
  │
  ├── Qwen3-VL-Embedding-2B 编码查询
  ├── FAISS Top-K 检索
  ├── 评分 + LLM Grader 复核相关性
  │
  ├── with_context → 组装上下文 → DeepSeek 生成
  └── no_context   → system prompt 直接回复
  │
  └── SSE 流式返回 citation → token → done
```

## 配置

`backend/.env`:

```env
DEEPSEEK_API_KEY=sk-xxx
```

`frontend/.env.local` (可选):

```env
VITE_API_BASE_URL=http://localhost:8001/api/v1
```
