# services/index_service.py
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import os, json, hashlib

# 复用你已有的数据目录结构
DATA_ROOT = Path("data")

# 统一知识库索引目录
UNIFIED_INDEX_DIR = DATA_ROOT / "unified_knowledge_base"

# 本地 Embedding 模型路径
EMBED_MODEL_PATH = r"d:\ydx\workplace06\MRAG\Qwen3-VL-Embedding-2B"

# 延迟导入 heavy 依赖
_embeddings_loaded = False
_qwen3vl_embeddings_cls = None
_markdown_header_splitter = None
_recursive_text_splitter = None
_document = None
_faiss = None

def _ensure_imports():
    """延迟导入 heavy 依赖"""
    global _embeddings_loaded
    global _markdown_header_splitter, _recursive_text_splitter
    global _document, _faiss, _qwen3vl_embeddings_cls
    
    if not _embeddings_loaded:
        from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
        from langchain_core.documents import Document
        from langchain_community.vectorstores import FAISS
        from services.rag_service import Qwen3VLEmbeddings
        
        _qwen3vl_embeddings_cls = Qwen3VLEmbeddings
        _markdown_header_splitter = MarkdownHeaderTextSplitter
        _recursive_text_splitter = RecursiveCharacterTextSplitter
        _document = Document
        _faiss = FAISS
        _embeddings_loaded = True

def workdir(file_id: str) -> Path:
    p = DATA_ROOT / file_id
    p.mkdir(parents=True, exist_ok=True)
    return p

def markdown_path(file_id: str) -> Path:
    return workdir(file_id) / "output.md"

def index_dir(file_id: str) -> Path:
    p = workdir(file_id) / "index_faiss"
    p.mkdir(parents=True, exist_ok=True)
    return p

def load_embeddings():
    _ensure_imports()
    return _qwen3vl_embeddings_cls(model_path=EMBED_MODEL_PATH)


# Chunk size configuration by file type
CHUNK_CONFIGS = {
    "pdf": {"chunk_size": 800, "chunk_overlap": 100},
    "docx": {"chunk_size": 600, "chunk_overlap": 80},
}


def split_markdown(md_text: str, file_type: str = "text"):
    """根据文件类型选择不同的切分策略"""
    _ensure_imports()
    
    # PDF/DOCX: 使用 RecursiveCharacterTextSplitter 智能切分
    config = CHUNK_CONFIGS.get(file_type)
    if config:
        chunk_size = config["chunk_size"]
        chunk_overlap = config["chunk_overlap"]
        splitter = _recursive_text_splitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
        )
        chunks = splitter.split_text(md_text)
        return [_document(page_content=txt, metadata={}) for txt in chunks if txt.strip()]
    
    # XLSX: 每行作为一个 chunk
    if file_type == "xlsx":
        docs = []
        for line in md_text.split("\n"):
            line = line.strip()
            if line:
                docs.append(_document(page_content=line, metadata={}))
        return docs
    
    # TXT/MD: 按 Markdown 标题切分（原有逻辑）
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
    ]
    splitter = _markdown_header_splitter(headers_to_split_on=headers_to_split_on)
    docs = splitter.split_text(md_text)
    
    cleaned = []
    for d in docs:
        txt = (d.page_content or "").strip()
        if not txt:
            continue
        # 限制太长的段落，避免向量化出错
        if len(txt) > 8000:
            txt = txt[:8000]
        cleaned.append(_document(page_content=txt, metadata=d.metadata))
    return cleaned


def detect_file_type(file_id: str) -> str:
    """检测文件类型"""
    wd = workdir(file_id)
    for ext, ftype in [(".pdf", "pdf"), (".docx", "docx"), (".xlsx", "xlsx"), (".txt", "txt"), (".md", "md")]:
        if (wd / ("original" + ext)).exists():
            return ftype
    return "text"


def _deduplicate_chunks(docs):
    """基于 MD5 哈希去重，移除完全相同的 chunk"""
    seen = set()
    unique_docs = []
    removed_count = 0
    for doc in docs:
        # 计算文本的 MD5 哈希
        content_hash = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()
        if content_hash not in seen:
            seen.add(content_hash)
            unique_docs.append(doc)
        else:
            removed_count += 1
    if removed_count > 0:
        print(f"  Deduplication: removed {removed_count} duplicate chunks, kept {len(unique_docs)}")
    return unique_docs


def build_faiss_index(file_id: str) -> Dict[str, Any]:
    _ensure_imports()
    
    md_file = markdown_path(file_id)
    if not md_file.exists():
        return {"ok": False, "error": "MARKDOWN_NOT_FOUND"}
    md_text = md_file.read_text(encoding="utf-8")

    file_type = detect_file_type(file_id)
    docs = split_markdown(md_text, file_type)
    if not docs:
        return {"ok": False, "error": "EMPTY_MD"}

    # 去重：移除完全相同的 chunk
    docs = _deduplicate_chunks(docs)
    if not docs:
        return {"ok": False, "error": "EMPTY_MD_AFTER_DEDUP"}

    embeddings = load_embeddings()
    vs = _faiss.from_documents(docs, embedding=embeddings)
    vs.save_local(str(index_dir(file_id)))
    return {"ok": True, "chunks": len(docs)}

def search_faiss(file_id: str, query: str, k: int = 5) -> Dict[str, Any]:
    _ensure_imports()
    
    idx = index_dir(file_id)
    if not (idx / "index.faiss").exists():
        return {"ok": False, "error": "INDEX_NOT_FOUND"}

    embeddings = load_embeddings()
    vs = _faiss.load_local(str(idx), embeddings, allow_dangerous_deserialization=True)
    hits = vs.similarity_search_with_score(query, k=k)
    results = []
    for doc, score in hits:
        results.append({
            "text": doc.page_content,
            "score": float(score),
            "metadata": doc.metadata,
        })
    return {"ok": True, "results": results}


def list_available_indexes(data_dir: str = None) -> List[Dict[str, Any]]:
    """Scan data/ directory for all built FAISS indexes.
    Returns list of dicts: [{file_id, name, file_type, created_at, ...}]
    """
    scan_dir = Path(data_dir) if data_dir else DATA_ROOT
    indexes = []
    if not scan_dir.exists():
        return indexes
    for item in sorted(scan_dir.iterdir()):
        if item.is_dir():
            idx = item / "index_faiss"
            if (idx / "index.faiss").exists():
                # Try to get metadata
                original = None
                for ext in [".pdf", ".docx", ".xlsx", ".txt", ".md"]:
                    p = item / ("original" + ext)
                    if p.exists():
                        original = p
                        break
                if original is None:
                    # fallback to old naming
                    old = item / "original.pdf"
                    if old.exists():
                        original = old

                created_ts = 0
                if original:
                    created_ts = int(original.stat().st_mtime)

                indexes.append({
                    "file_id": item.name,
                    "name": original.name if original else item.name,
                    "file_type": original.suffix.lstrip(".") if original else "unknown",
                    "status": "ready",
                    "created_at": created_ts,
                })
    return indexes


def search_all_indexes(query: str, k: int = 5, data_dir: str = None):
    """Search across ALL knowledge base indexes.
    Returns list of (file_id, score, metadata) tuples.
    """
    _ensure_imports()
    
    all_results = []
    scan_dir = Path(data_dir) if data_dir else DATA_ROOT
    indexes = list_available_indexes(data_dir)
    embeddings = load_embeddings()

    for idx_info in indexes:
        file_id = idx_info["file_id"]
        idx_path = scan_dir / file_id / "index_faiss"
        try:
            vs = _faiss.load_local(str(idx_path), embeddings, allow_dangerous_deserialization=True)
            hits = vs.similarity_search_with_score(query, k=k)
            for doc, score in hits:
                all_results.append((file_id, float(score), doc.metadata))
        except Exception:
            continue

    # Sort by score (lower = more similar in L2) and return top k
    all_results.sort(key=lambda x: x[1])
    return all_results[:k]


# ---------------------- 统一知识库索引功能 ----------------------

def unified_index_dir() -> Path:
    """返回统一知识库索引的目录路径"""
    UNIFIED_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    return UNIFIED_INDEX_DIR


def file_metadata_path() -> Path:
    """返回统一知识库文件元数据的存储路径"""
    UNIFIED_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    return UNIFIED_INDEX_DIR / "file_metadata.json"


def _load_raw_unified_index():
    """加载 standalone_kb_importer 生成的原始 FAISS + metadata.json 格式索引。

    返回 (faiss_index, metadata_list, embeddings) 三元组，失败返回 (None, None, None)。
    """
    import faiss as _faiss_raw
    import numpy as np

    idx_dir = unified_index_dir()
    index_file = idx_dir / "index.faiss"
    meta_file = idx_dir / "metadata.json"

    if not index_file.exists() or not meta_file.exists():
        return None, None, None

    try:
        raw_index = _faiss_raw.read_index(str(index_file))
        with open(meta_file, "r", encoding="utf-8") as f:
            metadata_list = json.load(f)
        embeddings = load_embeddings()
        return raw_index, metadata_list, embeddings
    except Exception as e:
        print(f"Failed to load raw unified index: {e}")
        return None, None, None


def load_unified_index():
    """加载统一知识库索引，不存在时返回 None。

    支持两种格式：
    1. LangChain FAISS (index.faiss + index.pkl)
    2. 原始 FAISS + metadata.json (standalone_kb_importer 生成)
    """
    _ensure_imports()
    
    idx_dir = unified_index_dir()
    if not (idx_dir / "index.faiss").exists():
        return None
    try:
        embeddings = load_embeddings()
        vs = _faiss.load_local(str(idx_dir), embeddings, allow_dangerous_deserialization=True)
        return vs
    except Exception:
        # LangChain 格式加载失败（可能缺少 index.pkl），尝试原始格式
        pass

    raw_idx, meta_list, embeddings = _load_raw_unified_index()
    if raw_idx is None:
        print("Failed to load unified index: no compatible format found")
        return None

    # 用原始 FAISS + metadata.json 重建 LangChain FAISS wrapper
    from langchain_core.documents import Document as _Doc
    from langchain_community.docstore.in_memory import InMemoryDocstore

    docs = []
    for entry in meta_list:
        meta = entry.get("metadata", {})
        meta["source_file"] = meta.get("file_name", "")
        meta["file_id"] = meta.get("file_id", "")
        doc = _Doc(page_content=entry.get("text", ""), metadata=meta)
        docs.append(doc)

    if docs:
        docstore_obj = InMemoryDocstore({i: docs[i] for i in range(len(docs))})
        id_map = {i: i for i in range(len(docs))}
        vs = _faiss(
            embedding_function=embeddings,
            index=raw_idx,
            docstore=docstore_obj,
            index_to_docstore_id=id_map,
            normalize_L2=False,
            distance_strategy=_faiss.DistanceStrategy.MAX_INNER_PRODUCT,
        )
        print(f"Loaded unified index from raw FAISS: {len(docs)} documents, "
              f"index size={raw_idx.ntotal}, dim={raw_idx.d}")
        return vs

    return None


def save_unified_index(vs):
    """保存统一知识库索引"""
    vs.save_local(str(unified_index_dir()))


def load_file_metadata() -> Dict[str, Any]:
    """加载文件元数据（记录哪些文件已加入统一知识库）"""
    meta_path = file_metadata_path()
    if not meta_path.exists():
        return {"files": {}}
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Failed to load file metadata: {e}")
        return {"files": {}}


def save_file_metadata(metadata: Dict[str, Any]):
    """保存文件元数据"""
    with open(file_metadata_path(), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def get_file_hash(file_path: Path) -> str:
    """计算文件内容的MD5哈希值"""
    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5.update(chunk)
    return md5.hexdigest()


def add_documents_to_unified_index(
    docs, 
    file_name: str, 
    file_id: str
) -> bool:
    """将文档添加到统一知识库索引中"""
    _ensure_imports()
    
    # 为每个文档添加来源文件信息
    for doc in docs:
        doc.metadata["source_file"] = file_name
        doc.metadata["file_id"] = file_id

    vs = load_unified_index()
    embeddings = load_embeddings()
    
    if vs is None:
        vs = _faiss.from_documents(docs, embedding=embeddings)
    else:
        vs.add_documents(docs)
    
    save_unified_index(vs)
    return True


def search_unified_index(query: str, k: int = 5) -> Dict[str, Any]:
    """在统一知识库中搜索（直接使用原始FAISS，不依赖LangChain wrapper）"""
    import faiss as _faiss_raw
    import numpy as np

    idx_dir = unified_index_dir()
    index_file = idx_dir / "index.faiss"
    meta_file = idx_dir / "metadata.json"

    if not index_file.exists() or not meta_file.exists():
        return {"ok": False, "error": "UNIFIED_INDEX_NOT_FOUND"}

    try:
        raw_index = _faiss_raw.read_index(str(index_file))
        with open(meta_file, "r", encoding="utf-8") as f:
            metadata_list = json.load(f)

        embeddings = load_embeddings()
        query_vec = np.array(embeddings.embed_query(query), dtype=np.float32)

        distances, indices = raw_index.search(query_vec.reshape(1, -1), min(k, raw_index.ntotal))

        results = []
        for i_idx, score in zip(indices[0], distances[0]):
            if i_idx < 0 or i_idx >= len(metadata_list):
                continue
            entry = metadata_list[i_idx]
            meta = entry.get("metadata", {})
            meta["source_file"] = meta.get("file_name", "")
            meta["file_id"] = meta.get("file_id", "")
            results.append({
                "text": entry.get("text", ""),
                "score": float(score),
                "metadata": meta,
            })

        if results:
            print(f"Unified KB raw search: query='{query[:60]}...', "
                  f"found {len(results)}, top_score={results[0]['score']:.4f}")
            print(f"  Top1 text preview: {results[0]['text'][:120]}...")
        else:
            print(f"Unified KB raw search: query='{query[:60]}...', no results")
        return {"ok": True, "results": results}

    except Exception as e:
        print(f"Unified KB raw search error: {e}")
        import traceback
        traceback.print_exc()
        return {"ok": False, "error": str(e)}


def check_file_already_imported(file_path: Path):
    """检查文件是否已导入到统一知识库"""
    metadata = load_file_metadata()
    file_hash = get_file_hash(file_path)
    
    for fid, info in metadata.get("files", {}).items():
        if info.get("hash") == file_hash:
            return True, fid
    
    return False, None
