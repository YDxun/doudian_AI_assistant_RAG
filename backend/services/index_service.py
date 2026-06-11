# services/index_service.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import json

DATA_ROOT = Path("data")
UNIFIED_INDEX_DIR = DATA_ROOT / "unified_knowledge_base"
EMBED_MODEL_PATH = r"d:\ydx\workplace06\MRAG\Qwen3-VL-Embedding-2B"

# 延迟导入 heavy 依赖
_qwen3vl_embeddings_cls = None
_faiss = None
_distance_strategy = None

def _ensure_imports():
    global _qwen3vl_embeddings_cls, _faiss, _distance_strategy

    if _qwen3vl_embeddings_cls is None:
        from langchain_community.vectorstores import FAISS
        from langchain_community.vectorstores.faiss import DistanceStrategy
        from services.rag_service import Qwen3VLEmbeddings

        _qwen3vl_embeddings_cls = Qwen3VLEmbeddings
        _faiss = FAISS
        _distance_strategy = DistanceStrategy


def load_embeddings():
    _ensure_imports()
    return _qwen3vl_embeddings_cls(model_path=EMBED_MODEL_PATH)


def unified_index_dir() -> Path:
    UNIFIED_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    return UNIFIED_INDEX_DIR


def file_metadata_path() -> Path:
    UNIFIED_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    return UNIFIED_INDEX_DIR / "file_metadata.json"


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


def _load_raw_unified_index():
    """加载 standalone_kb_importer 生成的原始 FAISS + metadata.json"""
    import faiss as _faiss_raw

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
    """加载统一知识库索引"""
    _ensure_imports()

    idx_dir = unified_index_dir()
    if not (idx_dir / "index.faiss").exists():
        return None
    try:
        embeddings = load_embeddings()
        vs = _faiss.load_local(str(idx_dir), embeddings, allow_dangerous_deserialization=True)
        return vs
    except Exception:
        pass

    raw_idx, meta_list, embeddings = _load_raw_unified_index()
    if raw_idx is None:
        print("Failed to load unified index: no compatible format found")
        return None

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
            distance_strategy=_distance_strategy.MAX_INNER_PRODUCT,
        )
        print(f"Loaded unified index from raw FAISS: {len(docs)} documents, "
              f"index size={raw_idx.ntotal}, dim={raw_idx.d}")
        return vs

    return None


def search_unified_index(query: str, k: int = 5) -> Dict[str, Any]:
    """在统一知识库中搜索（直接使用原始FAISS）"""
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
            print(f"Unified KB search: query='{query[:60]}', "
                  f"found {len(results)}, top_score={results[0]['score']:.4f}")
        else:
            print(f"Unified KB search: query='{query[:60]}', no results")
        return {"ok": True, "results": results}

    except Exception as e:
        print(f"Unified KB search error: {e}")
        import traceback
        traceback.print_exc()
        return {"ok": False, "error": str(e)}
