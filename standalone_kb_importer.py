#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Standalone Knowledge Base Importer
导入文件到知识库，支持多模态文档处理
"""
from __future__ import annotations
import os
import sys
import json
import hashlib
import tempfile
import subprocess
import shutil
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

# 禁用 oneDNN/MKLDNN 以避免 Windows 上的问题
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_enable_pir_inference"] = "0"
os.environ["FLAGS_enable_onednn_layouts"] = "0"

# 强制 CPU 模式：必须在 import torch 之前设置，防止 CUDA 初始化导致进程崩溃
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# 添加项目路径到 sys.path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

# 导入服务模块
from services.file_service import (
    workdir as file_workdir,
    find_actual_file,
    detect_file_type,
    save_upload,
    parse_pdf_with_mineru,
    parse_docx,
    parse_xlsx,
    parse_text_file,
    run_full_parse_pipeline,
    markdown_output,
    images_dir
)

# 直接导入 langchain 分块组件（不经过 index_service._ensure_imports 避免触发重依赖下载）
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_core.documents import Document as LangchainDocument

import torch
import numpy as np
from PIL import Image

# 尝试导入 FAISS
try:
    import faiss
    HAS_FAISS = True
except BaseException:
    HAS_FAISS = False
    print("警告: faiss 不可用，将使用替代方案")

# 配置路径
KB_IMPORT_DIR = PROJECT_ROOT / "backend" / "data" / "kb_import"
UNIFIED_INDEX_DIR = PROJECT_ROOT / "backend" / "data" / "unified_knowledge_base"
EMBED_MODEL_PATH = PROJECT_ROOT / "Qwen3-VL-Embedding-2B"


@dataclass
class DocumentChunk:
    """文档分块数据结构"""
    text: str
    image_path: Optional[str] = None
    metadata: Dict[str, Any] = None
    embedding: Optional[np.ndarray] = None


class EmbeddingModel:
    """统一的嵌入模型接口
    
    加载策略（按优先级）：
    1. Qwen3-VL-Embedding-2B (多模态，维度高，需要大内存)
    2. all-MiniLM-L6-v2 (纯文本，384维，仅 ~80MB，CPU 友好)
    """
    
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self._backend = None       # 实际嵌入器
        self._backend_name = ""    # 后端名称
        self._embedding_dim: int = 0
        
        # 按优先级尝试加载
        if not self._try_load_qwen3vl():
            if not self._try_load_sentence_transformers():
                raise RuntimeError(
                    "无法加载任何嵌入模型。\n"
                    "请安装 sentence-transformers: pip install sentence-transformers"
                )
    
    def _try_load_qwen3vl(self) -> bool:
        """尝试加载 Qwen3-VL-Embedding-2B"""
        print("尝试加载 Qwen3-VL-Embedding-2B ...", flush=True)
        try:
            scripts_dir = self.model_path / "scripts"
            if scripts_dir.exists():
                sys.path.insert(0, str(scripts_dir))
            
            try:
                from qwen3_vl_embedding import Qwen3VLEmbedder
            except ImportError:
                import importlib.util
                script_path = scripts_dir / "qwen3_vl_embedding.py"
                if not script_path.exists():
                    print("  Qwen3-VL 模型脚本不存在，跳过", flush=True)
                    return False
                spec = importlib.util.spec_from_file_location("qwen3_vl_embedding", script_path)
                qwen_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(qwen_module)
                Qwen3VLEmbedder = qwen_module.Qwen3VLEmbedder
            
            print(f"  加载模型权重: {self.model_path}", flush=True)
            self._backend = Qwen3VLEmbedder(
                model_name_or_path=str(self.model_path),
                torch_dtype=torch.float16,   # fp16 减少一半内存 (CPU 兼容)
            )
            
            # 探测维度
            print("  探测 embedding 维度...", flush=True)
            test_emb = self._backend.process([{"text": "test"}])
            if isinstance(test_emb, list):
                test_vec = test_emb[0]
            else:
                test_vec = test_emb
            if hasattr(test_vec, 'cpu'):
                test_vec = test_vec.cpu().numpy()
            else:
                test_vec = np.array(test_vec)
            self._embedding_dim = int(test_vec.shape[-1])
            
            self._backend_name = f"Qwen3-VL-Embedding-2B ({self._embedding_dim}维)"
            print(f"  Qwen3-VL 加载成功! 维度: {self._embedding_dim}", flush=True)
            return True
            
        except BaseException as e:
            print(f"  Qwen3-VL 加载失败: {type(e).__name__}", flush=True)
            self._backend = None
            return False
    
    def _try_load_sentence_transformers(self) -> bool:
        """回退方案：加载轻量级 sentence-transformers 模型"""
        print("回退到 sentence-transformers (all-MiniLM-L6-v2) ...", flush=True)
        try:
            from sentence_transformers import SentenceTransformer
            self._backend = SentenceTransformer('all-MiniLM-L6-v2')
            self._embedding_dim = 384
            self._backend_name = "all-MiniLM-L6-v2 (384维, ~80MB)"
            print(f"  sentence-transformers 加载成功!", flush=True)
            return True
        except BaseException as e:
            print(f"  sentence-transformers 加载失败: {type(e).__name__}: {e}", flush=True)
            return False
    
    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim
    
    def _to_numpy(self, tensor_or_array) -> np.ndarray:
        if hasattr(tensor_or_array, 'cpu'):
            return tensor_or_array.cpu().numpy()
        return np.array(tensor_or_array)
    
    def embed_chunks(self, chunks: List[DocumentChunk]) -> List[np.ndarray]:
        """批量生成嵌入向量"""
        if not chunks:
            return []
        
        # 提取纯文本
        texts = [c.text if c.text else "EMPTY" for c in chunks]
        
        if self._backend is None:
            raise RuntimeError("嵌入模型未初始化")
        
        print(f"  生成 {len(texts)} 个 embedding (后端: {self._backend_name})...", flush=True)
        
        if self._backend_name.startswith("Qwen3-VL"):
            # Qwen3-VL 路径：构建 dict 列表
            inputs = []
            for chunk in chunks:
                input_dict = {"text": chunk.text or "EMPTY"}
                if chunk.image_path and os.path.exists(chunk.image_path):
                    input_dict["image"] = chunk.image_path
                inputs.append(input_dict)
            
            with torch.no_grad():
                outputs = self._backend.process(inputs)
            
            embeddings = []
            if isinstance(outputs, list):
                # 列表形式：每项一个 tensor
                for out in outputs:
                    embeddings.append(self._to_numpy(out))
            else:
                # 单个 Tensor 形式：(batch_size, dim) → 按 axis=0 拆分
                arr = self._to_numpy(outputs)
                if arr.ndim == 2:
                    for i in range(arr.shape[0]):
                        embeddings.append(arr[i])
                else:
                    # 仅 1 个输入时的 1D 输出
                    embeddings.append(arr)
        else:
            # sentence-transformers 路径
            embeddings = self._backend.encode(
                texts,
                show_progress_bar=True,
                convert_to_numpy=True,
            )
            embeddings = [np.array(e) for e in embeddings]
        
        print(f"  embedding 完成 ({len(embeddings)} 个)", flush=True)
        return embeddings


class VectorStore:
    """向量存储管理类"""
    
    def __init__(self, index_dir: Path, embedding_dim: int = 0):
        self.index_dir = index_dir
        self.embedding_dim = embedding_dim  # 0 表示延迟初始化，在首次添加文档时自动检测
        self.index = None
        self.documents = []  # 存储文档元数据
        self._content_hashes: set = set()  # 用于去重的内容哈希集合
        self.index_file = index_dir / "index.faiss"
        self.metadata_file = index_dir / "metadata.json"
        self._hash_file = index_dir / "content_hashes.json"
        self._init_store()
    
    def _init_store(self):
        """初始化向量存储"""
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        if self.index_file.exists() and self.metadata_file.exists():
            self._load_store()
        else:
            self._create_new_store()
    
    def _create_new_store(self):
        """创建新的向量存储（延迟到知道维度再创建 FAISS 索引）"""
        # 先不创建 FAISS 索引，等第一次 add_documents 时根据实际维度创建
        self.index = None
        self.documents = []
        self._content_hashes = set()
    
    def _ensure_index(self, dim: int):
        """确保 FAISS 索引已按正确维度初始化"""
        if self.embedding_dim == 0:
            self.embedding_dim = dim
            print(f"  自动检测 embedding 维度: {self.embedding_dim}")
        elif self.embedding_dim != dim:
            raise ValueError(
                f"embedding 维度不匹配: 存储={self.embedding_dim}, 新数据={dim}"
            )
        
        if self.index is None and HAS_FAISS:
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            print(f"  创建 FAISS 索引 (维度={self.embedding_dim})")
    
    def _load_store(self):
        """加载已有存储"""
        if HAS_FAISS and self.index_file.exists():
            self.index = faiss.read_index(str(self.index_file))
            self.embedding_dim = self.index.d
        
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                self.documents = json.load(f)
        
        # 加载内容哈希用于去重
        if self._hash_file.exists():
            with open(self._hash_file, 'r', encoding='utf-8') as f:
                self._content_hashes = set(json.load(f))
    
    def _save_store(self):
        """保存存储"""
        if HAS_FAISS and self.index is not None:
            faiss.write_index(self.index, str(self.index_file))
        
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)
        
        # 持久化内容哈希
        with open(self._hash_file, 'w', encoding='utf-8') as f:
            json.dump(list(self._content_hashes), f)
    
    def add_documents(self, chunks: List[DocumentChunk], embeddings: List[np.ndarray]):
        """添加文档到存储（基于内容哈希去重）"""
        if not chunks or not embeddings:
            return
        
        # 基于内容哈希去重
        new_chunks = []
        new_embeddings = []
        skipped = 0
        for chunk, emb in zip(chunks, embeddings):
            content_hash = hashlib.md5(chunk.text.encode("utf-8")).hexdigest()
            if content_hash in self._content_hashes:
                skipped += 1
                continue
            self._content_hashes.add(content_hash)
            new_chunks.append(chunk)
            new_embeddings.append(emb)
        
        if skipped > 0:
            print(f"  [去重] 跳过 {skipped} 个重复分块")
        
        if not new_chunks:
            print("  [去重] 所有分块均为重复内容，无需添加")
            return
        
        # 根据实际 embedding 维度初始化 FAISS 索引
        first_dim = int(new_embeddings[0].shape[-1])
        self._ensure_index(first_dim)
        
        # 添加到 FAISS 索引
        if HAS_FAISS and self.index is not None:
            embeddings_array = np.vstack(new_embeddings).astype('float32')
            self.index.add(embeddings_array)
        
        # 保存元数据
        for chunk in new_chunks:
            doc_metadata = {
                "text": chunk.text,
                "image_path": chunk.image_path,
                "metadata": chunk.metadata or {}
            }
            self.documents.append(doc_metadata)
        
        self._save_store()
        print(f"已添加 {len(new_chunks)} 个分块到向量存储")


def generate_file_id(file_path: Path) -> str:
    """为文件生成唯一ID"""
    content = file_path.read_bytes()
    hash_obj = hashlib.md5(content)
    return hash_obj.hexdigest()


def extract_images_from_markdown(md_content: str, images_dir_path: Path) -> List[Tuple[int, str]]:
    """从 Markdown 内容中提取图片引用，相对于 markdown 文件所在目录解析路径"""
    images = []
    import re
    
    # 匹配 Markdown 图片语法 ![alt](path)
    pattern = r'!\[.*?\]\((.*?)\)'
    matches = re.finditer(pattern, md_content)
    
    # markdown 文件位于 images_dir_path 的父目录中
    # 图片相对路径（如 images/xxx.jpg）需相对于 markdown 文件所在目录解析
    base_dir = images_dir_path.parent  # data/{file_id}/
    
    for match in matches:
        rel_path = match.group(1)
        # 去除 ./ 或 .\ 前缀
        clean_path = rel_path.lstrip("./\\")
        full_path = base_dir / clean_path
        if full_path.exists():
            images.append((match.start(), str(full_path)))
    
    return images


# Chunk size configuration by file type
CHUNK_CONFIGS = {
    "pdf": {"chunk_size": 800, "chunk_overlap": 100},
    "docx": {"chunk_size": 600, "chunk_overlap": 80},
}


def _split_markdown_text(md_text: str, file_type: str):
    """根据文件类型选择不同的切分策略（本地实现，不依赖 index_service）"""
    # PDF/DOCX: 使用 RecursiveCharacterTextSplitter 智能切分
    config = CHUNK_CONFIGS.get(file_type)
    if config:
        chunk_size = config["chunk_size"]
        chunk_overlap = config["chunk_overlap"]
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
        )
        chunks = splitter.split_text(md_text)
        return [LangchainDocument(page_content=txt, metadata={}) for txt in chunks if txt.strip()]

    # XLSX: 每行作为一个 chunk
    if file_type == "xlsx":
        docs = []
        for line in md_text.split("\n"):
            line = line.strip()
            if line:
                docs.append(LangchainDocument(page_content=line, metadata={}))
        return docs

    # TXT/MD: 按 Markdown 标题切分
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
    ]
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    docs = splitter.split_text(md_text)

    cleaned = []
    for d in docs:
        txt = (d.page_content or "").strip()
        if not txt:
            continue
        if len(txt) > 8000:
            txt = txt[:8000]
        cleaned.append(LangchainDocument(page_content=txt, metadata=d.metadata))
    return cleaned


def _deduplicate_chunks_local(docs):
    """基于 MD5 哈希去重（本地实现）"""
    seen = set()
    unique_docs = []
    removed_count = 0
    for doc in docs:
        content_hash = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()
        if content_hash not in seen:
            seen.add(content_hash)
            unique_docs.append(doc)
        else:
            removed_count += 1
    if removed_count > 0:
        print(f"  [4/5] Deduplication: removed {removed_count} duplicate chunks, kept {len(unique_docs)}")
    return unique_docs


def process_document(file_path: Path, embed_model: EmbeddingModel) -> List[DocumentChunk]:
    """处理单个文档文件"""
    print(f"\n{'='*50}")
    print(f"处理文件: {file_path.name}")
    print(f"{'='*50}")
    
    try:
        # 生成文件ID
        file_id = generate_file_id(file_path)
        print(f"  [1/5] 文件ID: {file_id[:12]}...")
        
        # 复制文件到工作目录
        file_workdir(file_id)  # 创建目录
        ext = file_path.suffix.lower()
        dest_path = file_workdir(file_id) / f"original{ext}"
        shutil.copy2(file_path, dest_path)
        
        # 解析文件
        print(f"  [2/5] 开始解析文件...")
        try:
            parse_result = run_full_parse_pipeline(file_id)
            print(f"  [2/5] 文件解析完成")
        except Exception as e:
            print(f"  [2/5] 文件解析失败: {e}")
            traceback.print_exc()
            return []
        
        # 读取解析后的 Markdown
        md_file = markdown_output(file_id)
        if not md_file.exists():
            print(f"  [3/5] 未找到解析后的 Markdown 文件: {md_file}")
            return []
        
        md_content = md_file.read_text(encoding='utf-8')
        print(f"  [3/5] Markdown 读取成功，长度: {len(md_content)} 字符")
        
        # 检测文件类型并分块
        print(f"  [4/5] 开始分块...")
        file_type = ext.lstrip('.')  # 从文件扩展名确定类型
        print(f"  [4/5] 检测到文件类型: {file_type}")
        
        try:
            chunks = _split_markdown_text(md_content, file_type)
        except Exception as e:
            print(f"  [4/5] 分块失败: {e}")
            traceback.print_exc()
            return []
        
        if not chunks:
            print("  [4/5] 未生成有效分块")
            return []
        
        # 去重
        chunks = _deduplicate_chunks_local(chunks)
        print(f"  [4/5] 生成 {len(chunks)} 个分块 (去重后)")
        
        # 转换为 DocumentChunk 并提取图片
        print(f"  [5/5] 构建 DocumentChunk...")
        doc_chunks = []
        images_dir_path = images_dir(file_id)
        
        for i, chunk in enumerate(chunks):
            text = chunk.page_content
            
            # 查找这个分块中引用的图片
            chunk_images = extract_images_from_markdown(text, images_dir_path)
            
            if chunk_images:
                # 如果有图片，创建包含图片的分块
                for pos, img_path in chunk_images:
                    doc_chunk = DocumentChunk(
                        text=text,
                        image_path=img_path,
                        metadata={
                            "file_id": file_id,
                            "file_name": file_path.name,
                            "chunk_index": i,
                            "file_type": file_type,
                            **chunk.metadata
                        }
                    )
                    doc_chunks.append(doc_chunk)
            else:
                # 纯文本分块
                doc_chunk = DocumentChunk(
                    text=text,
                    metadata={
                        "file_id": file_id,
                        "file_name": file_path.name,
                        "chunk_index": i,
                        "file_type": file_type,
                        **chunk.metadata
                    }
                )
                doc_chunks.append(doc_chunk)
        
        print(f"  [5/5] 成功生成 {len(doc_chunks)} 个 DocumentChunk")
        return doc_chunks
        
    except Exception as e:
        print(f"\n  *** 处理文件时发生未预期的错误: {e}")
        traceback.print_exc()
        return []


def main():
    """主函数"""
    print("=" * 60)
    print("知识库批量导入工具")
    print("=" * 60)
    
    # 检查导入目录
    if not KB_IMPORT_DIR.exists():
        print(f"导入目录不存在: {KB_IMPORT_DIR}")
        KB_IMPORT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"已创建导入目录，请将文件放入: {KB_IMPORT_DIR}")
        return
    
    # 获取待处理文件
    supported_exts = ['.pdf', '.docx', '.xlsx', '.txt', '.md']
    files_to_process = []
    
    for file_path in KB_IMPORT_DIR.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in supported_exts:
            files_to_process.append(file_path)
    
    if not files_to_process:
        print(f"导入目录中没有支持的文件: {KB_IMPORT_DIR}")
        print("支持的格式: PDF, DOCX, XLSX, TXT, MD")
        return
    
    print(f"找到 {len(files_to_process)} 个文件待处理")
    for f in files_to_process:
        print(f"  - {f.name}")
    
    # 加载嵌入模型
    print("\n正在加载嵌入模型...")
    try:
        embed_model = EmbeddingModel(EMBED_MODEL_PATH)
    except Exception as e:
        print(f"模型加载失败: {e}")
        print("请确保 Qwen3-VL-Embedding-2B 模型在正确位置")
        return
    
    # 初始化向量存储
    vector_store = VectorStore(UNIFIED_INDEX_DIR)
    
    # 处理每个文件
    all_chunks = []
    success_count = 0
    fail_count = 0
    for file_path in files_to_process:
        try:
            chunks = process_document(file_path, embed_model)
            if chunks:
                all_chunks.extend(chunks)
                success_count += 1
            else:
                print(f"  >>> 文件 {file_path.name} 未生成有效分块，跳过")
                fail_count += 1
        except Exception as e:
            print(f"  >>> 处理文件 {file_path.name} 时发生异常: {e}")
            traceback.print_exc()
            fail_count += 1
    
    print(f"\n处理结果: 成功 {success_count} 个文件, 失败 {fail_count} 个文件")
    
    if not all_chunks:
        print("\n没有成功处理任何文档分块")
        return
    
    print(f"\n总共生成 {len(all_chunks)} 个分块")
    
    # 生成嵌入向量
    print("\n正在生成嵌入向量...")
    try:
        embeddings = embed_model.embed_chunks(all_chunks)
        print("嵌入向量生成完成")
    except Exception as e:
        print(f"嵌入生成失败: {e}")
        return
    
    # 保存到向量存储
    print("\n正在保存到向量存储...")
    vector_store.add_documents(all_chunks, embeddings)
    
    print("\n" + "=" * 60)
    print("导入完成!")
    print(f"处理文件数: {len(files_to_process)}")
    print(f"总分块数: {len(all_chunks)}")
    print(f"向量存储位置: {UNIFIED_INDEX_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
