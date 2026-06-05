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
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

# 禁用 oneDNN/MKLDNN 以避免 Windows 上的问题
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_enable_pir_inference"] = "0"
os.environ["FLAGS_enable_onednn_layouts"] = "0"

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
except ImportError:
    HAS_FAISS = False
    print("警告: faiss 未安装，将使用替代方案")

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


class Qwen3VLEmbeddingModel:
    """Qwen3-VL-Embedding 模型封装类"""
    
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.embedder = None
        self._load_model()
    
    def _load_model(self):
        """加载模型"""
        try:
            # 添加 scripts 目录到路径
            scripts_dir = self.model_path / "scripts"
            if scripts_dir.exists():
                sys.path.insert(0, str(scripts_dir))
                print(f"添加脚本目录到路径: {scripts_dir}")
            
            # 尝试导入 Qwen3VLEmbedder
            try:
                from qwen3_vl_embedding import Qwen3VLEmbedder
            except ImportError:
                # 尝试直接从 scripts 目录导入
                print("尝试从 scripts 目录导入...")
                import importlib.util
                script_path = scripts_dir / "qwen3_vl_embedding.py"
                if script_path.exists():
                    spec = importlib.util.spec_from_file_location("qwen3_vl_embedding", script_path)
                    qwen_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(qwen_module)
                    Qwen3VLEmbedder = qwen_module.Qwen3VLEmbedder
                else:
                    raise ImportError("无法找到 qwen3_vl_embedding 模块")
            
            print(f"正在加载模型: {self.model_path}")
            self.embedder = Qwen3VLEmbedder(
                model_name_or_path=str(self.model_path)
            )
            print("模型加载成功!")
        except Exception as e:
            print(f"模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def embed_chunk(self, chunk: DocumentChunk) -> np.ndarray:
        """为单个分块生成嵌入向量"""
        inputs = []
        input_dict = {}
        
        if chunk.text:
            input_dict["text"] = chunk.text
        
        if chunk.image_path and os.path.exists(chunk.image_path):
            input_dict["image"] = chunk.image_path
        
        if not input_dict:
            input_dict["text"] = "EMPTY"
        
        inputs.append(input_dict)
        
        # 生成嵌入
        with torch.no_grad():
            embeddings = self.embedder.process(inputs)
        
        return embeddings[0].cpu().numpy()
    
    def embed_chunks(self, chunks: List[DocumentChunk]) -> List[np.ndarray]:
        """批量生成嵌入向量"""
        embeddings = []
        for chunk in chunks:
            emb = self.embed_chunk(chunk)
            embeddings.append(emb)
        return embeddings


class VectorStore:
    """向量存储管理类"""
    
    def __init__(self, index_dir: Path, embedding_dim: int = 2048):
        self.index_dir = index_dir
        self.embedding_dim = embedding_dim
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
        """创建新的向量存储"""
        if HAS_FAISS:
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.documents = []
        self._content_hashes = set()
    
    def _load_store(self):
        """加载已有存储"""
        if HAS_FAISS and self.index_file.exists():
            self.index = faiss.read_index(str(self.index_file))
        
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
        
        # 添加到 FAISS 索引
        if HAS_FAISS and self.index is not None:
            embeddings_array = np.array(new_embeddings).astype('float32')
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


def process_document(file_path: Path, embed_model: Qwen3VLEmbeddingModel) -> List[DocumentChunk]:
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
            import traceback
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
            import traceback
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
        import traceback
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
        embed_model = Qwen3VLEmbeddingModel(EMBED_MODEL_PATH)
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
            import traceback
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
