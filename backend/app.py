import os
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Query, Body
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import asyncio, time, random, string, shutil, json
from typing import Optional, Dict, Any, List
from fastapi import HTTPException

from fastapi import BackgroundTasks
from services.file_service import (
    save_upload, run_full_parse_pipeline, find_actual_file,
    dir_original_pages, dir_parsed_pages, markdown_output, images_dir, detect_file_type
)
from services.index_service import (
    build_faiss_index, search_faiss, search_all_indexes, list_available_indexes,
    add_documents_to_unified_index, check_file_already_imported, load_file_metadata, 
    save_file_metadata, get_file_hash, search_unified_index, load_unified_index,
    split_markdown, _deduplicate_chunks, detect_file_type as detect_file_type_from_index
)
from fastapi.responses import StreamingResponse, JSONResponse
from services.rag_service import retrieve, retrieve_multi, answer_stream, clear_history, extract_text_from_file, is_simple_question

app = FastAPI(
    title="逗点生物AI客服助手 API",
    version="1.0.0",
    description="逗点生物AI客服助手后端API，支持PDF、DOCX、TXT、MD等多种文件格式的RAG问答。"
)

# 允许前端本地联调
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_PREFIX = "/api/v1"

# ---------------- 内存态存储 ----------------
current_file: Dict[str, Any] = {
    "fileId": None,
    "name": None,
    "pages": 0,
    "fileType": "pdf",     # pdf | docx | txt | md
    "status": "idle",      # idle | parsing | ready | error
    "progress": 0
}
citations: Dict[str, Dict[str, Any]] = {}   # citationId -> { fileId, page, snippet, bbox, previewUrl }

# 批量导入的进度状态
batch_import_status: Dict[str, Any] = {
    "status": "idle",  # idle | running | completed | error
    "total": 0,
    "processed": 0,
    "success": 0,
    "failed": 0,
    "results": [],
    "current_file": None,
    "current_step": None,
    "current_step_detail": None,
    "error_message": None
}

# ---------------- 工具函数 ----------------
def rid(prefix: str) -> str:
    return f"{prefix}_" + "".join(random.choices(string.ascii_lowercase + string.digits, k=8))

def now_ts() -> int:
    return int(time.time())

def err(code: str, message: str) -> Dict[str, Any]:
    return {"error": {"code": code, "message": message}, "requestId": rid("req"), "ts": now_ts()}

# ---------------- 常量 ----------------
FETCH_COUNT = 10       # 从向量库中拿取的候选切片数
SCORE_THRESHOLD = 0.5     # 仅保留得分 >= 此阈值的切片用于生成回答（内积得分，归一化后 ≈ cosine）

def _filter_unified_search(search_result):
    """从统一知识库搜索结果中过滤低分切片，返回 (citations, context_text, ok)"""
    citations = []
    if not (search_result.get("ok") and search_result.get("results")):
        return citations, "", False

    hits = [h for h in search_result["results"] if h["score"] >= SCORE_THRESHOLD]
    if not hits:
        print(f"Unified KB: all {len(search_result['results'])} results below threshold {SCORE_THRESHOLD}, "
              f"top_score={search_result['results'][0]['score']:.4f}")
        return citations, "", False

    ctx_snippets = []
    for i, hit in enumerate(hits, start=1):
        snippet_short = hit["text"][:500] + "..." if len(hit["text"]) > 500 else hit["text"]
        citations.append({
            "citation_id": f"unified-c{i}",
            "fileId": hit["metadata"].get("file_id", "unified"),
            "rank": i,
            "page": hit["metadata"].get("page"),
            "snippet": hit["text"][:4000],
            "score": hit["score"],
            "source_file": hit["metadata"].get("source_file"),
            "previewUrl": None
        })
        ctx_snippets.append(f"--- 知识库片段 {i} ---\n{snippet_short}")

    context_text = "\n\n".join(ctx_snippets)
    print(f"Unified KB: {len(hits)}/{len(search_result['results'])} passed threshold, "
          f"top_score={hits[0]['score']:.4f}")
    return citations, context_text, True

# ---------------- Pydantic 模型（契约） ----------------
class ChatRequest(BaseModel):
    message: str
    sessionId: Optional[str] = None
    pdfFileId: Optional[str] = None
    knowledgeBaseIds: Optional[List[str]] = None
    attachmentText: Optional[str] = None  # OCR-extracted text from attached file
    useUnifiedKB: Optional[bool] = Field(False, description="是否使用统一知识库")

# ---------------- Health ----------------
@app.get(f"{API_PREFIX}/health", tags=["Health"])
async def health():
    kb_count = len(list_available_indexes())
    return {"ok": True, "version": "1.0.0", "knowledge_bases": kb_count}

# ---------------- File: 上传聊天附件（用于OCR识别） ----------------
UPLOAD_TEMP_DIR = Path("data") / "chat_uploads"
UPLOAD_TEMP_DIR.mkdir(parents=True, exist_ok=True)

@app.post(f"{API_PREFIX}/files/upload", tags=["Files"])
async def file_upload(file: UploadFile = File(...)):
    """上传聊天附件，返回文件内容和OCR提取的文本"""
    if not file:
        return JSONResponse(err("NO_FILE", "缺少文件"), status_code=400)
    
    # 保存到临时目录
    import uuid
    temp_id = str(uuid.uuid4())[:8]
    ext = Path(file.filename or "").suffix.lower()
    temp_path = UPLOAD_TEMP_DIR / f"{temp_id}{ext}"
    
    content = await file.read()
    temp_path.write_bytes(content)
    
    # 先提取文本（OCR）- 必须在删除临时文件之前
    ocr_text = ""
    try:
        ocr_text = extract_text_from_file(str(temp_path), file.filename or "")
    except Exception as e:
        print(f"OCR extraction failed: {e}")
    
    # 清理临时文件
    try:
        temp_path.unlink()
    except:
        pass
    
    return {
        "fileName": file.filename,
        "fileType": ext,
        "extractedText": ocr_text
    }


# ---------------- Chat（SSE，POST 返回 event-stream） ----------------
@app.post(f"{API_PREFIX}/chat", tags=["Chat"])
async def chat_stream(req: ChatRequest):
    """
    SSE 事件：token | citation | done | error

    附件处理策略：
    - 用户原始问题（req.message）用于向量检索，保证检索精度
    - 附件文本（attachmentText 或从 pdfFileId 提取）作为附加上下文，在生成回答时一并传入 LLM
    - 当 pdfFileId 对应的文件索引不存在时，自动回退搜索其他知识库
    """
    async def gen():
        try:
            user_question = (req.message or "").strip()
            session_id = (req.sessionId or "default").strip()
            file_id = (req.pdfFileId or "").strip()
            kb_ids = req.knowledgeBaseIds or []
            use_unified_kb = req.useUnifiedKB if req.useUnifiedKB is not None else False

            # ---------- 提取附件内容（与检索 query 分离） ----------
            attachment_text = (req.attachmentText or "").strip()

            # 如果提供了 fileId 但没有 attachmentText，尝试从上传文件提取文本
            if file_id and not attachment_text:
                try:
                    actual_file = find_actual_file(file_id)
                    if actual_file.exists():
                        file_text = extract_text_from_file(str(actual_file), actual_file.name)
                        if file_text and file_text.strip():
                            attachment_text = file_text.strip()
                except Exception as e:
                    print(f"Chat attachment text extraction failed: {e}")

            # ---------- 检索：使用原始用户问题进行向量检索 ----------
            citations, context_text = [], ""
            branch = "no_context"

            # 检索 query 仅使用用户原始问题，不混入附件内容
            search_query = user_question if user_question else " "

            # 简单提问（问候/身份询问/感谢/告别/价格/COA等）无需检索，直接由 LLM 基于 system prompt 回复
            skip_retrieval = (
                not attachment_text
                and not file_id
                and not kb_ids
                and is_simple_question(user_question)
            )

            if not skip_retrieval:
                if use_unified_kb:
                    try:
                        search_result = search_unified_index(search_query, k=FETCH_COUNT)
                        unified_citations, context_text, branch_ok = _filter_unified_search(search_result)
                        if branch_ok:
                            citations.extend(unified_citations)
                            branch = "with_context"
                    except Exception as e:
                        print(f"Unified KB search error: {e}")
                elif file_id:
                    # 先尝试搜索指定文件的索引；若索引不存在则回退搜索所有知识库
                    try:
                        citations, context_text = await retrieve(search_query, file_id)
                        branch = "with_context" if context_text else "no_context"
                    except FileNotFoundError:
                        # 文件索引不存在，回退搜索所有知识库
                        print(f"Index not found for {file_id}, falling back to all knowledge bases")
                        try:
                            citations, context_text = await retrieve_multi(search_query, [])
                            branch = "with_context" if context_text else "no_context"
                        except Exception:
                            branch = "no_context"
                elif kb_ids:
                    try:
                        citations, context_text = await retrieve_multi(search_query, kb_ids)
                        branch = "with_context" if context_text else "no_context"
                    except Exception:
                        branch = "no_context"
                else:
                    # 无指定：优先统一知识库，否则搜索所有单独索引
                    try:
                        search_result = search_unified_index(search_query, k=FETCH_COUNT)
                        unified_citations, context_text, branch_ok = _filter_unified_search(search_result)
                        if branch_ok:
                            citations.extend(unified_citations)
                            branch = "with_context"
                        else:
                            citations, context_text = await retrieve_multi(search_query, [])
                            branch = "with_context" if context_text else "no_context"
                    except Exception:
                        branch = "no_context"

            # ---------- 组装最终给 LLM 的 question：附件内容作为附加上下文，不混入检索 query ----------
            if attachment_text:
                # 附件内容放在检索上下文之后、用户问题之前
                final_question = (
                    f"【附件内容】\n{attachment_text}\n\n"
                    f"【用户问题】\n{user_question}"
                )
            else:
                final_question = user_question

            # 先推送引用（若有）
            if branch == "with_context" and citations:
                for c in citations:
                    yield "event: citation\n"
                    yield f"data: {json.dumps(c, ensure_ascii=False)}\n\n"

            # 再推送 token 流（内部会写入历史）
            async for evt in answer_stream(
                question=final_question,
                citations=citations,
                context_text=context_text,
                branch=branch,
                session_id=session_id
            ):
                if evt["type"] == "token":
                    yield "event: token\n"
                    text = evt["data"].replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')
                    yield f'data: {{"text":"{text}"}}\n\n'
                elif evt["type"] == "citation":
                    yield "event: citation\n"
                    yield f"data: {json.dumps(evt['data'], ensure_ascii=False)}\n\n"
                elif evt["type"] == "done":
                    used = "true" if evt["data"].get("used_retrieval") else "false"
                    yield "event: done\n"
                    yield f'data: {{"used_retrieval": {used}}}\n\n'

        except Exception as e:
            yield "event: error\n"
            esc = str(e).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')
            yield f'data: {{"message":"{esc}"}}\n\n'

    headers = {"Cache-Control": "no-cache, no-transform", "Connection": "keep-alive"}
    return StreamingResponse(gen(), media_type="text/event-stream", headers=headers)

# ---------------- Chat: 清除对话 ----------------
class ClearChatRequest(BaseModel):
    sessionId: Optional[str] = None

@app.post(f"{API_PREFIX}/chat/clear", tags=["Chat"])
async def chat_clear(req: ClearChatRequest):
    sid = (req.sessionId or "default").strip()
    clear_history(sid)
    return {"ok": True, "sessionId": sid, "cleared": True}


# ---------------- File: 上传（仅单文件，直接替换） ----------------

@app.post(f"{API_PREFIX}/pdf/upload", tags=["PDF"])
async def pdf_upload(file: UploadFile = File(...), replace: Optional[bool] = True):
    if not file:
        return JSONResponse(err("NO_FILE", "缺少文件"), status_code=400)
    # 生成新的 fileId（替换策略：上传即替换）
    fid = rid("f")
    saved = save_upload(fid, await file.read(), file.filename)
    current_file.update({**saved, "status": "idle", "progress": 0})
    citations.clear()
    return saved

# ---------------- PDF: 触发解析 ----------------
@app.post(f"{API_PREFIX}/pdf/parse", tags=["PDF"])
async def pdf_parse(payload: Dict[str, Any] = Body(...), bg: BackgroundTasks = None):
    file_id = payload.get("fileId")
    if not current_file["fileId"] or current_file["fileId"] != file_id:
        return JSONResponse(err("FILE_NOT_FOUND", "未找到该文件"), status_code=400)

    current_file["status"] = "parsing"
    current_file["progress"] = 5

    def _job():
        try:
            # 20 → 60 → 100 三阶段进度示意
            current_file["progress"] = 20
            run_full_parse_pipeline(file_id)   # 真解析
            current_file["progress"] = 100
            current_file["status"] = "ready"
        except Exception as e:
            current_file["status"] = "error"
            current_file["progress"] = 0
            print("Parse error:", e)

    if bg is not None:
        bg.add_task(_job)
    else:
        _job()

    return {"jobId": rid("j")}

# ---------------- PDF: 状态 ----------------
@app.get(f"{API_PREFIX}/pdf/status", tags=["PDF"])
async def pdf_status(fileId: str = Query(...)):
    if not current_file["fileId"] or current_file["fileId"] != fileId:
        return {"status": "idle", "progress": 0}
    resp = {"status": current_file["status"], "progress": current_file["progress"]}
    if current_file["status"] == "error":
        resp["errorMsg"] = "解析失败"
    return resp

# ---------------- PDF: 页面图 ----------------
@app.get(f"{API_PREFIX}/pdf/page", tags=["PDF"])
async def pdf_page(
    fileId: str = Query(...),
    page: int = Query(..., ge=1),
    type: str = Query(..., pattern="^(original|parsed)$")
):
    if not current_file["fileId"] or current_file["fileId"] != fileId:
        return JSONResponse(status_code=404, content=None)

    if current_file["status"] != "ready" and type == "parsed":
        # 未解析就请求 parsed 页，按你的契约可以给 400/403；这里保持 204 更温和
        return JSONResponse(status_code=204, content=None)

    base = dir_original_pages(fileId) if type == "original" else dir_parsed_pages(fileId)
    img = base / f"page-{page:04d}.png"
    if not img.exists():
        return JSONResponse(err("PAGE_NOT_FOUND", "页面不存在或未渲染"), status_code=404)
    return FileResponse(str(img), media_type="image/png")

# ---------------- PDF: 图片文件 ----------------
@app.get(f"{API_PREFIX}/pdf/images", tags=["PDF"])
async def pdf_images(
    fileId: str = Query(...),
    imagePath: str = Query(...)
):
    """获取PDF解析后的图片文件"""
    if not current_file["fileId"] or current_file["fileId"] != fileId:
        return JSONResponse(status_code=404, content=None)

    # 构建图片文件的完整路径
    from services.file_service import images_dir
    image_file = images_dir(fileId) / imagePath
    
    if not image_file.exists():
        return JSONResponse(err("IMAGE_NOT_FOUND", "图片文件不存在"), status_code=404)
    
    # 检查文件是否在images目录内（安全考虑）
    try:
        image_file.resolve().relative_to(images_dir(fileId).resolve())
    except ValueError:
        return JSONResponse(err("INVALID_PATH", "无效的图片路径"), status_code=400)
    
    return FileResponse(str(image_file), media_type="image/png")

# ---------------- PDF: 引用片段 ----------------
@app.get(f"{API_PREFIX}/pdf/chunk", tags=["PDF"])
async def pdf_chunk(citationId: str = Query(...)):
    ref = citations.get(citationId)
    if not ref:
        return JSONResponse(err("NOT_FOUND", "无该引用"), status_code=404)
    return ref

class BuildIndexRequest(BaseModel):
    fileId: str

class SearchRequest(BaseModel):
    fileId: str
    query: str
    k: Optional[int] = 5

@app.post(f"{API_PREFIX}/index/build", tags=["Index"])
async def index_build(req: BuildIndexRequest):
    if not current_file["fileId"] or current_file["fileId"] != req.fileId:
        raise HTTPException(status_code=400, detail="FILE_NOT_FOUND_OR_NOT_CURRENT")
    if current_file["status"] != "ready":
        raise HTTPException(status_code=409, detail="NEED_PARSE_FIRST")

    out = build_faiss_index(req.fileId)
    if not out.get("ok"):
        return JSONResponse(err(out.get("error", "INDEX_BUILD_ERROR"), "索引构建失败"), status_code=500)
    return {"ok": True, "chunks": out["chunks"]}

@app.post(f"{API_PREFIX}/index/search", tags=["Index"])
async def index_search(req: SearchRequest):
    out = search_faiss(req.fileId, req.query, req.k or 5)
    if not out.get("ok"):
        code = out.get("error", "INDEX_NOT_FOUND")
        return JSONResponse(err(code, "请先构建索引"), status_code=400)
    return out

# ---------------- Knowledge Base: 批量导入 ----------------
class BatchImportRequest(BaseModel):
    pass

@app.post(f"{API_PREFIX}/files/batch-import", tags=["Files"])
async def batch_import():
    """从 data/batch_import/ 目录触发批量导入。"""
    import_dir = Path("data") / "batch_import"
    if not import_dir.exists():
        return {"total": 0, "success": 0, "failed": 0, "results": [], "message": "batch_import 目录不存在"}

    results = []
    total = 0
    success = 0
    failed = 0

    # 先统计总数，排除 archive 目录中的文件
    file_list = [f for f in sorted(import_dir.iterdir()) if f.is_file() and not f.name.startswith(".")]
    total = len(file_list)

    archive_dir = import_dir / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)

    for idx, file_path in enumerate(file_list, 1):
        fid = rid("f")
        print(f"Importing file {idx}/{total}: {file_path.name}")
        try:
            file_bytes = file_path.read_bytes()
            save_upload(fid, file_bytes, file_path.name)
            run_full_parse_pipeline(fid)
            build_result = build_faiss_index(fid)
            if build_result.get("ok"):
                success += 1
                results.append({"fileId": fid, "fileName": file_path.name, "status": "success"})
                # 导入成功后将文件移至 archive 目录
                shutil.move(str(file_path), str(archive_dir / file_path.name))
            else:
                failed += 1
                results.append({"fileId": fid, "fileName": file_path.name, "status": "failed", "error": build_result.get("error")})
        except Exception as e:
            failed += 1
            results.append({"fileId": fid, "fileName": file_path.name, "status": "failed", "error": str(e)})

    print(f"Batch import complete: {success} success, {failed} failed out of {total} total")
    return {"total": total, "success": success, "failed": failed, "results": results}


# ---------------- Knowledge Base: 列表 ----------------
@app.get(f"{API_PREFIX}/knowledge-base/list", tags=["KnowledgeBase"])
async def kb_list():
    """列出所有可用的知识库。"""
    indexes = list_available_indexes()
    return {"knowledge_bases": indexes}


# ---------------- Knowledge Base: 跨库搜索 ----------------
class KbSearchRequest(BaseModel):
    query: str
    k: Optional[int] = 5

@app.post(f"{API_PREFIX}/knowledge-base/search", tags=["KnowledgeBase"])
async def kb_search(req: KbSearchRequest):
    """跨知识库搜索。"""
    results = search_all_indexes(req.query, req.k or 5)
    return {"results": results}


# ---------------- Knowledge Base: 删除 ----------------
@app.delete(f"{API_PREFIX}/knowledge-base/{{file_id}}", tags=["KnowledgeBase"])
async def kb_delete(file_id: str):
    """删除指定知识库，移除 data/{file_id} 目录。"""
    kb_dir = Path("data") / file_id
    if not kb_dir.exists():
        raise HTTPException(status_code=404, detail=f"知识库 {file_id} 不存在")
    shutil.rmtree(str(kb_dir))
    return {"status": "ok"}


# ---------------------- 统一知识库相关 API ----------------------

def _update_batch_status(current_step: str, **kwargs):
    """更新批量导入的状态信息"""
    global batch_import_status
    batch_import_status.update({
        "current_step": current_step,
        **kwargs
    })


def _process_kb_import_file(file_path: Path, fid: str) -> Dict[str, Any]:
    """处理单个知识库文件的内部函数"""
    try:
        # 1. 保存文件到工作目录
        _update_batch_status("保存文件中...", current_step_detail=f"正在保存 {file_path.name}")
        file_bytes = file_path.read_bytes()
        saved_info = save_upload(fid, file_bytes, file_path.name)
        
        # 2. 解析文件
        file_type = detect_file_type(file_path.name)
        _update_batch_status("解析文件中...", current_step_detail=f"正在解析 {file_path.name} ({file_type})")
        run_full_parse_pipeline(fid)
        
        # 3. 读取解析后的 Markdown
        _update_batch_status("读取解析结果...", current_step_detail=f"正在读取 {file_path.name} 的解析结果")
        md_path = Path("data") / fid / "output.md"
        if not md_path.exists():
            return {"success": False, "error": "Markdown output not found"}
        
        md_text = md_path.read_text(encoding="utf-8")
        
        # 4. 检测文件类型并分块
        _update_batch_status("文本分块中...", current_step_detail=f"正在对 {file_path.name} 进行文本分块")
        file_type = detect_file_type_from_index(fid)
        docs = split_markdown(md_text, file_type)
        
        if not docs:
            return {"success": False, "error": "No content extracted"}
        
        # 5. 去重
        _update_batch_status("去重处理中...", current_step_detail=f"正在对 {file_path.name} 进行内容去重")
        docs = _deduplicate_chunks(docs)
        
        if not docs:
            return {"success": False, "error": "All content is duplicate"}
        
        # 6. 添加到统一知识库索引
        _update_batch_status("向量化中...", current_step_detail=f"正在向量化 {file_path.name} ({len(docs)} 个块)")
        add_documents_to_unified_index(docs, file_path.name, fid)
        
        # 7. 更新文件元数据
        _update_batch_status("更新元数据...", current_step_detail=f"正在更新 {file_path.name} 的元数据")
        metadata = load_file_metadata()
        file_hash = get_file_hash(file_path)
        metadata["files"][fid] = {
            "name": file_path.name,
            "hash": file_hash,
            "imported_at": now_ts(),
            "chunk_count": len(docs)
        }
        save_file_metadata(metadata)
        
        return {"success": True, "chunk_count": len(docs)}
    
    except Exception as e:
        print(f"Error processing {file_path.name}: {e}")
        return {"success": False, "error": str(e)}


@app.post(f"{API_PREFIX}/unified-kb/import", tags=["Unified Knowledge Base"])
async def unified_kb_import(background_tasks: BackgroundTasks):
    """
    从 data/kb_import/ 目录批量导入文件到统一知识库，支持重复导入检测。
    """
    global batch_import_status
    
    # 检查是否已有导入任务在运行
    if batch_import_status["status"] == "running":
        return JSONResponse(
            err("IMPORT_IN_PROGRESS", "已有导入任务正在运行"),
            status_code=409
        )
    
    # 检查 kb_import 目录是否存在
    kb_import_dir = Path("data") / "kb_import"
    if not kb_import_dir.exists():
        kb_import_dir.mkdir(parents=True, exist_ok=True)
        return {
            "status": "idle",
            "message": "kb_import 目录不存在，已创建。请将文件放入该目录后重新导入。",
            "total": 0
        }
    
    # 获取待导入的文件列表
    file_list = [f for f in sorted(kb_import_dir.iterdir()) if f.is_file() and not f.name.startswith(".")]
    
    if not file_list:
        return {
            "status": "idle",
            "message": "kb_import 目录中没有待导入的文件",
            "total": 0
        }
    
    # 重置导入状态
    batch_import_status.update({
        "status": "running",
        "total": len(file_list),
        "processed": 0,
        "success": 0,
        "failed": 0,
        "results": [],
        "current_file": None,
        "error_message": None
    })
    
    def _background_import_task():
        global batch_import_status
        try:
            archive_dir = kb_import_dir / "archive"
            archive_dir.mkdir(parents=True, exist_ok=True)
            
            for file_path in file_list:
                batch_import_status["current_file"] = file_path.name
                
                # 检查文件是否已导入
                already_imported, existing_fid = check_file_already_imported(file_path)
                if already_imported:
                    batch_import_status["processed"] += 1
                    batch_import_status["results"].append({
                        "fileName": file_path.name,
                        "status": "skipped",
                        "reason": "File already imported",
                        "fileId": existing_fid
                    })
                    # 移动到归档目录
                    shutil.move(str(file_path), str(archive_dir / file_path.name))
                    continue
                
                # 处理文件
                fid = rid("kb")
                result = _process_kb_import_file(file_path, fid)
                
                batch_import_status["processed"] += 1
                
                if result["success"]:
                    batch_import_status["success"] += 1
                    batch_import_status["results"].append({
                        "fileId": fid,
                        "fileName": file_path.name,
                        "status": "success",
                        "chunkCount": result["chunk_count"]
                    })
                    # 移动到归档目录
                    shutil.move(str(file_path), str(archive_dir / file_path.name))
                else:
                    batch_import_status["failed"] += 1
                    batch_import_status["results"].append({
                        "fileId": fid,
                        "fileName": file_path.name,
                        "status": "failed",
                        "error": result["error"]
                    })
            
            batch_import_status["status"] = "completed"
            batch_import_status["current_file"] = None
        
        except Exception as e:
            batch_import_status["status"] = "error"
            batch_import_status["error_message"] = str(e)
            print(f"Batch import error: {e}")
    
    background_tasks.add_task(_background_import_task)
    
    return {
        "status": "started",
        "total": len(file_list),
        "message": "批量导入任务已在后台启动"
    }


@app.get(f"{API_PREFIX}/unified-kb/import/status", tags=["Unified Knowledge Base"])
async def unified_kb_import_status():
    """查询统一知识库批量导入的进度状态"""
    return batch_import_status


@app.post(f"{API_PREFIX}/unified-kb/search", tags=["Unified Knowledge Base"])
async def unified_kb_search(req: dict):
    """在统一知识库中搜索"""
    query = req.get("query", "")
    k = req.get("k", 5)
    if not query:
        return JSONResponse(err("EMPTY_QUERY", "查询内容不能为空"), status_code=400)
    
    return search_unified_index(query, k)


@app.get(f"{API_PREFIX}/unified-kb/info", tags=["Unified Knowledge Base"])
async def unified_kb_info():
    """获取统一知识库的基本信息"""
    metadata = load_file_metadata()
    vs = load_unified_index()
    
    file_count = len(metadata.get("files", {}))
    total_chunks = sum(info.get("chunk_count", 0) for info in metadata.get("files", {}).values())
    
    return {
        "file_count": file_count,
        "total_chunks": total_chunks,
        "files": metadata.get("files", {}),
        "index_exists": vs is not None
    }


