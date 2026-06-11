import json
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
from fastapi import HTTPException

from services.index_service import (
    search_unified_index,
    load_unified_index,
)
from services.rag_service import (
    answer_stream,
    clear_history,
    extract_text_from_file,
    is_simple_question,
    build_search_query,
    preload_models,
)

app = FastAPI(
    title="逗点生物AI客服助手 API",
    version="1.0.0",
    description="逗点生物AI客服助手后端API，支持RAG问答。",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_PREFIX = "/api/v1"


@app.on_event("startup")
async def startup_preload():
    """启动时预加载模型和索引，避免首次请求等待"""
    import asyncio, sys
    loop = asyncio.get_event_loop()

    print("[startup] 预加载 FAISS 索引...")
    vs = await loop.run_in_executor(None, load_unified_index)
    if vs:
        print("[startup] FAISS 索引加载完成")
    else:
        print("[startup] FAISS 索引不存在，跳过")

    print("[startup] 预加载模型 (Embedding)...")
    await loop.run_in_executor(None, preload_models)

    print("[startup] 服务就绪，所有资源已预加载")


# ---------------- 内存态存储 ----------------
citations: Dict[str, Dict[str, Any]] = {}

# ---------------- 工具函数 ----------------
def err(code: str, message: str) -> Dict[str, Any]:
    return {"error": {"code": code, "message": message}}


# ---------------- 常量 ----------------
FETCH_COUNT = 10
SCORE_THRESHOLD = 0.5


def _filter_unified_search(search_result):
    """从统一知识库搜索结果中过滤低分切片"""
    citations_list = []
    if not (search_result.get("ok") and search_result.get("results")):
        return citations_list, "", False

    hits = [h for h in search_result["results"] if h["score"] >= SCORE_THRESHOLD]
    if not hits:
        print(f"Unified KB: all {len(search_result['results'])} results below threshold {SCORE_THRESHOLD}, "
              f"top_score={search_result['results'][0]['score']:.4f}")
        return citations_list, "", False

    ctx_snippets = []
    for i, hit in enumerate(hits, start=1):
        snippet_short = hit["text"][:500] + "..." if len(hit["text"]) > 500 else hit["text"]
        citations_list.append({
            "citation_id": f"unified-c{i}",
            "fileId": hit["metadata"].get("file_id", "unified"),
            "rank": i,
            "page": hit["metadata"].get("page"),
            "snippet": hit["text"][:4000],
            "score": hit["score"],
            "source_file": hit["metadata"].get("source_file"),
            "previewUrl": None,
        })
        ctx_snippets.append(snippet_short)

    context_text = "\n\n".join(ctx_snippets)
    print(f"Unified KB: {len(hits)}/{len(search_result['results'])} passed threshold, "
          f"top_score={hits[0]['score']:.4f}")
    return citations_list, context_text, True


# ---------------- Pydantic 模型 ----------------
class ChatRequest(BaseModel):
    message: str
    sessionId: Optional[str] = None
    attachmentText: Optional[str] = None


class ClearChatRequest(BaseModel):
    sessionId: Optional[str] = None


# ---------------- Health ----------------
@app.get(f"{API_PREFIX}/health", tags=["Health"])
async def health():
    vs = load_unified_index()
    return {"ok": True, "version": "1.0.0", "unified_kb_ready": vs is not None}


# ---------------- File: 上传聊天附件（仅支持图片，用于OCR识别） ----------------
UPLOAD_TEMP_DIR = Path("data") / "chat_uploads"
UPLOAD_TEMP_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif", ".tiff", ".tif"}

@app.post(f"{API_PREFIX}/files/upload", tags=["Files"])
async def file_upload(file: UploadFile = File(...)):
    """上传聊天图片附件，MinerU提取文字用于检索"""
    if not file:
        return JSONResponse(err("NO_FILE", "缺少文件"), status_code=400)

    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_IMAGE_EXTS:
        return JSONResponse(
            {"ok": False, "error": "UNSUPPORTED_TYPE",
             "message": "聊天附件仅支持图片格式（PNG/JPG/BMP/WEBP等），PDF/DOCX/XLSX 功能暂未开通"},
            status_code=400,
        )

    import uuid
    image_id = str(uuid.uuid4())[:12]
    saved_name = f"{image_id}{ext}"
    saved_path = UPLOAD_TEMP_DIR / saved_name

    content = await file.read()
    saved_path.write_bytes(content)

    ocr_text = ""
    try:
        ocr_text = extract_text_from_file(str(saved_path), file.filename or "")
        if ocr_text:
            print(f"Chat image OCR: extracted {len(ocr_text)} chars from {file.filename}")
        else:
            print(f"Chat image OCR: no text found in {file.filename}")
    except Exception as e:
        print(f"Chat image OCR failed: {e}")

    image_url = f"{API_PREFIX}/files/image/{saved_name}"

    return {
        "ok": True,
        "fileName": file.filename,
        "fileType": ext.lstrip("."),
        "extractedText": ocr_text,
        "imageId": image_id,
        "imageName": saved_name,
        "imageUrl": image_url,
    }


@app.get(f"{API_PREFIX}/files/image/{{image_name}}", tags=["Files"])
async def serve_uploaded_image(image_name: str):
    """提供已上传聊天图片的静态访问"""
    import re as _re
    if not _re.match(r'^[a-zA-Z0-9._-]+$', image_name):
        raise HTTPException(status_code=400, detail="Invalid image name")
    image_path = UPLOAD_TEMP_DIR / image_name
    if not image_path.exists() or not image_path.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(str(image_path))


# ---------------- Chat（SSE） ----------------
@app.post(f"{API_PREFIX}/chat", tags=["Chat"])
async def chat_stream(req: ChatRequest):
    """SSE 事件：token | citation | done | error"""
    async def gen():
        try:
            user_question = (req.message or "").strip()
            session_id = (req.sessionId or "default").strip()
            attachment_text = (req.attachmentText or "").strip()

            search_query = build_search_query(user_question, attachment_text)

            # 简单提问无需检索
            skip_retrieval = (
                not attachment_text
                and is_simple_question(user_question)
            )

            citations_list, context_text = [], ""
            branch = "no_context"

            if not skip_retrieval:
                try:
                    search_result = search_unified_index(search_query, k=FETCH_COUNT)
                    unified_citations, context_text, branch_ok = _filter_unified_search(search_result)
                    if branch_ok:
                        citations_list.extend(unified_citations)
                        branch = "with_context"
                except Exception as e:
                    print(f"Unified KB search error: {e}")

            if attachment_text:
                final_question = (
                    f"【附件内容】\n{attachment_text}\n\n"
                    f"【用户问题】\n{user_question}"
                )
            else:
                final_question = user_question

            # 推送引用
            if branch == "with_context" and citations_list:
                _global_citations = globals()["citations"]
                for c in citations_list:
                    _global_citations[c["citation_id"]] = c
                for c in citations_list:
                    yield "event: citation\n"
                    yield f"data: {json.dumps(c, ensure_ascii=False)}\n\n"

            # 推送 token 流
            async for evt in answer_stream(
                question=final_question,
                citations=citations_list,
                context_text=context_text,
                branch=branch,
                session_id=session_id,
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
@app.post(f"{API_PREFIX}/chat/clear", tags=["Chat"])
async def chat_clear(req: ClearChatRequest):
    sid = (req.sessionId or "default").strip()
    clear_history(sid)
    return {"ok": True, "sessionId": sid, "cleared": True}


# ---------------- Citation 查询 ----------------
@app.get(f"{API_PREFIX}/pdf/chunk", tags=["Chat"])
async def pdf_chunk(citationId: str = Query(...)):
    ref = citations.get(citationId)
    if not ref:
        return JSONResponse(err("NOT_FOUND", "无该引用"), status_code=404)
    return ref


# ---------------- 统一知识库查询 ----------------
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
    metadata_file = Path("data") / "unified_knowledge_base" / "metadata.json"
    files_info = {}
    total_chunks = 0
    if metadata_file.exists():
        try:
            metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
            files_info = metadata.get("files", {})
            total_chunks = sum(info.get("chunk_count", 0) for info in files_info.values())
        except Exception:
            pass

    vs = load_unified_index()
    return {
        "file_count": len(files_info),
        "total_chunks": total_chunks,
        "files": files_info,
        "index_exists": vs is not None,
    }
