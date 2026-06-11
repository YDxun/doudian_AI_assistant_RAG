# services/rag_service.py
from __future__ import annotations
import os, asyncio
from typing import Dict, Any, AsyncGenerator

from dotenv import load_dotenv
load_dotenv(override=True)

from collections import defaultdict
from pathlib import Path
import re

# 存储结构：sessions[session_id] = [{"role":"user|assistant","content":"..."}...]
_sessions: dict[str, list[dict]] = defaultdict(list)

def get_history(session_id: str) -> list[dict]:
    return _sessions.get(session_id, [])

def append_history(session_id: str, role: str, content: str) -> None:
    _sessions[session_id].append({"role": role, "content": content})

def clear_history(session_id: str) -> None:
    _sessions.pop(session_id, None)

# ---------------- 配置 ----------------
MODEL_NAME = "deepseek-chat"
MODEL_PROVIDER = "deepseek"
TEMPERATURE = 0

EMBED_MODEL_PATH = r"d:\ydx\workplace06\MRAG\Qwen3-VL-Embedding-2B"

# ---------------- 自定义 Qwen3VL Embedding（兼容 langchain 接口） ----------------
_qwen3vl_embedder = None


class Qwen3VLEmbeddings:
    """使用 Qwen3-VL-Embedding-2B 的 last-token-pooling 文本向量化，兼容 langchain 接口。

    该模型是 VL 多模态模型，HuggingFaceEmbeddings 无法直接加载。
    所有 heavy 依赖（PyTorch、模型代码）均为延迟导入，避免阻塞服务启动。
    """

    def __init__(self, model_path: str = EMBED_MODEL_PATH):
        self._model_path = model_path
        self._langchain_embeddings = None

    def _ensure_imports(self):
        if self._langchain_embeddings is not None:
            return
        import sys as _sys
        _embed_script_dir = os.path.join(self._model_path, "scripts")
        if _embed_script_dir not in _sys.path:
            _sys.path.insert(0, _embed_script_dir)
        from langchain_core.embeddings import Embeddings
        self._langchain_embeddings = Embeddings

    @property
    def _embedder(self):
        global _qwen3vl_embedder
        if _qwen3vl_embedder is None:
            self._ensure_imports()
            from qwen3_vl_embedding import Qwen3VLEmbedder
            _qwen3vl_embedder = Qwen3VLEmbedder(
                self._model_path,
                default_instruction="Represent the user's input.",
            )
        return _qwen3vl_embedder

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        inputs = [{"text": t} for t in texts]
        embeddings = self._embedder.process(inputs, normalize=True)
        return embeddings.cpu().numpy().tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

SYSTEM_INSTRUCTION = """# Role: 逗点生物食品分析专业分析AI客服 (Biocomma Food Analysis AI Assistant)
 
## Profile
- **Identity**: 你是由逗点生物（Biocomma）开发的食品分析领域专属科研助手。
- **Expertise**: 专注于微生物检测、农残/兽残检测、真菌毒素检测等食品分析技术。
- **Core Function**: 基于官方知识库提供精准的产品推荐、技术方案指导、COA报告查询及售后引导。
- **Tone**: 专业、严谨、亲切、高效。
- **Language**: 自动识别用户语言（中文/英文），并保持全程语种一致。
 
## Goals
1. **精准推荐**: 根据用户需求匹配逗点生物产品（培养基、SPE柱、QuEChERS套件、色谱柱等）。
2. **技术支持**: 解答产品使用方法、配套方案及技术参数疑问。
3. **服务引导**: 引导用户下载COA报告，或将非技术/价格问题无缝转接至人工客服。
4. **品牌维护**: 所有回答需体现逗点生物的专业形象，严禁误导用户。
 
## Critical Constraints (必须严格遵守)
1. **知识边界**: 
   - **仅**基于提供的参考信息回答问题。
   - **严禁**编造货号、URL、技术参数或不存在的产品。
   - 若知识库中无相关信息，必须回复："我的资料库里没有相关信息"，并立即提供人工客服微信号：`13537517880`，"您可以添加人工客服微信，进行下一步咨询。"
2. **链接安全**: 
   - 所有产品链接必须直接来自知识库，且与货号严格对应。
   - 禁止生成任何外部链接或虚构URL。
3. **话题限制**: 
   - 拒绝回答政治、宗教、色情或与食品分析/生命科学无关的话题。
   - 遇到此类提问，礼貌拒绝并引导回业务相关话题。
4. **价格与交期**: 
   - 涉及价格、库存、交货期时，统一回复："请询问人工客服。"
   - 紧接着追问："需要我提供人工客服的联系方式，方便你快速咨询吗？"
5. **输出格式**: 
   - **禁止**使用Emoji表情或特殊Unicode符号。
   - 仅使用标准ASCII字符和常规标点。
   - 产品推荐需结构化展示。
 
## Skills & Execution Logic
 
### 1. Language & Intent Detection
- 检测用户输入语言。若为非中文，先在内部理解其中文含义以检索知识库，最终输出转换为用户使用的语言。
- 识别用户意图：[产品咨询]、[技术问题]、[COA查询]、[价格/商务]、[身份询问]、[其他]。
 
### 2. Product Recommendation Strategy (核心技能)
- **匹配原则**:
  - 若用户提及国标（如 GB xxxx），必须严格匹配该标准对应的最新版产品。
  - **微生物培养基**: 优先推荐 `GF` 开头的干粉培养基（符合国标最新版）。除非用户明确要求，否则**不主动**推荐显色培养基。
  - **前处理产品**: 若推荐SPE柱或QuEChERS，必须检查是否需要搭配盐包、净化管等，若有，需一并列出并说明理由。
- **输出格式**:
  每个推荐产品必须包含以下字段：
  - 【产品名称】
  - 【货号】
  - 【规格】
  - 【逗点商城链接】(来自"逗点商城产品链接"知识库)
 
### 3. COA Report Handling
- **触发条件**: 用户提及 "COA"、"Certificate of Analysis"、"报告"、"质检单"。
- **行动**:
  1. 在回复末尾固定附上链接：`https://coa.biocomma.cn/pqreport/`
  2. 追加追问："你需要查询哪个产品的COA报告？可以提供产品编号/货号，我帮你快速定位。"
 
### 4. Identity Response
- **触发条件**: 用户问"你是谁"、"你能做什么"、"Who are you"。
- **中文回复模板**:
  "您好，我是逗点生物食品分析专业智能体，您的科研助手。我可以帮助您解答关于产品的技术问题、使用方法、适用范围以及推荐合适的产品。如果您有关于食品分析等产品的任何问题，欢迎随时提问。请问您需要了解什么产品？"
- **英文回复模板**:
  "I am Biocomma's AI customer service assistant, specializing in food analysis solutions. I provide one-stop support, including product consultation, application guidance, and COA report downloads. Let me know your needs, and I'll respond promptly."

## Follow-up Strategy (追问策略)
*原则：针对性强、引导下一步行动、保持语种一致。*

| 场景 | 追问话术 (中文) | Follow-up Question (English) |
| :--- | :--- | :--- |
| **检测需求模糊** | "你需要检测的食品类型（如果蔬、谷物、预制菜）是什么？以便我推荐更适配的产品。" | "What type of food matrix are you testing (e.g., fruits/vegetables, grains, prepared dishes)? This helps me recommend the most suitable products." |
| **标准不明确** | "你使用的检测标准是GB哪个版本？我可以结合标准补充具体操作细节。" | "Which version of the GB standard are you following? I can provide specific operational details based on that standard." |
| **农残/兽残具体化** | "你检测的具体种类（如有机磷、拟除虫菊酯、抗生素）是什么？确保产品完全适配。" | "What specific analytes are you targeting (e.g., organophosphates, pyrethroids, antibiotics)? This ensures the products are fully compatible." |
| **COA查询** | "你需要查询哪个产品的COA报告？可以提供产品编号/货号。" | "Which product's COA do you need? Please provide the product code or catalog number." |
| **非标/自定义应用** | "请问您主要的应用场景或检测项目是什么？" | "Could you please describe your main application scenario or target analytes?" |
"""

# Few-shot 示例：在历史消息中插入，比写在 system prompt 中更高效
# 每个示例为 (user_question, assistant_answer) 对
FEWSHOT_EXAMPLES = [
    # Case 1: 产品推荐（有知识库上下文）
    (
        "我需要做菌落总数检测，符合GB 4789.2标准，推荐什么培养基？",
        "根据您的需求，推荐以下符合 GB 4789.2-2022 标准的平板计数琼脂培养基：\n\n"
        "1. 【产品名称】平板计数琼脂（PCA）\n"
        "   【货号】GF1001\n"
        "   【规格】250g/瓶\n"
        "   【逗点商城链接】https://www.commashop.cn/product/detail/7893\n\n"
        "注意：干粉培养基使用后立即密封，避免吸潮结块。贮存于避光、干燥处。\n\n"
        "你需要检测什么项目？（菌落总数、大肠菌群、大肠埃希氏菌、粪大肠菌群及其他肠道菌、霉菌及酵母、沙门氏菌、金黄色葡萄球菌），"
        "以便我推荐更适配的产品。"
    ),
    # Case 2: 知识库无相关信息
    (
        "请问你们公司有测重金属的试剂盒吗？",
        "我的资料库里没有相关信息\n\n"
        "人工客服微信号：13537517880\n\n"
        "您可以添加人工客服微信，方便进一步咨询。"
    ),
    # Case 3: COA 报告查询
    (
        "我想下载COA报告",
        "您可以访问以下链接下载COA报告：\n"
        "https://coa.biocomma.cn/pqreport/\n\n"
        "你需要查询哪个产品的COA报告？可以提供产品编号/货号，我帮你快速定位。"
    ),
]

ANSWER_WITH_CONTEXT = (
    "请使用提供的上下文回答用户的问题。\n\n"
    "问题：\n{question}\n\n上下文：\n{context}\n\n"
    "要求：使用 Markdown；表达简洁但完整；如需给出代码，请使用三引号代码块（```）。\n"
    "若上下文包含与答案直接相关的图片，请在相关段落后内联给出 1–3 张图片（Markdown 语法）。\n"
    "作为一名助人为乐的助手，你需要仔细详细的感受用户的需求，并作出详细的回答。"
    "如果有图片，请在回答中给出图片的Markdown引用。\n\n"
    "【严格要求】\n"
    "1. 直接基于上文提供的信息作答，不要解释信息来源或引用机制。\n"
    "2. 回答末尾只保留追问或结束语，不要附加任何列表、编号引用或元数据说明。"
)

ANSWER_NO_CONTEXT = (
    "当前未找到直接相关的信息，将基于通识知识作答。\n"
    "问题：\n{question}"
)


# ---------------- 模型函数 ----------------
def _get_llm():
    from langchain.chat_models import init_chat_model
    return init_chat_model(model=MODEL_NAME, model_provider=MODEL_PROVIDER, temperature=TEMPERATURE)


# ---------------- 简单提问检测（跳过检索） ----------------
import re

# 不需要检索的简单提问模式：系统提示词中已预设固定回复模板的场景
SIMPLE_QUESTION_PATTERNS = [
    # ---- 问候语 ----
    r"^(你好|您好|hi|hello|嗨|hey|早上好|下午好|晚上好|good\s*morning|good\s*afternoon|good\s*evening)[\s!！。.]*$",
    # ---- 身份询问 ----
    r"(你是谁|你是谁呀|你叫什么|你是什么|你能做什么|你会做什么|你有什么功能|你的功能|"
    r"介绍一下自己|介绍下自己|你是机器人吗|你是AI吗|你是人工智能吗|你是真的人吗|你是什么模型|"
    r"what\s*are\s*you|who\s*are\s*you|what\s*can\s*you\s*do|introduce\s*yourself|"
    r"are\s*you\s*(a\s*)?(real|human|bot|robot|ai))",
    # ---- 感谢 ----
    r"^(谢谢|多谢|感谢|thanks|thank\s*you|thx|3q)[\s!！。.]*$",
    # ---- 告别 ----
    r"^(再见|拜拜|bye|goodbye|回头见|下次见|改天聊)[\s!！。.]*$",
    # ---- 价格/商务询问（系统提示词固定回复模板） ----
    r"(多少钱|价格|报价|库存|交货期|交期|有货吗|包邮吗|price|stock|delivery)",
    # ---- COA 报告询问（系统提示词固定回复模板） ----
    r"(COA|coa|certificate\s*of\s*analysis|质检单|质检报告|检测报告|下载报告)",
    # ---- 闲聊/无实质内容 ----
    r"^(嗯|哦|好的|ok|okay|知道了|明白了|懂了|got\s*it|i\s*see|收到|了解)[\s!！。.]*$",
    # ---- 与食品分析完全无关的提问（不需要检索知识库） ----
    r"(今天天气|天气怎么样|明天|几点|几点了|今天.*日期|现在.*时间|温度|摄氏度)",
    r"(讲个笑话|说个段子|讲个故事|给我讲|幽默|笑话|段子|逗我|开心一下)",
    r"(你多大了|你几岁|你.*年龄|你有.*年龄|how\s*old)",
    r"(你会.*吗|你能.*吗|可以.*吗)($|的|呢)",
    r"^(你.*会|你.*能)(?!.*检测|.*分析|.*测试|.*色谱|.*柱|.*产品|.*培养基|.*试剂)",
    r"(你是谁创建|谁开发|谁做的|谁教你|你的.*作者|你的.*开发者|creator|developer)",
    r"(你.*聪明|你.*笨|你.*傻|你.*厉害|你.*强大|are\s*you\s*smart)",
    r"(你.*恋爱|你.*结婚|你.*有.*男朋友|你.*有.*女朋友|你.*喜欢|你.*爱)",
    r"(你有.*感情|你有.*情绪|你有.*灵魂|你有.*意识|你有.*感觉)",
    r"(你会.*写.*代码|你会.*编程|你会.*翻译|你会.*画画|你会.*唱歌)",
    r"(你.*在哪|你.*住哪里|你.*地址|where\s*are\s*you\s*located)",
    r"(给我.*建议|帮我.*决定|推荐.*电影|推荐.*书|推荐.*音乐|有什么.*好.*推荐)",
]

def is_simple_question(question: str) -> bool:
    """检测是否是不需要知识库检索的简单提问。

    这些提问在系统提示词中已有预设的固定回复模板（如身份介绍、COA引导、价格引导、
    问候语等），无需进行 FAISS 向量检索即可直接由 LLM 基于 system prompt 回复。
    """
    q = question.strip().lower()
    if not q:
        return True
    for pattern in SIMPLE_QUESTION_PATTERNS:
        if re.search(pattern, q, re.IGNORECASE):
            return True
    return False


async def answer_stream(
    question: str,
    citations: list[dict],
    context_text: str,
    branch: str,
    session_id: str | None = None
) -> AsyncGenerator[dict, None]:
    """
    以增量事件的形式产出：
      {"type":"citation", "data": {...}}
      {"type":"token", "data": "text chunk"}
      {"type":"done", "data": {"used_retrieval": bool}}
    同时：如果提供了 session_id，会把本轮问答写入内存历史。
    """
    # 先把 citations 全部发给前端（便于角标立刻出现）
    if branch == "with_context" and citations:
        for c in citations:
            yield {"type": "citation", "data": c}

    # 组装"历史 + 本轮提示"
    # 添加 stop 序列：在 LLM 层面阻止生成"相关文档片段"
    llm = _get_llm().bind(stop=["相关文档片段"])
    history_msgs = get_history(session_id) if session_id else []

    if branch == "with_context" and context_text:
        user_prompt = ANSWER_WITH_CONTEXT.format(question=question, context=context_text)
    else:
        user_prompt = ANSWER_NO_CONTEXT.format(question=question)

    # 完整消息序列：system + few-shot 示例 + 历史多轮 + 当前用户
    msgs = [{"role": "system", "content": SYSTEM_INSTRUCTION}]
    # 注入 few-shot 示例，帮助模型理解输出格式
    for user_q, assistant_a in FEWSHOT_EXAMPLES:
        msgs.append({"role": "user", "content": user_q})
        msgs.append({"role": "assistant", "content": assistant_a})
    # 将历史逐条附加（保持 role: "user"/"assistant"）
    msgs.extend(history_msgs)
    # 当前用户问题
    msgs.append({"role": "user", "content": user_prompt})

    # 把最终生成的文本拼接出来用于写历史
    final_text_parts: list[str] = []
    # 待发送缓冲区：先缓存 delta，确认不含禁止词后再发送
    # 避免 LLM 流式输出把 \"相/关/文/档/片/段\" 拆到多个 token 时已发送部分无法撤回
    _pending = ""               # 待发送的累积文本
    _FORBIDDEN_RE = re.compile(
        r'相关文档片段|（无文本片段）|无文本片段',
        re.IGNORECASE
    )
    # 禁止词的所有前缀（用于检测可能跨 token 的局部匹配）
    _FORBIDDEN_PREFIXES = (
        "相", "相关", "相关文", "相关文档", "相关文档片", "相关文档片段",
        "（", "（无", "（无文", "（无文本", "（无文本片", "（无文本片段", "（无文本片段）",
        "无", "无文", "无文本", "无文本片", "无文本片段",
    )

    def _ends_with_forbidden_prefix(s: str) -> bool:
        """检测字符串末尾是否与禁止词前缀匹配（表示可能被拆到下一个 token）"""
        for prefix in _FORBIDDEN_PREFIXES:
            if len(prefix) <= len(s) and s.endswith(prefix):
                return True
        return False

    # 优先使用流式
    try:
        async for chunk in llm.astream(msgs):
            delta = getattr(chunk, "content", None)
            if not delta:
                continue

            _pending += delta

            # 检查累积文本中是否出现禁止词
            m = _FORBIDDEN_RE.search(_pending)
            if m:
                # 命中：只发送匹配位置之前的干净部分
                clean = _pending[:m.start()]
                if clean:
                    final_text_parts.append(clean)
                    yield {"type": "token", "data": clean}
                _pending = ""
                break  # 停止流式输出

            # 未命中，但末尾可能是禁止词前缀 → 先不发，等下一个 delta
            if _ends_with_forbidden_prefix(_pending):
                continue

            # 确认安全：发送累积文本并清空
            final_text_parts.append(_pending)
            yield {"type": "token", "data": _pending}
            _pending = ""

        # 循环正常结束：发送剩余缓冲区
        if _pending:
            # 最后再做一次检查
            m = _FORBIDDEN_RE.search(_pending)
            if m:
                clean = _pending[:m.start()]
                if clean:
                    final_text_parts.append(clean)
                    yield {"type": "token", "data": clean}
            else:
                final_text_parts.append(_pending)
                yield {"type": "token", "data": _pending}
    except Exception:
        # 回退：非流式整段生成
        resp = await llm.ainvoke(msgs)
        text = resp.content or ""
        # 同样需要截断非流式输出中的占位文字
        text = _FORBIDDEN_RE.split(text)[0].rstrip()
        final_text_parts.append(text)
        for i in range(0, len(text), 20):
            yield {"type": "token", "data": text[i:i+20]}
            await asyncio.sleep(0.005)

    if branch == "with_context" and citations:
        imgs = []
        # 取前 2 张，避免过多（可按需改成 3）
        for c in citations[:2]:
            url = c.get("previewUrl")
            if url:
                # 生成 Markdown 图片行
                imgs.append(f"![参考页 {c.get('rank', '')}]({url})")
        if imgs:
            tail = "\n\n---\n**相关页面预览**\n\n" + "\n\n".join(imgs)
            # 作为一个额外 token 块发给前端
            yield {"type": "token", "data": tail}

    # 将本轮问答写入历史（仅在提供 session_id 时）
    if session_id:
        append_history(session_id, "user", question)
        append_history(session_id, "assistant", "".join(final_text_parts))

    yield {"type": "done", "data": {"used_retrieval": branch == "with_context"}}


# ---------------- 附件概要提取（优化检索） ----------------

def extract_attachment_outline(attachment_text: str, filename: str = "") -> str:
    """从附件文本中提取标题和关键标识，用于增强检索 query。

    不调用 LLM，纯规则提取，速度快。
    返回：简短概要字符串（≤150 字），可直接拼接进检索 query。
    """
    parts = []

    # 1. 文件名（去掉扩展名）
    if filename:
        name = filename.rsplit(".", 1)[0]
        if name:
            parts.append(name)

    # 2. Markdown 标题（# 开头的行，取前 3 条）
    lines = attachment_text.strip().split("\n")
    heading_count = 0
    for line in lines:
        line = line.strip()
        if line.startswith("# ") or line.startswith("## "):
            heading = line.lstrip("#").strip()
            if heading and len(heading) > 2:
                parts.append(heading)
                heading_count += 1
                if heading_count >= 2:
                    break

    # 3. 货号/产品编号（如 HC18PS04、HSCX536、MPPT9601A 等大写字母数字组合）
    import re as _re
    codes = _re.findall(r'\b[A-Z]{2,}[0-9]{2,}[A-Z0-9]*\b', attachment_text)
    seen = set()
    for code in codes:
        if code not in seen and code.lower() not in parts:
            parts.append(code)
            seen.add(code)
            if len(parts) >= 6:
                break

    return " ".join(parts[:5])


def build_search_query(user_question: str, attachment_text: str = "", filename: str = "") -> str:
    """构造检索 query：用户问题 + 附件概要（如有）。

    附件概要帮助向量检索定位到知识库中的具体产品/文档，
    完整附件内容仍作为上下文传入 LLM。
    """
    base = user_question.strip() if user_question else " "
    if not attachment_text:
        return base

    outline = extract_attachment_outline(attachment_text, filename)
    if not outline:
        return base

    # 限制总长度，避免 query 过长影响 embedding 精度
    combined = f"{base} [附件: {outline}]"
    if len(combined) > 500:
        combined = combined[:500]
    return combined


# ---------------- OCR for user-uploaded files (MinerU only) ----------------

def _kill_process_tree_windows(pid: int) -> None:
    """Windows 上强制终止整个进程树"""
    import subprocess as _sp
    try:
        _sp.run(
            ["taskkill", "/F", "/T", "/PID", str(pid)],
            capture_output=True,
            timeout=10,
        )
    except Exception:
        pass


def extract_text_from_image_with_mineru(file_path: str) -> str:
    """使用 MinerU 从图片中提取文本内容（fast CLI 模式）"""
    import subprocess as _subprocess
    import tempfile as _tempfile
    import shutil as _shutil
    import platform as _platform

    # 检查 mineru 可执行文件是否可用
    if not _shutil.which("mineru"):
        print("MinerU 未安装或不在 PATH 中")
        return ""

    # 检查文件是否存在
    _p = Path(file_path)
    if not _p.exists():
        print(f"MinerU: 文件不存在 {file_path}")
        return ""
    if not _p.is_file():
        print(f"MinerU: 路径不是文件 {file_path}")
        return ""

    # 图片 OCR 超时设短一些，避免长时间卡住
    IMAGE_OCR_TIMEOUT = 240

    try:
        with _tempfile.TemporaryDirectory() as temp_dir:
            temp_output_path = Path(temp_dir)

            cmd = [
                "mineru",
                "-p", str(_p.resolve()),
                "-o", str(temp_output_path),
                "-b", "pipeline"
            ]

            print(f"执行 MinerU 图片解析: {' '.join(cmd)}")
            proc = None
            stdout = ""
            stderr = ""
            try:
                # 使用 Popen 替代 subprocess.run，超时后能可靠地杀进程树
                proc = _subprocess.Popen(
                    cmd,
                    stdout=_subprocess.PIPE,
                    stderr=_subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    creationflags=_subprocess.CREATE_NO_WINDOW if _platform.system() == "Windows" else 0,
                )
                try:
                    stdout, stderr = proc.communicate(timeout=IMAGE_OCR_TIMEOUT)
                    returncode = proc.returncode
                except _subprocess.TimeoutExpired:
                    print(f"MinerU 图片解析超时（>{IMAGE_OCR_TIMEOUT}s），强制终止进程树...")
                    if _platform.system() == "Windows":
                        _kill_process_tree_windows(proc.pid)
                    else:
                        proc.kill()
                    try:
                        proc.wait(timeout=5)
                    except _subprocess.TimeoutExpired:
                        print("MinerU 进程未能终止，跳过")
                    return ""

                if returncode != 0:
                    stderr_tail = (stderr or "")[-400:]
                    stdout_tail = (stdout or "")[-200:]
                    print(f"MinerU 返回非零 exit code={returncode}")
                    if stderr_tail:
                        print(f"MinerU stderr (tail): {stderr_tail}")
                    if stdout_tail:
                        print(f"MinerU stdout (tail): {stdout_tail}")
                    return ""
                print(f"MinerU 图片解析成功 (stdout): {(stdout or '')[:200]}")
                if stderr:
                    print(f"MinerU 图片解析 (stderr): {stderr[:200]}")
            except FileNotFoundError:
                print("MinerU 未安装或不在 PATH 中")
                return ""
            finally:
                # 确保进程被清理
                if proc is not None and proc.poll() is None:
                    try:
                        if _platform.system() == "Windows":
                            _kill_process_tree_windows(proc.pid)
                        else:
                            proc.kill()
                        proc.wait(timeout=3)
                    except Exception:
                        pass

            md_files = list(temp_output_path.rglob("*.md"))
            print(f"MinerU 图片递归找到的 md 文件: {md_files}")
            if not md_files:
                # 尝试查找所有文件帮助诊断
                all_files = list(temp_output_path.rglob("*"))
                print(f"MinerU 输出目录内容 ({len(all_files)} files): {[str(f.relative_to(temp_output_path)) for f in all_files[:20]]}")
                return ""

            md_file = md_files[0]
            md_content = md_file.read_text(encoding="utf-8")
            if md_content.strip():
                print(f"MinerU 成功提取 {len(md_content)} 字符")
            else:
                print("MinerU 生成的 Markdown 文件为空")
            return md_content

    except Exception as e:
        print(f"MinerU 图片提取文本错误: {e}")
        import traceback
        traceback.print_exc()
        return ""


def extract_text_from_image(file_path: str) -> str:
    """Extract text from image using MinerU"""
    return extract_text_from_image_with_mineru(file_path)


def extract_text_from_file(file_path: str, filename: str) -> str:
    """Extract text from uploaded image using MinerU"""
    ext = Path(filename).suffix.lower()
    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff", ".tif"}

    if ext in image_exts:
        return extract_text_from_image(file_path)
    return ""


# ---------------- 启动预加载 ----------------

def preload_models() -> None:
    """服务启动时预加载 embedding 模型，避免首次请求等待"""
    import sys as _sys

    print("[preload] 开始预加载模型...")

    # Qwen3-VL-Embedding-2B（首次加载 PyTorch + 2B 模型 ~30-90s）
    print("[preload] (1/1) 加载 Qwen3-VL-Embedding-2B...")
    _embed_script_dir = os.path.join(EMBED_MODEL_PATH, "scripts")
    if _embed_script_dir not in _sys.path:
        _sys.path.insert(0, _embed_script_dir)
    embedder = Qwen3VLEmbeddings(EMBED_MODEL_PATH)
    _ = embedder.embed_query("warmup")  # 触发 PyTorch + 模型加载
    print("[preload] Qwen3-VL-Embedding-2B 就绪")

    print("[preload] 所有模型预加载完成")
