import { useState, useRef, useEffect, useMemo } from "react";
import { Button } from "./ui/button";
import { Textarea } from "./ui/textarea";
import { ScrollArea } from "./ui/scroll-area";
import { Avatar, AvatarFallback } from "./ui/avatar";
import { Send, User, Bot, Sparkles, Paperclip, X, Loader2 } from "lucide-react";
import { MarkdownRenderer } from "./MarkdownRenderer";
import { processChatStream, clearSession, uploadPdf, startParse, waitForParseReady, buildIndex } from "../services/api";
import { toast } from "sonner";

type Reference = {
  id: number;
  text: string;
  page: number;
  citationId?: string;
  rank?: number;
  snippet?: string;
};

type Message = {
  id: string;
  type: "user" | "assistant";
  content: string;
  timestamp: Date;
  references?: Reference[];
  attachments?: string[];
};

type AttachedFile = {
  file: File;
  uploadProgress: number; // 0-100, -1 = uploaded, -2 = error
  fileId?: string;
};

export function ChatInterface({
  onClearChat,
}: {
  onClearChat: () => void;
}) {
  const initialAssistant =
    "您好，我是逗点生物食品分析专业智能体，您的科研助手。我可以帮助您解答关于产品的技术问题、使用方法、适用范围以及推荐合适的产品。";

  const [messages, setMessages] = useState<Message[]>([
    { id: "welcome", type: "assistant", content: initialAssistant, timestamp: new Date() },
  ]);
  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [attachedFiles, setAttachedFiles] = useState<AttachedFile[]>([]);

  // 当前流式生成中的内容与引用
  const [currentResponse, setCurrentResponse] = useState("");
  const [currentReferences, setCurrentReferences] = useState<Reference[]>([]);

  // refs 用于避免闭包 & 在 onDone 时拿到最新值
  const currentResponseRef = useRef("");
  const currentReferencesRef = useRef<Reference[]>([]);
  const citationIdsRef = useRef<Set<string>>(new Set());

  // 终止当前 SSE 的控制器
  const abortRef = useRef<AbortController | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const canSend = useMemo(() => input.trim().length > 0 && !isTyping, [input, isTyping]);

  // 自动滚动到底
  const scrollToBottom = () => messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  useEffect(() => {
    scrollToBottom();
  }, [messages, isTyping, currentResponse, currentReferences]);

  // 输入框自动高度
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = Math.min(textareaRef.current.scrollHeight, 120) + "px";
    }
  }, [input]);

  // 卸载时中断流
  useEffect(() => {
    return () => {
      abortRef.current?.abort();
    };
  }, []);

  const handleFileSelect = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    // Reset file input so the same file can be selected again
    if (fileInputRef.current) fileInputRef.current.value = '';

    const newFile: AttachedFile = { file, uploadProgress: 0 };
    setAttachedFiles(prev => [...prev, newFile]);

    // Start upload in background
    let progressInterval: NodeJS.Timeout | undefined;
    try {
      progressInterval = setInterval(() => {
        setAttachedFiles(prev => prev.map(f =>
          f === newFile && f.uploadProgress < 90
            ? { ...f, uploadProgress: f.uploadProgress + 15 }
            : f
        ));
      }, 200);

      const uploadResponse = await uploadPdf(file);
      if (progressInterval) clearInterval(progressInterval);

      // Start parsing
      await startParse(uploadResponse.fileId);
      
      // Wait for parsing to complete before building index
      setAttachedFiles(prev => prev.map(f =>
        f === newFile
          ? { ...f, uploadProgress: 50 }
          : f
      ));
      await waitForParseReady(uploadResponse.fileId);
      
      // Build index
      try {
        await buildIndex(uploadResponse.fileId);
      } catch {
        // Index build may fail but continue anyway
      }

      setAttachedFiles(prev => prev.map(f =>
        f === newFile
          ? { ...f, uploadProgress: -1, fileId: uploadResponse.fileId }
          : f
      ));
      toast.success(`文件 "${file.name}" 已上传并解析`);
    } catch (error) {
      if (progressInterval) clearInterval(progressInterval);
      setAttachedFiles(prev => prev.map(f =>
        f === newFile ? { ...f, uploadProgress: -2 } : f
      ));
      toast.error(`文件上传失败: ${file.name}`);
    }
  };

  const removeAttachment = (index: number) => {
    setAttachedFiles(prev => prev.filter((_, i) => i !== index));
  };

  const handleSend = async () => {
    if (!canSend && attachedFiles.length === 0) return;

    // 若有进行中的流，先中断
    abortRef.current?.abort();

    // 获取已上传的文件 fileId
    const uploadedFiles = attachedFiles.filter(f => f.fileId);
    // 优先使用最新上传的文件
    const fileIdForQuery = uploadedFiles.length > 0 ? uploadedFiles[uploadedFiles.length - 1].fileId : undefined;

    // 先落地用户消息（显示附件信息）
    const userMessage: Message = {
      id: Date.now().toString(),
      type: "user",
      content: input.trim() || "请分析附件内容",
      timestamp: new Date(),
      attachments: attachedFiles.map(f => f.file.name),
    };
    setMessages((prev) => [...prev, userMessage]);

    // 准备流式状态
    const userText = input.trim() || `请分析附件内容: ${attachedFiles.map(f => f.file.name).join(', ')}`;
    setInput("");
    setAttachedFiles([]);
    setIsTyping(true);
    setCurrentResponse("");
    setCurrentReferences([]);
    currentResponseRef.current = "";
    currentReferencesRef.current = [];
    citationIdsRef.current = new Set();

    try {
      await processChatStream(
        userText,
        // onToken
        (token: string) => {
          setCurrentResponse((prev) => prev + token);
          currentResponseRef.current += token;
        },
        // onCitation
        (c: {
          citation_id: string;
          fileId: string;
          rank: number;
          page: number;
          previewUrl: string;
          snippet?: string;
        }) => {
          if (!c.citation_id || citationIdsRef.current.has(c.citation_id)) return;
          citationIdsRef.current.add(c.citation_id);

          const newRef: Reference = {
            id: currentReferencesRef.current.length + 1,
            text: `第 ${c.page ?? "?"} 页相关内容`,
            page: c.page ?? 0,
            citationId: c.citation_id,
            rank: c.rank,
            snippet: c.snippet,
          };

          setCurrentReferences((prev) => [...prev, newRef]);
          currentReferencesRef.current = [...currentReferencesRef.current, newRef];
        },
        // onDone
        (meta: { used_retrieval: boolean }) => {
          const finalResponse = currentResponseRef.current;
          const finalRefs = [...currentReferencesRef.current];

          const assistantMessage: Message = {
            id: (Date.now() + 1).toString(),
            type: "assistant",
            content: finalResponse || "_（空响应）_",
            timestamp: new Date(),
            references: finalRefs.length ? finalRefs : undefined,
          };

          setMessages((prev) => [...prev, assistantMessage]);
          setIsTyping(false);
          setCurrentResponse("");
          setCurrentReferences([]);
          currentResponseRef.current = "";
          currentReferencesRef.current = [];
          citationIdsRef.current.clear();

          if (meta?.used_retrieval) {
            toast.success("已基于知识库生成回答");
          }
          textareaRef.current?.focus();
        },
        // onError
        (errText: string) => {
          console.error("Chat error:", errText);
          setIsTyping(false);
          setCurrentResponse("");
          setCurrentReferences([]);
          currentResponseRef.current = "";
          currentReferencesRef.current = [];
          citationIdsRef.current.clear();

          const errorMessage: Message = {
            id: (Date.now() + 1).toString(),
            type: "assistant",
            content: `抱歉，处理你的请求时出现错误：${errText}`,
            timestamp: new Date(),
          };
          setMessages((prev) => [...prev, errorMessage]);
          toast.error("获取回复失败");
        },
        // 传递 fileId
        fileIdForQuery,
      );
    } catch (e) {
      console.error("Chat request failed:", e);
      setIsTyping(false);
      setCurrentResponse("");
      setCurrentReferences([]);
      currentResponseRef.current = "";
      currentReferencesRef.current = [];
      citationIdsRef.current.clear();
      toast.error("Failed to send message");
    }
  };

  const clearChat = async () => {
    try {
      abortRef.current?.abort();
      await clearSession();
      setMessages([
        {
          id: "welcome",
          type: "assistant",
          content: initialAssistant,
          timestamp: new Date(),
        },
      ]);
      onClearChat();
      toast.success("聊天记录已清空");
    } catch (error) {
      if (error instanceof TypeError && String(error).includes("Failed to fetch")) {
        setMessages([
          {
            id: "welcome",
            type: "assistant",
            content: initialAssistant,
            timestamp: new Date(),
          },
        ]);
        onClearChat();
        toast.success("聊天记录已清空（本地）");
        return;
      }
      console.error("Failed to clear chat:", error);
      toast.error("清空聊天记录失败");
    } finally {
      textareaRef.current?.focus();
    }
  };

  return (
    <div className="glass-panel-bright h-full flex flex-col max-h-full relative overflow-hidden">
      {/* 背景装饰 */}
      <div className="absolute inset-0 opacity-5">
        <div className="absolute inset-0 bg-gradient-to-br from-primary/20 via-transparent to-purple-500/20"></div>
      </div>

      {/* 头部 */}
      <div className="relative p-6 border-b border-border/80 flex-shrink-0">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-primary/15 border border-primary/30 shadow-lg">
              <Sparkles className="w-5 h-5 text-primary" />
            </div>
            <div>
              <h2 className="elegant-title text-base">智能客服</h2>
              <p className="text-xs text-muted-foreground/80 mt-1">基于RAG技术</p>
            </div>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={clearChat}
            className="text-muted-foreground hover:text-white hover:bg-destructive/90 hover:border-destructive hover:shadow-lg hover:shadow-destructive/25 border border-border/60 transition-all duration-300 hover:scale-105 cursor-pointer"
          >
            清空
          </Button>
        </div>
      </div>

      {/* 消息区 */}
      <div className="flex-1 min-h-0 overflow-hidden relative">
        <ScrollArea className="h-full">
          <div className="p-6">
            <div className="space-y-4">
              {messages.map((m) => (
                <div key={m.id} className={`flex gap-4 ${m.type === "user" ? "justify-end" : "justify-start"}`}>
                  {m.type === "assistant" && (
                    <Avatar className="w-9 h-9 border-2 border-primary/30 flex-shrink-0 shadow-lg">
                      <AvatarFallback className="bg-gradient-to-br from-primary/15 to-purple-500/15">
                        <Bot className="w-5 h-5 text-primary" />
                      </AvatarFallback>
                    </Avatar>
                  )}

                  <div className={`max-w-[80%] ${m.type === "user" ? "order-first" : ""}`}>
                    <div
                      className={`p-4 rounded-2xl shadow-xl ${
                        m.type === "user"
                          ? "bg-gradient-to-br from-primary to-primary/80 text-primary-foreground ml-auto border border-primary/30"
                          : "bg-secondary/40 border border-border/40 backdrop-blur-sm"
                      }`}
                    >
                      {m.type === "user" ? (
                        <div className="space-y-2">
                          <p className="text-primary-foreground leading-relaxed text-base whitespace-pre-wrap">{m.content}</p>
                          {m.attachments && m.attachments.length > 0 && (
                            <div className="flex flex-wrap gap-2 pt-2 border-t border-primary/20">
                              {m.attachments.map((name, i) => (
                                <div key={i} className="flex items-center gap-1.5 px-2 py-1 rounded-lg text-xs bg-primary-foreground/20 border border-primary-foreground/30">
                                  <Paperclip className="w-3 h-3 text-primary-foreground/80" />
                                  <span className="max-w-32 truncate">{name}</span>
                                </div>
                              ))}
                            </div>
                          )}
                        </div>
                      ) : (
                        <MarkdownRenderer content={m.content} references={m.references} />
                      )}
                    </div>
                  </div>

                  {m.type === "user" && (
                    <Avatar className="w-9 h-9 border-2 border-border/40 flex-shrink-0 shadow-lg">
                      <AvatarFallback className="bg-gradient-to-br from-muted to-muted/80">
                        <User className="w-5 h-5" />
                      </AvatarFallback>
                    </Avatar>
                  )}
                </div>
              ))}

              {/* 正在生成中 */}
              {isTyping && (
                <div className="flex gap-4">
                  <Avatar className="w-9 h-9 border-2 border-primary/30 flex-shrink-0 shadow-lg">
                    <AvatarFallback className="bg-gradient-to-br from-primary/15 to-purple-500/15">
                      <Bot className="w-5 h-5 text-primary" />
                    </AvatarFallback>
                  </Avatar>
                  <div className="max-w-[80%]">
                    <div className="bg-secondary/40 border border-border/40 backdrop-blur-sm rounded-2xl p-4 shadow-xl">
                      {currentResponse ? (
                        <MarkdownRenderer content={currentResponse} references={currentReferences} />
                      ) : (
                        <div className="flex space-x-2">
                          <div className="w-2 h-2 bg-primary/70 rounded-full animate-bounce"></div>
                          <div className="w-2 h-2 bg-primary/70 rounded-full animate-bounce" style={{ animationDelay: "0.1s" }}></div>
                          <div className="w-2 h-2 bg-primary/70 rounded-full animate-bounce" style={{ animationDelay: "0.2s" }}></div>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>
          </div>
        </ScrollArea>
      </div>

      {/* 输入区 */}
      <div className="relative p-6 border-t border-border/60 flex-shrink-0 bg-card/40">
        {/* 附件标签 */}
        {attachedFiles.length > 0 && (
          <div className="flex flex-wrap gap-2 mb-3">
            {attachedFiles.map((f, i) => (
              <div
                key={i}
                className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs bg-primary/10 border border-primary/20"
              >
                <Paperclip className="w-3 h-3 text-primary" />
                <span className="max-w-32 truncate text-foreground">{f.file.name}</span>
                {f.uploadProgress >= 0 && (
                  <Loader2 className="w-3 h-3 text-primary animate-spin" />
                )}
                {f.uploadProgress === -1 && (
                  <span className="text-green-500">✓</span>
                )}
                {f.uploadProgress === -2 && (
                  <span className="text-red-500">✗</span>
                )}
                <button
                  onClick={() => removeAttachment(i)}
                  className="text-muted-foreground hover:text-destructive transition-colors"
                >
                  <X className="w-3 h-3" />
                </button>
              </div>
            ))}
          </div>
        )}

        <div className="flex gap-2 items-end">
          {/* 附件按钮 */}
          <Button
            variant="outline"
            size="sm"
            onClick={() => fileInputRef.current?.click()}
            disabled={isTyping}
            className="h-[52px] w-[52px] p-0 rounded-xl border-border/40 hover:bg-primary/10 transition-all duration-200 flex-shrink-0"
          >
            <Paperclip className="w-5 h-5" />
          </Button>

          <div className="relative flex-1">
            <Textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  handleSend();
                }
              }}
              placeholder="输入您的食品分析问题...（可点击左侧 📎 附加文件）"
              className="flex-1 bg-input/60 border-border/40 focus:border-primary/60 glow-ring text-foreground placeholder:text-muted-foreground/70 rounded-xl px-4 py-3 backdrop-blur-sm resize-none min-h-[52px] max-h-[120px] text-base leading-relaxed flex items-center"
              disabled={isTyping}
              rows={1}
            />
          </div>
          <Button
            onClick={handleSend}
            disabled={!canSend}
            className="bg-gradient-to-r from-primary to-primary/80 hover:from-primary/90 hover:to-primary/70 text-primary-foreground h-[52px] w-[52px] p-0 rounded-xl shadow-lg transition-all duration-200 border border-primary/30 flex-shrink-0"
          >
            <Send className="w-5 h-5" />
          </Button>
        </div>

        {/* 隐藏的文件输入 */}
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,.docx,.xlsx,.txt,.md,.png,.jpg,.jpeg,.bmp,.webp"
          onChange={handleFileSelect}
          className="hidden"
        />
      </div>
    </div>
  );
}
