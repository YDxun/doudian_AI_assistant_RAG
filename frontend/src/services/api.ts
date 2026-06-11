// API服务层 - 处理所有后端API调用
const API_BASE_URL = (import.meta as any).env?.VITE_API_BASE_URL || 'http://localhost:8001/api/v1';

export interface ChatReference {
  id: number;
  text: string;
  page: number;
  citationId?: string;
  rank?: number;
  snippet?: string;
}

// 健康检查
export async function checkHealth(): Promise<{ status: string }> {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    if (!response.ok) throw new Error('Health check failed');
    return response.json();
  } catch (error) {
    throw new Error('API unavailable');
  }
}

// 上传聊天图片附件（OCR提取文字 + 返回图片URL）
export async function uploadChatImage(file: File): Promise<{
  ok: boolean;
  fileName: string;
  fileType: string;
  extractedText: string;
  imageId: string;
  imageName: string;
  imageUrl: string;
}> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE_URL}/files/upload`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errData = await response.json().catch(() => ({}));
    throw new Error((errData as any).message || `Upload failed: ${response.statusText}`);
  }

  return response.json();
}

// 获取Citation详情
export async function getCitationChunk(citationId: string): Promise<{
  id: string;
  fileId: string;
  page: number;
  snippet: string;
  bbox: [number, number, number, number];
  previewUrl: string;
}> {
  const response = await fetch(`${API_BASE_URL}/pdf/chunk?citationId=${encodeURIComponent(citationId)}`);

  if (!response.ok) {
    throw new Error(`Citation fetch failed: ${response.statusText}`);
  }

  return response.json();
}

// SSE流式对话
export async function processChatStream(
  message: string,
  onToken: (text: string) => void,
  onCitation: (citation: {
    citation_id: string;
    fileId: string;
    rank: number;
    page: number;
    previewUrl: string;
    snippet?: string;
  }) => void,
  onDone: (data: { used_retrieval: boolean }) => void,
  onError: (error: string) => void,
  attachmentText?: string,
  sessionId = 'default',
) {
  try {
    const response = await fetch(`${API_BASE_URL}/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'text/event-stream',
      },
      body: JSON.stringify({
        message,
        sessionId,
        ...(attachmentText && { attachmentText }),
      }),
    });

    if (!response.ok) {
      throw new Error(`Chat request failed: ${response.statusText}`);
    }

    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error('No response body');
    }

    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      const events = buffer.split('\n\n');
      buffer = events.pop() || '';

      for (const event of events) {
        if (!event.trim()) continue;

        const lines = event.split('\n');
        let eventType = '';
        let eventData = '';

        for (const line of lines) {
          if (line.startsWith('event: ')) {
            eventType = line.substring(7);
          } else if (line.startsWith('data: ')) {
            eventData = line.substring(6);
          }
        }

        if (eventType && eventData) {
          try {
            const data = JSON.parse(eventData);

            switch (eventType) {
              case 'citation':
                onCitation(data);
                break;
              case 'token':
                onToken(data.text);
                break;
              case 'done':
                onDone(data);
                return;
              case 'error':
                onError(data.message || 'Unknown error');
                return;
            }
          } catch (e) {
            console.error('Failed to parse SSE data:', e);
          }
        }
      }
    }
  } catch (error) {
    if (error instanceof TypeError && error.message.includes('Failed to fetch')) {
      const mockResponse = `I understand you're asking about: "${message}".

Since the backend API is not currently available, I'm showing you a demonstration of the interface.

## Key Features Demonstrated:
- **Markdown rendering**: This response shows how text formatting works
- **Code blocks**: Here's an example:

\`\`\`javascript
function processDocument(content) {
  return content.split('\\n').map(line => ({
    text: line,
    analysis: performAnalysis(line)
  }));
}
\`\`\`

- **Reference citations**: This would normally include citations like [1] and [2] when connected to a real backend
- **Streaming responses**: Text appears progressively as it would from the AI

To see the full functionality, please start the backend server at \`localhost:8001\`.`;

      const words = mockResponse.split(' ');
      let currentIndex = 0;

      const streamInterval = setInterval(() => {
        if (currentIndex < words.length) {
          onToken(words[currentIndex] + ' ');
          currentIndex++;
        } else {
          clearInterval(streamInterval);
          onDone({ used_retrieval: false });
        }
      }, 50);

      return;
    }

    onError(error instanceof Error ? error.message : 'Unknown error');
  }
}

// 清空聊天会话
export async function clearSession(sessionId = 'default'): Promise<{
  ok: boolean;
  sessionId: string;
  cleared: boolean;
}> {
  const response = await fetch(`${API_BASE_URL}/chat/clear`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ sessionId }),
  });

  if (!response.ok) {
    throw new Error(`Clear session failed: ${response.statusText}`);
  }

  return response.json();
}
