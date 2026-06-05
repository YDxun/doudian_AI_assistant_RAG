import { useState } from "react";
import { Header } from "./components/Header";
import { ChatInterface } from "./components/ChatInterface";
import { Toaster } from "./components/ui/sonner";

export default function App() {
  const [chatKey, setChatKey] = useState(0);

  const handleClearChat = () => {
    setChatKey((prev) => prev + 1);
  };

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="h-screen flex flex-col">
        {/* Header */}
        <div className="max-w-3xl mx-auto w-full">
          <Header />
        </div>

        {/* Main Content - Centered single column */}
        <div className="flex-1 flex justify-center min-h-0 px-6 pb-6">
          <div className="w-full max-w-3xl min-h-0">
            <ChatInterface
              key={chatKey}
              onClearChat={handleClearChat}
            />
          </div>
        </div>
      </div>

      {/* Toast notifications */}
      <Toaster 
        position="top-right"
        expand={false}
        richColors
        closeButton
        theme="light"
      />
    </div>
  );
}
