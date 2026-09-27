import { useState } from "react";
import axios from "axios";

function App() {
  const [messages, setMessages] = useState([
    {
      sender: "bot",
      text: "👋 Hi! I'm your MCA Assistant. Ask me anything about your MCA course, subjects, syllabus, exams, or college documents.",
    },
  ]);

  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSend = async (e) => {
    e.preventDefault();

    if (!input.trim()) return;

    const newMessages = [
      ...messages,
      { sender: "user", text: input },
    ];

    setMessages(newMessages);
    setInput("");
    setLoading(true);

    try {
      const response = await axios.post(
        "https://mca-rag-assistant.onrender.com/ask",
        {
          question: input,
        }
      );

      const answer =
        response.data.answer ||
        "Sorry, I couldn't find that in the available MCA documents.";

      setMessages([
        ...newMessages,
        { sender: "bot", text: answer },
      ]);
    } catch (error) {
      console.error("❌ Backend error:", error);

      setMessages([
        ...newMessages,
        {
          sender: "bot",
          text: "⚠️ Error connecting to backend.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-gray-100">
      {/* Header */}
      <div className="bg-blue-600 text-white p-4 text-xl font-semibold shadow">
        🧠 MCA Assistant Chatbot
      </div>

      {/* Chat Area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-3">
        {messages.map((msg, idx) => (
          <div
            key={idx}
            className={`flex ${
              msg.sender === "user"
                ? "justify-end"
                : "justify-start"
            }`}
          >
            <div
              className={`max-w-lg px-4 py-2 rounded-2xl shadow text-sm ${
                msg.sender === "user"
                  ? "bg-blue-500 text-white rounded-br-none"
                  : "bg-white text-gray-800 rounded-bl-none"
              }`}
            >
              {msg.text}
            </div>
          </div>
        ))}

        {loading && (
          <div className="text-gray-500 italic text-sm">
            Thinking...
          </div>
        )}
      </div>

      {/* Input Box */}
      <form
        onSubmit={handleSend}
        className="flex items-center p-3 bg-white border-t border-gray-200"
      >
        <input
          type="text"
          className="flex-1 border rounded-full px-4 py-2 outline-none text-sm"
          placeholder="Ask about MCA subjects, syllabus, exams, etc..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
        />

        <button
          type="submit"
          className="ml-2 bg-blue-600 text-white px-4 py-2 rounded-full hover:bg-blue-700"
          disabled={loading}
        >
          Send
        </button>
      </form>
    </div>
  );
}

export default App;