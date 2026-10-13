import React, { useState, useEffect, useRef } from "react";
import "./App.css";
import { fetchDocumentLibrary, uploadDocumentStream, askQuestionStream } from "./api";
import {
  FileText,
  Eye,
  Type,
  Upload,
  Send,
  Sparkles,
  Layers,
  Clock,
  Database,
  Trash2,
  Settings,
  HelpCircle,
  Menu,
  ChevronDown,
  Cpu,
  RefreshCw,
  Search,
  CheckCircle,
  FileCheck,
  ChevronRight,
  Table,
  BarChart2
} from "lucide-react";

export default function App() {
  const [pipelineMode, setPipelineMode] = useState("text"); // "text" or "vision"
  const [activeSessionId, setActiveSessionId] = useState(null);
  const [activeDoc, setActiveDoc] = useState(null);
  const [documentLibrary, setDocumentLibrary] = useState([]);
  const [retrievedSources, setRetrievedSources] = useState([]);
  const [recentQueries, setRecentQueries] = useState([]);

  // Accordion Sections (open/close like <details open>)
  const [openSections, setOpenSections] = useState({
    document: true,
    library: true,
    sources: true,
    recent: true,
  });

  // Chat State
  const [messages, setMessages] = useState([]);
  const [inputQuery, setInputQuery] = useState("");
  const [isInferencing, setIsInferencing] = useState(false);
  const [statusMessage, setStatusMessage] = useState("No document loaded");
  const [statusType, setStatusType] = useState("warning"); // "warning", "loading", "success"

  // Upload Overlay State
  const [uploadProgress, setUploadProgress] = useState(null);
  const [isDragging, setIsDragging] = useState(false);

  // Mobile sidebar toggle
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const chatEndRef = useRef(null);
  const fileInputRef = useRef(null);

  useEffect(() => {
    loadLibrary();
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isInferencing]);

  async function loadLibrary() {
    const docs = await fetchDocumentLibrary();
    setDocumentLibrary(docs);
  }

  function toggleSection(key) {
    setOpenSections((prev) => ({ ...prev, [key]: !prev[key] }));
  }

  function handleSelectDoc(doc) {
    const meta = doc.metadata || {};
    setActiveSessionId(doc.sessionId);
    setActiveDoc({
      filename: doc.filename || doc.doc_stem,
      doc_stem: doc.doc_stem,
      mode: doc.mode || "text",
      pages: meta.pages ?? "--",
      tables: meta.tables ?? "--",
      images: meta.images ?? "--",
      chunks: meta.chunks ?? "--",
    });
    setPipelineMode(doc.mode || "text");
    setStatusType("success");
    setStatusMessage(`Active: ${doc.filename || doc.doc_stem}`);
  }

  function handleClearChat() {
    setMessages([]);
    setRecentQueries([]);
    setStatusMessage(activeDoc ? "Ready" : "No document loaded");
  }

  function handleReplaceDocument() {
    setActiveSessionId(null);
    setActiveDoc(null);
    setRetrievedSources([]);
    setStatusType("warning");
    setStatusMessage("No document loaded");
  }

  async function handleFileProcess(file) {
    if (!file || !file.name.toLowerCase().endsWith(".pdf")) {
      alert("Please upload a PDF document.");
      return;
    }

    setUploadProgress({ progress: 5, status: "Staging document..." });
    setStatusType("loading");
    setStatusMessage("Processing document...");

    try {
      const res = await uploadDocumentStream(
        file,
        activeSessionId,
        pipelineMode,
        (progress, status) => {
          setUploadProgress({ progress, status });
        }
      );

      setActiveSessionId(res.sessionId);
      setActiveDoc({
        filename: file.name,
        doc_stem: file.name.replace(/\.[^/.]+$/, ""),
        mode: pipelineMode,
        pages: res.metadata?.pages ?? 0,
        tables: res.metadata?.tables ?? 0,
        images: res.metadata?.images ?? 0,
        chunks: res.metadata?.chunks ?? 0,
      });

      setUploadProgress(null);
      setStatusType("success");
      setStatusMessage("Document processing complete");
      
      const updatedDocs = await fetchDocumentLibrary();
      setDocumentLibrary(updatedDocs);
    } catch (err) {
      alert("Upload failed: " + err.message);
      setUploadProgress(null);
      setStatusType("warning");
      setStatusMessage("Upload failed");
    }
  }

  async function handleSendMessage(customPrompt) {
    const query = (customPrompt || inputQuery).trim();
    if (!query || isInferencing) return;

    if (!customPrompt) setInputQuery("");

    let currentSession = activeSessionId;
    if (!currentSession) {
      setMessages((prev) => [
        ...prev,
        { role: "user", content: query },
        {
          role: "assistant",
          content: "📄 **No active document selected.**\n\nPlease click **'Upload New PDF'** in the sidebar or select a document from the **Document Library** below to ask questions about it.",
          isStreaming: false,
        },
      ]);
      setStatusType("warning");
      setStatusMessage("Please select or upload a document");
      return;
    }

    // Add user message
    setMessages((prev) => [...prev, { role: "user", content: query }]);

    // Add assistant placeholder
    setMessages((prev) => [
      ...prev,
      { role: "assistant", content: "", isStreaming: true },
    ]);

    setIsInferencing(true);
    setStatusType("loading");
    setStatusMessage("Retrieving context & running offline model...");

    if (!recentQueries.includes(query)) {
      setRecentQueries((prev) => [query, ...prev.slice(0, 6)]);
    }

    try {
      await askQuestionStream(
        query,
        currentSession,
        pipelineMode,
        (event) => {
          if (event.type === "status") {
            setStatusMessage(event.content);
          } else if (event.type === "sources") {
            setRetrievedSources(event.content || []);
            setMessages((prev) => {
              const updated = [...prev];
              const last = updated[updated.length - 1];
              if (last && last.role === "assistant") {
                last.sources = event.content || [];
              }
              return updated;
            });
          } else if (event.type === "token") {
            setMessages((prev) => {
              const updated = [...prev];
              const last = updated[updated.length - 1];
              if (last && last.role === "assistant") {
                last.content = event.content;
              }
              return updated;
            });
          } else if (event.type === "done") {
            setMessages((prev) => {
              const updated = [...prev];
              const last = updated[updated.length - 1];
              if (last) last.isStreaming = false;
              return updated;
            });
            setIsInferencing(false);
            setStatusType("success");
            setStatusMessage("Ready");
          }
        }
      );
    } catch (err) {
      setMessages((prev) => {
        const updated = [...prev];
        const last = updated[updated.length - 1];
        if (last) {
          last.content += `\n\n⚠️ Error during inference: ${err.message}`;
          last.isStreaming = false;
        }
        return updated;
      });
      setIsInferencing(false);
      setStatusType("warning");
      setStatusMessage("Execution interrupted");
    }
  }

  // Render markdown text, images, and telemetry pills
  function renderFormattedContent(text) {
    if (!text) return null;

    // Check if text has telemetry footer: `⚡ Retrieval: ...`
    let bodyText = text;
    let telemetryData = null;

    const telemMatch = bodyText.match(/`⚡ Retrieval: (\d+)ms \| Rerank: (\d+)ms \| Generation: (\d+)ms \| Confidence: (HIGH|MEDIUM|LOW) \(([\d.]+)\)`/);
    if (telemMatch) {
      telemetryData = {
        retrieval: telemMatch[1],
        rerank: telemMatch[2],
        generation: telemMatch[3],
        tier: telemMatch[4],
        score: telemMatch[5],
      };
      bodyText = bodyText.replace(/---\s*`⚡ Retrieval:[^`]+`/, "").trim();
    }

    const imgRegex = /!\[(.*?)\]\((.*?)\)/g;
    const parts = [];
    let lastIndex = 0;
    let match;

    while ((match = imgRegex.exec(bodyText)) !== null) {
      if (match.index > lastIndex) {
        parts.push(bodyText.slice(lastIndex, match.index));
      }
      const alt = match[1];
      const rawUrl = match[2];
      const cleanUrl = rawUrl.startsWith("http")
        ? rawUrl
        : `http://127.0.0.1:8000/assets/${rawUrl.replace(/^[/\\]+/, "")}`;

      parts.push(
        <div key={match.index} className="chat-figure">
          <img
            src={cleanUrl}
            alt={alt || "Extracted Figure"}
            onClick={() => window.open(cleanUrl, "_blank")}
          />
          <div className="chat-figure-caption">
            🔍 {alt || "Extracted Document Figure"} (Click to enlarge)
          </div>
        </div>
      );
      lastIndex = match.index + match[0].length;
    }

    if (lastIndex < bodyText.length) {
      parts.push(bodyText.slice(lastIndex));
    }

    return (
      <div>
        {parts.map((p, idx) =>
          typeof p === "string" ? (
            <div key={idx} style={{ whiteSpace: "pre-wrap" }}>
              {p}
            </div>
          ) : (
            p
          )
        )}
        {telemetryData && (
          <div className="chat-telemetry-badge">
            <span>⚡</span>
            <span>Retrieval: <strong>{telemetryData.retrieval}ms</strong></span>
            <span>•</span>
            <span>Rerank: <strong>{telemetryData.rerank}ms</strong></span>
            <span>•</span>
            <span>Gen: <strong>{telemetryData.generation}ms</strong></span>
            <span>•</span>
            <span>Confidence:</span>
            <span className={`tier-pill ${telemetryData.tier.toLowerCase()}`}>
              {telemetryData.tier} ({telemetryData.score})
            </span>
          </div>
        )}
      </div>
    );
  }

  function MessageCitations({ sources, isStreaming }) {
    const [expanded, setExpanded] = useState(false);
    // Don't render citation section if no valid sources or while still actively streaming tokens
    if (!sources || sources.length === 0 || isStreaming) return null;

    // Additional deduplication safeguard by normalized text prefix
    const uniqueSources = [];
    const seenText = new Set();
    for (const s of sources) {
      const key = (s.text || "").slice(0, 150).trim();
      if (!seenText.has(key)) {
        seenText.add(key);
        uniqueSources.push(s);
      }
    }
    if (uniqueSources.length === 0) return null;

    return (
      <div className="message-citations">
        <button
          className="citations-toggle-btn"
          onClick={() => setExpanded(!expanded)}
          title="Click to view reference chunks"
        >
          {expanded ? "▼ Hide Sources" : `▶ View Sources (${uniqueSources.length})`}
        </button>
        {expanded && (
          <div className="citations-cards-grid">
            {uniqueSources.map((src, idx) => (
              <div key={idx} className="mini-citation-card">
                <div className="mini-citation-header">
                  <span>📄 Page {src.page ?? 1}</span>
                  <span className="mini-citation-score">
                    {src.score ? `${Math.round(src.score * 100)}% relevance` : ""}
                  </span>
                </div>
                <div className="mini-citation-text">{src.text}</div>
              </div>
            ))}
          </div>
        )}
      </div>
    );
  }

  return (
    <div
      className="app-container"
      onDragOver={(e) => {
        e.preventDefault();
        setIsDragging(true);
      }}
      onDragLeave={(e) => {
        if (!e.relatedTarget) setIsDragging(false);
      }}
      onDrop={(e) => {
        e.preventDefault();
        setIsDragging(false);
        if (e.dataTransfer.files[0]) handleFileProcess(e.dataTransfer.files[0]);
      }}
    >
      {/* Drop overlay */}
      {isDragging && (
        <div className="drop-full-overlay">
          <div className="drop-full-box">
            <Upload size={48} color="var(--primary)" />
            <h2>Drop PDF Document Here</h2>
            <p>Release to instantly parse, chunk, and index into the local vector space.</p>
          </div>
        </div>
      )}

      {/* Top Header */}
      <header className="app-header">
        <div className="brand">
          <button
            className="mobile-menu-btn"
            onClick={() => setSidebarOpen(!sidebarOpen)}
          >
            <Menu size={20} />
          </button>
          <div className="brand-icon">
            <Cpu size={20} color="#fff" />
          </div>
          <div>
            <div className="brand-title">NAYAN CHATBOT</div>
            <div className="brand-subtitle">Offline RAG v1.0</div>
          </div>
        </div>

        <div className="header-actions">
          <div className={`header-status ${statusType}`}>
            <div className={`status-dot ${statusType}`} />
            <span>{statusMessage}</span>
          </div>

          <button
            className="icon-action-btn"
            title="Clear Chat History"
            onClick={handleClearChat}
          >
            <Trash2 size={18} />
          </button>
        </div>
      </header>

      {/* Main Body */}
      <div className="app-body">
        {/* Sidebar */}
        <aside className={`app-sidebar ${sidebarOpen ? "open" : ""}`}>
          {/* SECTION 1: Document Upload & Active Stats */}
          <div className="sidebar-section">
            <div
              className="section-title"
              onClick={() => toggleSection("document")}
            >
              <div className="section-title-left">
                <FileText size={16} />
                <span>Document</span>
              </div>
              <ChevronDown
                size={16}
                className={`chevron ${openSections.document ? "open" : ""}`}
              />
            </div>

            {openSections.document && (
              <div className="section-content">
                {/* Pipeline Mode Switcher */}
                <div className="processing-toggle-container">
                  <span className="toggle-label">Pipeline Mode:</span>
                  <div className="toggle-switch-wrapper">
                    <button
                      className={`mode-pill ${pipelineMode === "text" ? "active" : ""}`}
                      onClick={() => setPipelineMode("text")}
                    >
                      Text Only
                    </button>
                    <button
                      className={`mode-pill ${pipelineMode === "vision" ? "active" : ""}`}
                      onClick={() => setPipelineMode("vision")}
                    >
                      Vision
                    </button>
                  </div>
                </div>

                {/* Upload Card - Always Available */}
                <input
                  id="react-pdf-file-input"
                  type="file"
                  ref={fileInputRef}
                  accept=".pdf"
                  style={{ display: "none" }}
                  onChange={(e) => {
                    if (e.target.files && e.target.files[0]) {
                      handleFileProcess(e.target.files[0]);
                      e.target.value = "";
                    }
                  }}
                />
                <label
                  htmlFor="react-pdf-file-input"
                  className="upload-card"
                  style={{ cursor: "pointer", display: "flex", marginBottom: 16 }}
                >
                  <div className="upload-icon-circle">
                    <Upload size={22} color="var(--primary)" />
                  </div>
                  <h3>Upload New PDF</h3>
                  <p>Drag &amp; drop or click anywhere to upload</p>
                </label>

                {/* Active Document Details (if loaded) */}
                {activeDoc && (
                  <div className="document-card">
                    <div className="document-header">
                      <FileCheck size={20} color="var(--success)" />
                      <div>
                        <div className="document-name">{activeDoc.filename}</div>
                        <span className="document-status-text">Active Selection</span>
                      </div>
                    </div>

                    <div className="stats-grid">
                      <div className="stat-box">
                        <span>Pages</span>
                        <strong>{activeDoc.pages ?? "--"}</strong>
                      </div>
                      <div className="stat-box">
                        <span>Tables</span>
                        <strong>{activeDoc.tables ?? "--"}</strong>
                      </div>
                      <div className="stat-box">
                        <span>Images</span>
                        <strong>{activeDoc.images ?? "--"}</strong>
                      </div>
                      <div className="stat-box">
                        <span>Chunks</span>
                        <strong>{activeDoc.chunks ?? "--"}</strong>
                      </div>
                    </div>

                    <div style={{ marginTop: 12 }}>
                      <button
                        className="secondary-btn"
                        onClick={handleReplaceDocument}
                      >
                        Clear Selection
                      </button>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>

          <hr className="separator" />

          {/* SECTION 2: Document Library */}
          <div className="sidebar-section">
            <div
              className="section-title"
              onClick={() => toggleSection("library")}
            >
              <div className="section-title-left">
                <Database size={16} />
                <span>Document Library ({documentLibrary.length})</span>
              </div>
              <ChevronDown
                size={16}
                className={`chevron ${openSections.library ? "open" : ""}`}
              />
            </div>

            {openSections.library && (
              <div className="section-content">
                <div className="library-list">
                  {documentLibrary.length === 0 ? (
                    <div className="placeholder-card">No stored document sessions.</div>
                  ) : (
                    documentLibrary.map((doc, idx) => (
                      <div
                        key={idx}
                        className={`library-item ${
                          activeDoc?.doc_stem === doc.doc_stem ? "active" : ""
                        }`}
                        onClick={() => handleSelectDoc(doc)}
                      >
                        <div className="library-item-meta">
                          <FileText size={16} color="var(--primary)" />
                          <div className="library-item-details">
                            <span className="library-item-title">
                              {doc.filename || doc.doc_stem}
                            </span>
                            <span
                              className={`library-item-badge ${
                                doc.mode === "vision" ? "badge-vision" : "badge-text"
                              }`}
                            >
                              {doc.mode || "text"} mode
                            </span>
                          </div>
                        </div>
                        <ChevronRight size={14} color="var(--text-dim)" />
                      </div>
                    ))
                  )}
                </div>
              </div>
            )}
          </div>

          <hr className="separator" />

          {/* SECTION 3: Retrieved Sources */}
          <div className="sidebar-section">
            <div
              className="section-title"
              onClick={() => toggleSection("sources")}
            >
              <div className="section-title-left">
                <Layers size={16} />
                <span>Retrieved Sources</span>
              </div>
              <ChevronDown
                size={16}
                className={`chevron ${openSections.sources ? "open" : ""}`}
              />
            </div>

            {openSections.sources && (
              <div className="section-content">
                <div className="source-list">
                  {retrievedSources.length === 0 ? (
                    <div className="placeholder-card">No retrieved sources yet.</div>
                  ) : (
                    retrievedSources.map((src, idx) => (
                      <div key={idx} className="source-item">
                        <div className="source-meta">
                          <span className="source-badge">
                            Source #{idx + 1} (Page {src.page ?? "N/A"})
                          </span>
                          <span className="source-score">
                            {src.score ? `${(src.score * 100).toFixed(1)}% match` : ""}
                          </span>
                        </div>
                        <p className="source-preview">{src.text}</p>
                      </div>
                    ))
                  )}
                </div>
              </div>
            )}
          </div>

          <hr className="separator" />

          {/* SECTION 4: Recent Queries */}
          <div className="sidebar-section">
            <div
              className="section-title"
              onClick={() => toggleSection("recent")}
            >
              <div className="section-title-left">
                <Clock size={16} />
                <span>Recent Queries</span>
              </div>
              <ChevronDown
                size={16}
                className={`chevron ${openSections.recent ? "open" : ""}`}
              />
            </div>

            {openSections.recent && (
              <div className="section-content">
                <div className="recent-list">
                  {recentQueries.length === 0 ? (
                    <div className="placeholder-card">No previous queries.</div>
                  ) : (
                    recentQueries.map((q, idx) => (
                      <div
                        key={idx}
                        className="recent-query"
                        onClick={() => {
                          setInputQuery(q);
                        }}
                      >
                        {q}
                      </div>
                    ))
                  )}
                </div>
              </div>
            )}
          </div>
        </aside>

        {/* Chat Main Area */}
        <main className="chat-main">
          <div className="chat-messages">
            {messages.length === 0 ? (
              <div className="welcome-screen">
                <div className="welcome-icon">
                  <Sparkles size={36} color="var(--primary)" />
                </div>
                <h2>Ask anything about your documents</h2>
                <p>
                  Upload a PDF and ask questions naturally. Everything runs locally
                  without sending your documents to the cloud.
                </p>

                <div className="welcome-features">
                  <button
                    className="feature-chip"
                    onClick={() =>
                      handleSendMessage("What is the primary objective of this document?")
                    }
                  >
                    <HelpCircle size={14} />
                    <span>Ask Questions</span>
                  </button>
                  <button
                    className="feature-chip"
                    onClick={() =>
                      handleSendMessage("Summarize the key findings and conclusions.")
                    }
                  >
                    <FileText size={14} />
                    <span>Summarize</span>
                  </button>
                  <button
                    className="feature-chip"
                    onClick={() =>
                      handleSendMessage("Extract all key data tables and their values.")
                    }
                  >
                    <Table size={14} />
                    <span>Extract Tables</span>
                  </button>
                  <button
                    className="feature-chip"
                    onClick={() =>
                      handleSendMessage("Find and explain the architectural diagrams.")
                    }
                  >
                    <Eye size={14} />
                    <span>Find Images</span>
                  </button>
                  <button
                    className="feature-chip"
                    onClick={() =>
                      handleSendMessage("Provide high-level insights from this analysis.")
                    }
                  >
                    <BarChart2 size={14} />
                    <span>Generate Insights</span>
                  </button>
                </div>
              </div>
            ) : (
              messages.map((msg, i) => (
                <div key={i} className={`message-row ${msg.role}`}>
                  <div className={`avatar ${msg.role}`}>
                    {msg.role === "user" ? (
                      <FileText size={16} />
                    ) : (
                      <Cpu size={16} color="#fff" />
                    )}
                  </div>
                  <div className="message-bubble">
                    {renderFormattedContent(msg.content)}
                    {msg.role === "assistant" && msg.sources && msg.sources.length > 0 && (
                      <MessageCitations sources={msg.sources} isStreaming={msg.isStreaming} />
                    )}
                  </div>
                </div>
              ))
            )}
            <div ref={chatEndRef} />
          </div>

          {/* Typing Indicator */}
          {isInferencing && (
            <div className="typing-indicator-bar">
              <div className="typing-dots">
                <span></span>
                <span></span>
                <span></span>
              </div>
              <span style={{ fontSize: "0.78rem", color: "var(--text-muted)" }}>
                Generating answer from local model...
              </span>
            </div>
          )}

          {/* Chat Input Bar */}
          <footer className="chat-input-area">
            <form
              onSubmit={(e) => {
                e.preventDefault();
                handleSendMessage();
              }}
              className="input-container"
            >
              <textarea
                className="chat-input"
                rows={1}
                placeholder={
                  activeDoc
                    ? "Ask a question about your processed document..."
                    : "Upload a document to start chatting..."
                }
                value={inputQuery}
                disabled={!activeDoc && !documentLibrary.length}
                onChange={(e) => setInputQuery(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    handleSendMessage();
                  }
                }}
              />
              <button
                type="submit"
                className="send-button"
                disabled={!inputQuery.trim() || isInferencing}
                aria-label="Send query"
              >
                <Send size={18} />
              </button>
            </form>

            <div className="chat-footer">
              <div className="footer-left">
                <CheckCircle size={14} color="var(--success)" />
                <span>Offline Mode &bull; Ready &bull; Ollama Connected</span>
              </div>
              <div className="footer-right">
                <span>100% Air-Gapped Local Ingestion</span>
              </div>
            </div>
          </footer>
        </main>
      </div>

      {/* Upload Progress Modal */}
      {uploadProgress && (
        <div className="loading-overlay">
          <div className="loading-card">
            <div className="spinner"></div>
            <h3 style={{ marginBottom: 6, fontSize: "1.05rem" }}>
              Processing Document
            </h3>
            <p style={{ fontSize: "0.82rem", color: "var(--text-muted)" }}>
              {uploadProgress.status}
            </p>
            <div className="progress-bar">
              <div
                className="progress-fill"
                style={{ width: `${uploadProgress.progress}%` }}
              />
            </div>
            <div className="progress-text">{uploadProgress.progress}%</div>
          </div>
        </div>
      )}
    </div>
  );
}
