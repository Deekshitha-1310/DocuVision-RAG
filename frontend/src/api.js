const API_BASE = "http://127.0.0.1:8000";

export async function fetchDocumentLibrary() {
  try {
    const res = await fetch(`${API_BASE}/documents`);
    if (!res.ok) return [];
    const data = await res.json();
    return data.documents || [];
  } catch (err) {
    console.error("Failed to load document library:", err);
    return [];
  }
}

export async function uploadDocumentStream(file, sessionId, mode, onProgress) {
  const activeSessionId = sessionId || crypto.randomUUID();
  const formData = new FormData();
  formData.append("file", file);

  const url = `${API_BASE}/upload?session_id=${encodeURIComponent(activeSessionId)}&mode=${encodeURIComponent(mode)}`;
  const response = await fetch(url, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`Upload failed with status: ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let partial = "";
  let finalMetadata = { name: file.name };

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;

    partial += decoder.decode(value, { stream: true });
    const lines = partial.split("\n");
    partial = lines.pop();

    for (const line of lines) {
      const clean = line.trim();
      if (!clean || !clean.startsWith("data:")) continue;

      try {
        const payload = JSON.parse(clean.replace(/^data:\s*/, ""));
        if (onProgress) {
          onProgress(payload.progress, payload.status);
        }
        if (payload.payload) {
          finalMetadata = { ...finalMetadata, ...payload.payload };
        }
      } catch (e) {
        console.warn("Error parsing upload chunk:", e);
      }
    }
  }

  // Process any leftover chunk in buffer
  if (partial && partial.trim().startsWith("data:")) {
    try {
      const payload = JSON.parse(partial.trim().replace(/^data:\s*/, ""));
      if (payload.payload) {
        finalMetadata = { ...finalMetadata, ...payload.payload };
      }
    } catch (e) {}
  }

  return { sessionId: activeSessionId, metadata: finalMetadata };
}

export async function askQuestionStream(query, sessionId, mode, onEvent) {
  const response = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, sessionId, mode }),
  });

  if (!response.ok) {
    throw new Error(`Chat error status: ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let partial = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;

    partial += decoder.decode(value, { stream: true });
    const lines = partial.split("\n");
    partial = lines.pop();

    for (const line of lines) {
      const clean = line.trim();
      if (!clean || !clean.startsWith("data:")) continue;

      try {
        const payload = JSON.parse(clean.replace(/^data:\s*/, ""));
        if (payload.status && payload.progress < 98) {
          onEvent({ type: "status", content: payload.status });
        }
        if ("sources" in payload) {
          onEvent({ type: "sources", content: payload.sources });
        }
        if ("answer" in payload) {
          onEvent({ type: "token", content: payload.answer });
        }
        if (payload.error) {
          throw new Error(payload.status || "Inference failed");
        }
        if (payload.done) {
          onEvent({ type: "done" });
        }
      } catch (e) {
        if (e.message && !e.message.startsWith("JSON")) {
          throw e;
        }
        console.warn("SSE frame error:", e);
      }
    }
  }
}
