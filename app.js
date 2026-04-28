// app.js
// Local AI front-end
// Conversation storage is entirely server-side.
// localStorage is used only for UI preferences (last model, theme).

"use strict";

// ---------------------------------------------------------------------------
// marked.js configuration
// breaks: false — single newlines do NOT become <br>
// Use marked.use() (stable v2–v14). setOptions is deprecated in v9+.
// Wrapped in try-catch: a marked version mismatch must NEVER kill the app.
// ---------------------------------------------------------------------------
if (typeof marked !== "undefined") {
    try {
        if (typeof marked.use === "function") {
            marked.use({ breaks: false });
        } else if (typeof marked.setOptions === "function") {
            marked.setOptions({ breaks: false });
        }
    } catch (e) {
        console.warn("marked.js configuration error (non-fatal):", e);
    }
}

// Strip <think>…</think> blocks that the reasoning parser occasionally leaks
// into the visible content field instead of the reasoning_content field.
function stripThinkTags(text) {
    if (!text) return text;
    return text
        .replace(/<think>[\s\S]*?<\/think>/gi, "")
        .replace(/^\s+/, "");   // trim leading whitespace left behind
}

// ---------------------------------------------------------------------------
// DOM refs
// ---------------------------------------------------------------------------
const messagesDiv      = document.getElementById("messages");
const promptBox        = document.getElementById("prompt");
const modelSelect      = document.getElementById("model");
const modelInfoDiv     = document.getElementById("modelInfo");
const threadsList      = document.getElementById("threads");
const statusDiv        = document.getElementById("status");
const uploadBtn        = document.getElementById("uploadBtn");
const fileInput        = document.getElementById("fileInput");
const thinkingToggle   = document.getElementById("thinkingMode");
const thinkingStatus   = document.getElementById("thinkingStatus");
const attachedFilesDiv = document.getElementById("attachedFiles");
const toolAutoMode     = document.getElementById("toolAutoMode");
const toolListDiv      = document.getElementById("toolList");
const toolStatusDiv    = document.getElementById("toolStatus");

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let convMap         = new Map();   // conv_id → {id, display_name, created_at}
let convOrder       = [];
let currentConvId   = null;
let currentAttachments = [];
let models          = [];
let modelsById      = {};
let identity        = "";
let _renameActive   = false;
let availableTools  = [];          // [{id, name, description}] from /tools

// ---------------------------------------------------------------------------
// WebSocket
// ---------------------------------------------------------------------------
let ws = null;
function connectWS() {
  ws = new WebSocket("ws://127.0.0.1:8765");
  ws.onopen  = () => console.log("WS connected");
  ws.onclose = () => { setTimeout(connectWS, 3000); };
}
connectWS();

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------
function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;")
    .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

function getFileIcon(filename) {
  const ext = (filename.split(".").pop() || "").toLowerCase();
  if (["png","jpg","jpeg","gif","webp","bmp","tiff"].includes(ext)) return "🖼️";
  if (ext === "pdf")  return "📄";
  if (["xlsx","xls"].includes(ext)) return "📊";
  if (["docx","doc"].includes(ext)) return "📝";
  if (["pptx","ppt"].includes(ext)) return "📽️";
  if (["mp4","avi","mov","mkv"].includes(ext)) return "🎥";
  return "📎";
}

function showBanner(message, type = "info") {
  const banner = document.createElement("div");
  banner.className = `banner ${type}-banner`;
  banner.textContent = message;
  messagesDiv.parentNode.insertBefore(banner, messagesDiv);
  setTimeout(() => banner.remove(), 6000);
}

function scrollBottom() {
  messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

// ---------------------------------------------------------------------------
// Phase indicator
// ---------------------------------------------------------------------------
let _phaseEl    = null;
let _phaseTimer = null;
const PHASES = [
  { delay: 0,     text: "⏳ Interpreting your request…" },
  { delay: 4000,  text: "🧠 Thinking…"                  },
  { delay: 10000, text: "✍️  Preparing answer…"          },
];

function startPhaseIndicator() {
  _phaseEl = document.createElement("div");
  _phaseEl.className   = "phase-indicator";
  _phaseEl.textContent = PHASES[0].text;
  messagesDiv.appendChild(_phaseEl);
  scrollBottom();
  _phaseTimer = [];
  PHASES.slice(1).forEach(({ delay, text }) => {
    _phaseTimer.push(setTimeout(() => { if (_phaseEl) _phaseEl.textContent = text; }, delay));
  });
}

function stopPhaseIndicator() {
  if (_phaseTimer) { _phaseTimer.forEach(clearTimeout); _phaseTimer = null; }
  if (_phaseEl)    { _phaseEl.remove(); _phaseEl = null; }
}

// ---------------------------------------------------------------------------
// Tool panel
// ---------------------------------------------------------------------------

async function loadTools() {
  try {
    const data = await apiGet("/tools");
    availableTools = data.builtin_tools || [];
    renderToolList();
  } catch (e) {
    console.warn("Could not load tools:", e);
  }
}

function renderToolList() {
  toolListDiv.innerHTML = "";
  availableTools.forEach(tool => {
    const row = document.createElement("label");
    row.className = "tool-item";
    row.innerHTML = `
      <input type="checkbox" class="tool-checkbox" value="${escapeHtml(tool.id)}">
      <span class="tool-name">${escapeHtml(tool.name)}</span>
      <span class="tool-desc" title="${escapeHtml(tool.description)}">ℹ️</span>
    `;
    toolListDiv.appendChild(row);
  });
}

// When Auto mode is toggled, enable/disable individual tool checkboxes.
toolAutoMode.onchange = () => {
  const isAuto = toolAutoMode.checked;
  toolListDiv.classList.toggle("tool-list-disabled", isAuto);
  toolListDiv.querySelectorAll(".tool-checkbox").forEach(cb => {
    cb.disabled = isAuto;
    if (isAuto) cb.checked = false;
  });
  toolStatusDiv.textContent = isAuto
    ? ""
    : "Select the tools the AI may use for this conversation.";
};

function getSelectedTools() {
  if (toolAutoMode.checked) return [];   // empty = auto mode
  const checked = [];
  toolListDiv.querySelectorAll(".tool-checkbox:checked").forEach(cb => {
    checked.push(cb.value);
  });
  return checked;
}

// ---------------------------------------------------------------------------
// Render a message bubble
// ---------------------------------------------------------------------------
function renderMessage(role, content, reasoning, toolCalls, timestamp) {
  const wrap = document.createElement("div");
  wrap.className = `message-wrap ${role}-wrap`;

  if (timestamp) {
    const ts = document.createElement("div");
    ts.className   = "message-timestamp";
    ts.textContent = timestamp;
    wrap.appendChild(ts);
  }

  if (reasoning) {
    const details = document.createElement("details");
    details.className = "reasoning-block";
    const summary = document.createElement("summary");
    summary.textContent = "▶ Reasoning";
    const pre = document.createElement("pre");
    pre.textContent = reasoning;
    details.appendChild(summary);
    details.appendChild(pre);
    wrap.appendChild(details);
  }

  if (toolCalls && toolCalls.length > 0) {
    toolCalls.forEach(tc => {
      const details = document.createElement("details");
      details.className = "tool-block";
      const summary = document.createElement("summary");
      summary.textContent = `▶ Tool used: ${tc.function?.name || tc.name || "tool"}`;
      const pre = document.createElement("pre");
      pre.textContent = JSON.stringify(tc, null, 2);
      details.appendChild(summary);
      details.appendChild(pre);
      wrap.appendChild(details);
    });
  }

  const bubble = document.createElement("div");
  bubble.className = `message-bubble ${role}-bubble`;
  const label = document.createElement("strong");
  label.textContent = role === "user" ? "You:" : "AI:";
  bubble.appendChild(label);
  const body = document.createElement("div");
  body.className = "message-body";
  body.innerHTML = marked.parse(stripThinkTags(content) || "");
  bubble.appendChild(body);
  wrap.appendChild(bubble);

  messagesDiv.appendChild(wrap);
  scrollBottom();
}

// ---------------------------------------------------------------------------
// API helpers
// ---------------------------------------------------------------------------
async function apiGet(path) {
  const r = await fetch(`http://127.0.0.1:8770${path}`);
  if (!r.ok) throw new Error(`GET ${path} → ${r.status}`);
  return r.json();
}
async function apiPost(path, body) {
  const r = await fetch(`http://127.0.0.1:8770${path}`, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`POST ${path} → ${r.status}`);
  return r.json();
}
async function apiDelete(path) {
  const r = await fetch(`http://127.0.0.1:8770${path}`, { method: "DELETE" });
  if (!r.ok) throw new Error(`DELETE ${path} → ${r.status}`);
  return r.json();
}

// ---------------------------------------------------------------------------
// Load a conversation's messages
// ---------------------------------------------------------------------------
async function loadConversation(convId) {
  currentConvId = convId;
  messagesDiv.innerHTML = "";
  currentAttachments = [];
  updateAttachmentDisplay();
  refreshThreadList();
  try {
    const data = await apiGet(`/conversations/${convId}/messages`);
    (data.messages || []).forEach(msg => {
      renderMessage(
        msg.role, msg.content,
        msg.reasoning_content || null, null,
        msg.timestamp || null,
      );
    });
  } catch (e) {
    showBanner("Failed to load conversation messages.", "error");
  }
}

// ---------------------------------------------------------------------------
// Thread list — stable, never rebuilds during active rename
// ---------------------------------------------------------------------------
function refreshThreadList() {
  if (_renameActive) return;
  threadsList.innerHTML = "";
  convOrder.forEach(convId => {
    const conv = convMap.get(convId);
    if (!conv) return;

    const li = document.createElement("li");
    li.className    = "thread-item" + (convId === currentConvId ? " active-thread" : "");
    li.dataset.convId = convId;

    const nameSpan = document.createElement("span");
    nameSpan.className   = "thread-name";
    nameSpan.textContent = conv.display_name;
    nameSpan.title       = conv.created_at ? `Created: ${conv.created_at}` : "";
    nameSpan.onclick     = () => loadConversation(convId);
    nameSpan.ondblclick  = (e) => { e.stopPropagation(); startInlineRename(li, nameSpan, conv); };

    const renameBtn = document.createElement("button");
    renameBtn.textContent = "✏️";
    renameBtn.className   = "thread-rename-btn";
    renameBtn.title       = "Rename (or double-click the name)";
    renameBtn.onclick = (e) => { e.stopPropagation(); startInlineRename(li, nameSpan, conv); };

    const deleteBtn = document.createElement("button");
    deleteBtn.textContent = "🗑️";
    deleteBtn.className   = "thread-delete";
    deleteBtn.title       = "Delete conversation";
    deleteBtn.onclick = async (e) => { e.stopPropagation(); await deleteConversation(convId); };

    li.appendChild(nameSpan);
    li.appendChild(renameBtn);
    li.appendChild(deleteBtn);
    threadsList.appendChild(li);
  });
}

// ---------------------------------------------------------------------------
// Inline rename
// ---------------------------------------------------------------------------
function startInlineRename(li, nameSpan, conv) {
  _renameActive = true;
  const input = document.createElement("input");
  input.type      = "text";
  input.value     = conv.display_name;
  input.className = "thread-rename-input";

  const commit = async () => {
    const newName = input.value.trim();
    _renameActive = false;
    if (!newName || newName === conv.display_name) {
      li.replaceChild(nameSpan, input); return;
    }
    try {
      await apiPost(`/conversations/${conv.id}/rename`, { display_name: newName });
      conv.display_name    = newName;
      nameSpan.textContent = newName;
    } catch (e) {
      showBanner("Rename failed.", "error");
    }
    li.replaceChild(nameSpan, input);
    refreshThreadList();
  };

  input.onblur    = commit;
  input.onkeydown = (e) => {
    if (e.key === "Enter")  { e.preventDefault(); commit(); }
    if (e.key === "Escape") { _renameActive = false; li.replaceChild(nameSpan, input); }
  };
  li.replaceChild(input, nameSpan);
  input.focus();
  input.select();
}

// ---------------------------------------------------------------------------
// Create / delete conversation
// ---------------------------------------------------------------------------
async function createConversation(name) {
  const displayName = (name || "").trim() || `Chat ${convOrder.length + 1}`;
  try {
    const conv = await apiPost("/conversations", { display_name: displayName });
    convMap.set(conv.id, conv);
    convOrder.push(conv.id);
    refreshThreadList();
    await loadConversation(conv.id);
  } catch (e) {
    showBanner("Failed to create conversation.", "error");
  }
}

async function deleteConversation(convId) {
  const conv = convMap.get(convId);
  if (!confirm(`Delete "${conv?.display_name || convId}"?`)) return;
  try {
    await apiDelete(`/conversations/${convId}`);
    convMap.delete(convId);
    convOrder = convOrder.filter(id => id !== convId);
    if (currentConvId === convId) { currentConvId = null; messagesDiv.innerHTML = ""; }
    if (convOrder.length === 0) {
      await createConversation("Chat 1");
    } else if (!currentConvId) {
      await loadConversation(convOrder[convOrder.length - 1]);
    }
    refreshThreadList();
  } catch (e) {
    showBanner("Failed to delete conversation.", "error");
  }
}

document.getElementById("newThread").onclick = () => createConversation(null);

// ---------------------------------------------------------------------------
// Initial conversation load
// ---------------------------------------------------------------------------
async function refreshConversations() {
  try {
    const data = await apiGet("/conversations");
    identity   = data.identity || identity;
    convMap.clear();
    convOrder = [];
    (data.conversations || []).forEach(conv => {
      convMap.set(conv.id, conv);
      convOrder.push(conv.id);
    });
    refreshThreadList();
    if (convOrder.length === 0) {
      await createConversation("Chat 1");
    } else if (!currentConvId) {
      await loadConversation(convOrder[convOrder.length - 1]);
    }
  } catch (e) {
    console.error("Failed to load conversations:", e);
    showBanner("Backend unreachable. Is LocalAI running?", "error");
  }
}

// ---------------------------------------------------------------------------
// Theme
// ---------------------------------------------------------------------------
document.getElementById("theme").onclick = () => {
  document.body.classList.toggle("light");
  localStorage.setItem("theme", document.body.classList.contains("light") ? "light" : "dark");
};
if (localStorage.getItem("theme") === "light") document.body.classList.add("light");

// ---------------------------------------------------------------------------
// File upload
// ---------------------------------------------------------------------------
uploadBtn.onclick = () => {
  if (!currentConvId) { showBanner("Select or create a conversation first.", "error"); return; }
  fileInput.click();
};

fileInput.onchange = async (e) => {
  const files = Array.from(e.target.files);
  if (!files.length || !currentConvId) return;
  showBanner(`Uploading ${files.length} file(s)…`, "info");
  const formData = new FormData();
  files.forEach(f => formData.append("files", f));
  try {
    const response = await fetch(
      `http://127.0.0.1:8771/upload?conv_id=${encodeURIComponent(currentConvId)}`,
      { method: "POST", body: formData }
    );
    const result = await response.json();
    if (!response.ok) { showBanner(result.error || "Upload failed", "error"); return; }
    currentAttachments.push(...result.files);
    updateAttachmentDisplay();
    const withOcr = result.files.filter(f => f.ocr_text).length;
    showBanner(
      `Uploaded ${result.files.length} file(s).` +
      (withOcr ? ` OCR extracted text from ${withOcr} file(s).` : ""),
      "success"
    );
  } catch (err) {
    console.error(err);
    showBanner("Upload service unavailable. Is upload_server.py running?", "error");
  }
  fileInput.value = "";
};

function updateAttachmentDisplay() {
  attachedFilesDiv.innerHTML = "";
  currentAttachments.forEach((file, index) => {
    const chip = document.createElement("div");
    chip.className = "file-chip";
    const shortName = file.name.length > 18 ? file.name.substring(0, 16) + "…" : file.name;
    chip.innerHTML = `
      <span class="file-icon">${getFileIcon(file.name)}</span>
      <span title="${escapeHtml(file.name)}">${escapeHtml(shortName)}</span>
      ${file.ocr_text ? '<span class="ocr-badge" title="OCR text extracted">OCR</span>' : ""}
      <button class="file-remove" data-index="${index}">&times;</button>
    `;
    chip.querySelector(".file-remove").onclick = () => {
      currentAttachments.splice(index, 1);
      updateAttachmentDisplay();
    };
    attachedFilesDiv.appendChild(chip);
  });
}

// ---------------------------------------------------------------------------
// Backend status polling
// ---------------------------------------------------------------------------
async function updateStatus() {
  try {
    const data = await apiGet("/status");
    const alive = data.vllm_alive
      ? "✅ ready"
      : (data.running ? "⏳ loading model…" : "❌ stopped");
    statusDiv.textContent =
      `Backend: ${alive} | Model: ${data.current_model_id} | Clients: ${data.clients}`;
    identity = data.identity || identity;
    setControlsEnabled(data.vllm_alive);
  } catch (e) {
    statusDiv.textContent = "Backend: ❌ unreachable";
    setControlsEnabled(false);
  }
}

function setControlsEnabled(enabled) {
  promptBox.disabled = !enabled;
  document.getElementById("send").disabled = !enabled;
  modelSelect.disabled = !enabled;
}

setInterval(updateStatus, 4000);
updateStatus();

// ---------------------------------------------------------------------------
// Backend control buttons
// ---------------------------------------------------------------------------
document.getElementById("restartBackend").onclick = async () => {
  const m = modelsById[modelSelect.value];
  if (!m) return;
  try {
    await apiPost("/restart", { model_id: m.id, model_path: m.path });
    showBanner("Backend restarting…", "info");
  } catch (e) { showBanner("Failed to restart backend.", "error"); }
};

document.getElementById("shutdownBackend").onclick = async () => {
  try {
    await apiPost("/shutdown", {});
    showBanner("Backend shutting down…", "info");
  } catch (e) { showBanner("Failed to shutdown backend.", "error"); }
};

// ---------------------------------------------------------------------------
// Thinking mode toggle — verified against backend
// ---------------------------------------------------------------------------
thinkingToggle.onchange = async () => {
  const desired = thinkingToggle.checked;
  thinkingStatus.textContent = "⏳ checking…";
  thinkingStatus.className   = "thinking-status pending";
  try {
    const data = await apiPost("/thinking", { enabled: desired });
    thinkingToggle.checked = data.thinking_enabled;
    if (!data.vllm_alive) {
      thinkingStatus.textContent = "❌ backend unreachable";
      thinkingStatus.className   = "thinking-status error";
    } else if (data.thinking_enabled) {
      thinkingStatus.textContent = "✅ Thinking ON";
      thinkingStatus.className   = "thinking-status on";
    } else {
      thinkingStatus.textContent = "✅ Thinking OFF";
      thinkingStatus.className   = "thinking-status off";
    }
  } catch (e) {
    thinkingToggle.checked     = !desired;
    thinkingStatus.textContent = "❌ error";
    thinkingStatus.className   = "thinking-status error";
  }
};

// ---------------------------------------------------------------------------
// Models
// ---------------------------------------------------------------------------
async function loadModels() {
  modelSelect.innerHTML = "<option>Loading…</option>";
  try {
    const json = await apiGet("/models");
    models     = json.models || [];
    modelsById = {};
    modelSelect.innerHTML = "";
    models.forEach(m => {
      modelsById[m.id] = m;
      const opt = document.createElement("option");
      opt.value = m.id; opt.textContent = m.name || m.id;
      modelSelect.appendChild(opt);
    });
    if (!models.length) {
      modelSelect.innerHTML = "<option disabled>No models found</option>";
      modelSelect.disabled  = true; return;
    }
    const last = localStorage.getItem("lastModel");
    if (last && modelsById[last]) modelSelect.value = last;
    updateModelInfo();
  } catch (e) {
    console.error("loadModels:", e);
    modelSelect.innerHTML = "<option>Error loading models</option>";
    modelSelect.disabled  = true;
  }
}

function updateModelInfo() {
  const m = modelsById[modelSelect.value];
  if (!m) { modelInfoDiv.innerHTML = ""; return; }
  const sizeGB = m.size_bytes ? (m.size_bytes / 1024 ** 3).toFixed(2) : "N/A";
  const caps   = (m.capabilities || []).join(", ");
  modelInfoDiv.innerHTML = `
    <div class="model-meta">
      <div><strong>Parameters:</strong> ${m.params_b ? m.params_b + "B" : "N/A"}</div>
      <div><strong>Quantization:</strong> ${escapeHtml(m.quantization || "?")}</div>
      <div><strong>Size:</strong> ${sizeGB} GB</div>
      <div><strong>VRAM:</strong> ${m.vram_gb ? m.vram_gb + " GB" : "N/A"}</div>
      <div><strong>GPU util:</strong> ${m.gpu_memory_utilization != null ? (m.gpu_memory_utilization * 100).toFixed(0) + "%" : "N/A"}</div>
      <div><strong>Max context:</strong> ${m.max_model_len || 9216} tokens</div>
      ${m.description ? `<div><strong>Description:</strong> ${escapeHtml(m.description)}</div>` : ""}
      ${caps ? `<div><strong>Capabilities:</strong> ${escapeHtml(caps)}</div>` : ""}
    </div>`;
}

modelSelect.onchange = async () => {
  const m = modelsById[modelSelect.value];
  if (!m) return;
  localStorage.setItem("lastModel", m.id);
  updateModelInfo();
  try {
    await apiPost("/restart", { model_id: m.id, model_path: m.path });
    showBanner(`Switching to model: ${m.name || m.id}`, "info");
  } catch (e) { showBanner("Failed to switch model.", "error"); }
};

loadModels();

// ---------------------------------------------------------------------------
// Send message
// ---------------------------------------------------------------------------
document.getElementById("send").onclick = sendMessage;
promptBox.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) sendMessage();
});

async function sendMessage() {
  const prompt = promptBox.value.trim();
  if (!prompt && currentAttachments.length === 0) return;
  if (!currentConvId) { showBanner("Select or create a conversation first.", "error"); return; }

  const now = new Date().toLocaleString("sv-SE").replace("T", " ");

  const attachmentNote = currentAttachments.length
    ? `\n\n*Attachments: ${currentAttachments.map(f => f.name).join(", ")}*`
    : "";
  renderMessage("user", prompt + attachmentNote, null, null, now);

  const attachmentsSnapshot = [...currentAttachments];
  const selectedTools       = getSelectedTools();

  promptBox.value    = "";
  currentAttachments = [];
  updateAttachmentDisplay();
  promptBox.disabled = true;
  document.getElementById("send").disabled = true;

  startPhaseIndicator();

  try {
    const response = await fetch(
      `http://127.0.0.1:8770/conversations/${currentConvId}/chat`,
      {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({
          message:        prompt,
          attachments:    attachmentsSnapshot,
          selected_tools: selectedTools,
        }),
      }
    );

    stopPhaseIndicator();

    if (!response.ok) {
      const err = await response.json().catch(() => ({ error: response.statusText }));
      renderMessage("assistant", `[Error: ${err.error || response.status}]`, null, null, null);
      return;
    }

    const data = await response.json();
    renderMessage(
      "assistant",
      data.content,
      data.reasoning_content || null,
      data.tool_calls        || null,
      data.ai_timestamp      || null,
    );

  } catch (e) {
    stopPhaseIndicator();
    console.error("sendMessage error:", e);
    renderMessage("assistant", `[Network error: ${e.message}]`, null, null, null);
  } finally {
    promptBox.disabled = false;
    document.getElementById("send").disabled = false;
    promptBox.focus();
  }
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
refreshConversations();
loadTools();
