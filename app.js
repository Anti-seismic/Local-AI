// app.js — LocalAI frontend
// Conversation storage is entirely server-side.
// localStorage is used only for UI preferences (last model, theme).

"use strict";

// ---------------------------------------------------------------------------
// Service address constants
// Mirror of constants.py — update both files when changing ports.
// Future datacenter note: replace these with env-injected config
// (e.g. a /config endpoint served by the launcher) when services
// are split across nodes.
// ---------------------------------------------------------------------------
const PORTS = Object.freeze({
  LAUNCHER:  8770,
  WEBSOCKET: 8765,
  UPLOAD:    8771,
  STATIC:    8080,
  VLLM_API:  8000,
});
const HOST         = "127.0.0.1";
const URL_LAUNCHER = `http://${HOST}:${PORTS.LAUNCHER}`;
const URL_UPLOAD   = `http://${HOST}:${PORTS.UPLOAD}`;
const URL_WS       = `ws://${HOST}:${PORTS.WEBSOCKET}`;

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

// ---------------------------------------------------------------------------
// Model output cleaning — single authoritative pipeline before marked.parse
// All text processing happens here. marked.parse receives clean markdown.
// ---------------------------------------------------------------------------

function cleanModelOutput(text) {
    if (!text) return text;

    // 1. Strip <think>…</think> blocks.
    text = text.replace(/<think>[\s\S]*?<\/think>/gi, "");

    // 2. Normalise line endings.
    text = text.replace(/\r\n/g, "\n").replace(/\r/g, "\n");

    // 3. Strip trailing ` *` used by the model as a list-item terminator.
    text = text.replace(/ \*\s*$/gm, "");

    // 4. Strip trailing ` ####` / ` ###` etc. used as visual separators mid-line.
    text = text.replace(/ #{2,}\s*$/gm, "");

    // 5. Strip trailing ` ---` used as inline section separators.
    //    Standalone `---` on its own line is kept (renders as <hr>).
    //    Capture the preceding non-space char so it is preserved.
    text = text.replace(/(\S) ---\s*$/gm, "$1");

    // 6. Join lines where the NEXT line starts with continuation punctuation
    //    (comma, closing paren, closing bracket, semicolon) — the model
    //    splits sentences mid-clause and puts the punctuation on the next line.
    text = text.replace(/\n([,);][\s]*)/g, " $1");

    // 7. Join orphaned numbered-list labels: lines containing ONLY "N." or "N. "
    //    that were split from their content (e.g. "2.\nGrow:" → "2. Grow:").
    text = text.replace(/^(\d+\.)\s*\n(?![\n])/gm, "$1 ");

    // 8. Join short mid-sentence line fragments onto the next line.
    //    A line is a "fragment" when ALL of these hold:
    //      a) It is short (≤ 55 chars after trim)
    //      b) It does not end with sentence-terminal punctuation (. ? ! :)
    //      c) It does not start with a markdown structural character
    //         (heading #, list - * >, code `, table |, ordered list digit.)
    //      d) The next line is not blank and not a markdown structure line
    //    This catches "Working closely with\nFPC's Project Innovate\nto overcome"
    //    and "Here is the summary for\npages 41–46\nof the document".
    //    We run 3 passes to handle chains of 3+ fragment lines.
    const structStart = /^(\s{0,3})(#{1,6}\s|[-*+>]|\d+\.\s|`{3}|\|)/;
    const terminalEnd = /[.?!:]\s*$/;
    for (let pass = 0; pass < 3; pass++) {
        text = text.replace(/^([^\n]{1,55})\n(?!\n)([^\n]+)/gm, (match, lineA, lineB) => {
            // Don't join if lineA ends with terminal punctuation
            if (terminalEnd.test(lineA)) return match;
            // Don't join if either line starts with markdown structure
            if (structStart.test(lineA) || structStart.test(lineB)) return match;
            // Don't join if lineB starts with continuation punctuation (handled above)
            if (/^[,);]/.test(lineB.trimStart())) return match;
            return lineA + " " + lineB;
        });
    }

    // 9. Join lines where a markdown bold span opens but doesn't close
    //    on the same line (closing ** appears on the next line).
    for (let pass = 0; pass < 2; pass++) {
        text = text.replace(/^((?:[^*\n]|\*(?!\*))*\*\*(?:[^*\n])+)\n(?!\s*\*\*)/gm, "$1 ");
    }

    // 9.5. Promote standalone bold lines to headings.
    //    A line consisting ENTIRELY of "**text**" (nothing else, optionally
    //    a trailing colon) is semantically a section header even though the
    //    model used emphasis syntax instead of '#'. The model is
    //    inconsistent across replies — some use "## Page 41: ..." (real
    //    heading, coloured) and others use "**Page 21 (...):**" on its own
    //    line (plain bold, not coloured) for the exact same role. Promoting
    //    the latter to an h3 makes both render identically.
    //    Bullet labels like "**Title:** description text" are NOT affected —
    //    this only matches when the ENTIRE line is the bold span.
    text = text.replace(/^\*\*([^*\n]+)\*\*(:?)\s*$/gm, "### $1$2");

    // 10. Collapse isolated punctuation-only lines.
    text = text.replace(/\n([,.\[\]]+)\n/g, " $1\n");
    text = text.replace(/^[,.\[\]]+\s*$/gm, "");

    // 11. Collapse 3+ consecutive blank lines to 2.
    text = text.replace(/\n{3,}/g, "\n\n");

    // 12. Strip leading whitespace.
    text = text.replace(/^\s+/, "");

    return text;
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
const protectedDocChk  = document.getElementById("protectedDoc");
const securityBanners  = document.getElementById("securityBanners");

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

// Upload flow state
let _pdfDpi          = parseInt(localStorage.getItem("pdfDpi") || "150");
let _pendingFiles    = [];         // File objects awaiting DPI/password confirmation
let _pendingSavedName = null;      // saved_name from server when password_required

// ---------------------------------------------------------------------------
// WebSocket
// ---------------------------------------------------------------------------
let ws = null;
function connectWS() {
  ws = new WebSocket(URL_WS);
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

// ---------------------------------------------------------------------------
// Banner system — deduplicating, auto-dismissing
// Same message text → increment counter badge instead of stacking.
// Banners live in #securityBanners (above the message feed) so they never
// obscure conversation history.
// ---------------------------------------------------------------------------
const _bannerTimers = new Map();   // message → {el, count, timerId}

function showBanner(message, type = "info") {
  // Deduplicate: if this exact message is already showing, just bump the count.
  if (_bannerTimers.has(message)) {
    const entry = _bannerTimers.get(message);
    entry.count++;
    const badge = entry.el.querySelector(".banner-count");
    if (badge) {
      badge.textContent = `×${entry.count}`;
      badge.style.display = "";
    }
    // Reset the auto-dismiss timer.
    clearTimeout(entry.timerId);
    entry.timerId = setTimeout(() => _dismissBanner(message), 6000);
    return;
  }

  const banner = document.createElement("div");
  banner.className = `banner ${type}-banner`;

  const msgSpan = document.createElement("span");
  msgSpan.className   = "banner-message";
  msgSpan.textContent = message;

  const badge = document.createElement("span");
  badge.className     = "banner-count";
  badge.style.display = "none";   // hidden until count > 1

  const closeBtn = document.createElement("button");
  closeBtn.className   = "banner-close";
  closeBtn.textContent = "×";
  closeBtn.title       = "Dismiss";
  closeBtn.onclick     = () => _dismissBanner(message);

  banner.appendChild(msgSpan);
  banner.appendChild(badge);
  banner.appendChild(closeBtn);
  securityBanners.appendChild(banner);

  const timerId = setTimeout(() => _dismissBanner(message), 6000);
  _bannerTimers.set(message, { el: banner, count: 1, timerId });
}

function _dismissBanner(message) {
  const entry = _bannerTimers.get(message);
  if (!entry) return;
  clearTimeout(entry.timerId);
  entry.el.remove();
  _bannerTimers.delete(message);
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
  label.className = "message-role-label";
  label.textContent = role === "user" ? "You:" : "AI:";
  bubble.appendChild(label);
  const body = document.createElement("div");
  body.className = "message-body";
  body.innerHTML = marked.parse(cleanModelOutput(content) || "");
  bubble.appendChild(body);
  wrap.appendChild(bubble);

  messagesDiv.appendChild(wrap);
  scrollBottom();
}

// ---------------------------------------------------------------------------
// API helpers
// ---------------------------------------------------------------------------
async function apiGet(path) {
  const r = await fetch(`${URL_LAUNCHER}${path}`);
  if (!r.ok) throw new Error(`GET ${path} → ${r.status}`);
  return r.json();
}
async function apiPost(path, body) {
  const r = await fetch(`${URL_LAUNCHER}${path}`, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`POST ${path} → ${r.status}`);
  return r.json();
}
async function apiDelete(path) {
  const r = await fetch(`${URL_LAUNCHER}${path}`, { method: "DELETE" });
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
// File upload — multi-step flow:
//   PDF with "Protected" ticked  →  DPI selector  →  password prompt  →  upload
//   PDF without "Protected"      →  DPI selector  →  upload
//     └─ server returns password_required  →  auto-tick  →  password prompt
//   Non-PDF                      →  upload immediately
// ---------------------------------------------------------------------------

const DPI_OPTIONS = [
  { value: 150,  hint: "text only"       },
  { value: 300,  hint: "text + charts"   },
  { value: 400,  hint: "mixed"           },
  { value: 500,  hint: "mixed"           },
  { value: 600,  hint: "images + text"   },
  { value: 700,  hint: "image-heavy"     },
  { value: 800,  hint: "images only"     },
  { value: 900,  hint: "high detail"     },
  { value: 1000, hint: "photos"          },
  { value: 1100, hint: "photos"          },
  { value: 1200, hint: "photos only"     },
];

uploadBtn.onclick = () => {
  if (!currentConvId) { showBanner("Select or create a conversation first.", "error"); return; }
  fileInput.click();
};

fileInput.onchange = async (e) => {
  const files = Array.from(e.target.files);
  if (!files.length || !currentConvId) return;
  _pendingFiles = files;

  const hasPdf = files.some(f => f.name.toLowerCase().endsWith(".pdf"));
  if (hasPdf) {
    await showDpiSelector(files);   // DPI selector drives the rest
  } else if (protectedDocChk.checked) {
    showPasswordPrompt(files, null);
  } else {
    await uploadFiles(files, null);
  }
  fileInput.value = "";
};

// ── DPI selector ────────────────────────────────────────────────────────────

async function showDpiSelector(files) {
  hideDpiSelector();

  // Pre-flight /pdf_info for recommendation + encryption hint
  let recommendedDpi = _pdfDpi;
  let contentHint    = "";
  let dpiSafeMax     = null;   // null = no cap from server
  const pdfFile = files.find(f => f.name.toLowerCase().endsWith(".pdf"));
  if (pdfFile) {
    try {
      const fd = new FormData();
      fd.append("file", pdfFile);
      const r = await fetch(`${URL_UPLOAD}/pdf_info`, { method: "POST", body: fd });
      if (r.ok) {
        const info = await r.json();
        if (info.recommended_dpi) { recommendedDpi = info.recommended_dpi; }
        if (info.content_hint)    { contentHint = info.content_hint; }
        if (info.dpi_safe_max)    { dpiSafeMax = info.dpi_safe_max; }
        // Server-computed DPI ceiling — honour it in the selector.
        if (info.dpi_safe_max && info.dpi_safe_max < recommendedDpi) {
          recommendedDpi = info.dpi_safe_max;
        }
        if (info.dpi_cap_warning) {
          showBanner(`📐 DPI adjusted: ${info.dpi_cap_warning}`, "info");
        }
        if (info.encrypted) {
          protectedDocChk.checked = true;
          localStorage.setItem("protectedDoc", "true");
          showBanner("🔒 Encrypted PDF detected — please enter the password.", "info");
        }
        if (info.security && info.security.severity !== "CLEAN") {
          showBanner(`⚠️ Pre-flight scan: ${info.security.threats.slice(0,2).join("; ")}`, "error");
        }
      }
    } catch (_) { /* /pdf_info unavailable — proceed with defaults */ }
  }

  const bar = document.getElementById("attachmentBar");
  const div = document.createElement("div");
  div.id        = "dpiSelector";
  div.className = "dpi-selector";

  const label = document.createElement("span");
  label.className   = "dpi-label";
  label.textContent = "📐 Resolution:";

  const select = document.createElement("select");
  select.className = "dpi-select";

  // If the safe max DPI is below the lowest standard option, inject it
  // as the first option so the user can see and select the enforced ceiling.
  const effectiveDpi = recommendedDpi;
  const optionValues = DPI_OPTIONS.map(o => o.value);
  const buildOptions = [...DPI_OPTIONS];
  if (!optionValues.includes(effectiveDpi) && effectiveDpi < optionValues[0]) {
    buildOptions.unshift({ value: effectiveDpi, hint: "max safe (auto-capped)" });
  }

  buildOptions.forEach(opt => {
    const o      = document.createElement("option");
    o.value      = opt.value;
    const isRec  = opt.value === effectiveDpi;
    const isCap  = dpiSafeMax !== null && opt.value > dpiSafeMax;
    o.textContent = `${opt.value} DPI — ${opt.hint}`
      + (isRec  ? " ★ recommended" : "")
      + (isCap  ? " ⚠️ above safe cap" : "");
    if (isRec) o.selected = true;
    select.appendChild(o);
  });

  // Sync _pdfDpi to the recommended/capped value so the upload uses it
  // even if the user doesn't interact with the selector.
  _pdfDpi = effectiveDpi;
  localStorage.setItem("pdfDpi", _pdfDpi);

  if (contentHint) {
    const hint = document.createElement("span");
    hint.className   = "dpi-hint";
    hint.textContent = `Detected: ${contentHint}`;
    div.appendChild(hint);
  }

  const btn = document.createElement("button");
  btn.className   = "dpi-convert-btn";
  btn.textContent = "Convert →";
  btn.onclick = async () => {
    _pdfDpi = parseInt(select.value);
    localStorage.setItem("pdfDpi", _pdfDpi);
    hideDpiSelector();
    if (protectedDocChk.checked) {
      showPasswordPrompt(files, null);
    } else {
      await uploadFiles(files, null);
    }
  };

  div.appendChild(label);
  div.appendChild(select);
  div.appendChild(btn);
  bar.insertBefore(div, document.getElementById("attachedFiles"));
}

function hideDpiSelector() {
  const el = document.getElementById("dpiSelector");
  if (el) el.remove();
}

// ── Password prompt ──────────────────────────────────────────────────────────

function showPasswordPrompt(files, savedName) {
  hidePasswordPrompt();
  _pendingFiles    = files;
  _pendingSavedName = savedName;

  const div = document.createElement("div");
  div.id        = "passwordPrompt";
  div.className = "password-prompt";

  const lbl = document.createElement("span");
  lbl.textContent = "🔒 Document password:";

  const input = document.createElement("input");
  input.type        = "password";
  input.className   = "doc-password-input";
  input.placeholder = "Enter password…";
  input.setAttribute("autocomplete", "new-password");

  const submitBtn = document.createElement("button");
  submitBtn.textContent = "Unlock";
  submitBtn.className   = "pwd-action-btn";

  const cancelBtn = document.createElement("button");
  cancelBtn.textContent = "Cancel";
  cancelBtn.className   = "pwd-cancel-btn";

  const errSpan = document.createElement("span");
  errSpan.className = "pwd-error";
  errSpan.id        = "pwdError";

  const doSubmit = async () => {
    const pwd = input.value;   // opaque — never logged, never stored
    if (savedName) {
      await unlockFile(savedName, pwd);
    } else {
      await uploadFiles(files, pwd);
    }
  };

  submitBtn.onclick     = doSubmit;
  cancelBtn.onclick     = () => {
    hidePasswordPrompt();
    protectedDocChk.checked = false;
    localStorage.setItem("protectedDoc", "false");
  };
  input.onkeydown = (ev) => {
    if (ev.key === "Enter")  { ev.preventDefault(); doSubmit(); }
    if (ev.key === "Escape") cancelBtn.click();
  };

  div.append(lbl, input, submitBtn, cancelBtn, errSpan);

  const chat = document.getElementById("chat");
  chat.insertBefore(div, securityBanners);
  input.focus();
}

function hidePasswordPrompt() {
  const el = document.getElementById("passwordPrompt");
  if (el) el.remove();
}

function _setPasswordError(msg) {
  const el = document.getElementById("pwdError");
  if (el) el.textContent = msg;
}

// ── Core upload ──────────────────────────────────────────────────────────────

async function uploadFiles(files, password) {
  if (!files || !files.length || !currentConvId) return;
  showBanner(`Processing ${files.length} file(s)…`, "info");

  const fd = new FormData();
  files.forEach(f => fd.append("files", f));
  if (password !== null && password !== undefined) {
    fd.append("password", password);   // opaque — server never logs it
  }

  const url = `${URL_UPLOAD}/upload` +
              `?conv_id=${encodeURIComponent(currentConvId)}&pdf_dpi=${_pdfDpi}`;
  try {
    const resp   = await fetch(url, { method: "POST", body: fd });
    const result = await resp.json();

    if (!resp.ok) {
      if (result.password_invalid) { _setPasswordError("❌ Incorrect password — try again."); return; }
      showBanner(result.error || "Upload failed.", "error");
      return;
    }

    if (result.password_required) {
      // Server detected encryption without a password being supplied
      handlePasswordRequired(result, files);
      return;
    }

    hidePasswordPrompt();
    const newFiles = result.files || [];
    currentAttachments.push(...newFiles);
    _renderSecurityBanners(newFiles);
    updateAttachmentDisplay();

    const pages   = newFiles.filter(f => f.pdf_source).length;
    const blocked = newFiles.filter(f => f.type === "security_blocked").length;
    let msg = `Uploaded ${files.length} file(s).`;
    if (pages)   msg += ` ${pages} PDF page(s) ready.`;
    if (blocked) msg += ` ⚠️ ${blocked} file(s) blocked (security).`;
    showBanner(msg, blocked ? "error" : "success");

  } catch (err) {
    console.error(err);
    showBanner("Upload service unavailable. Is upload_server.py running?", "error");
  }
}

// ── Unlock (password-protected, already saved on server) ────────────────────

async function unlockFile(savedName, password) {
  const fd = new FormData();
  fd.append("conv_id",    currentConvId);
  fd.append("saved_name", savedName);
  fd.append("password",   password);   // opaque — server never logs it

  const url = `${URL_UPLOAD}/unlock?pdf_dpi=${_pdfDpi}`;
  try {
    const resp   = await fetch(url, { method: "POST", body: fd });
    const result = await resp.json();

    if (!resp.ok || result.password_invalid) {
      _setPasswordError("❌ Incorrect password — try again.");
      return;
    }

    hidePasswordPrompt();
    const newFiles = result.files || [];
    currentAttachments.push(...newFiles);
    _renderSecurityBanners(newFiles);
    updateAttachmentDisplay();
    showBanner(`PDF unlocked: ${newFiles.length} page(s) converted.`, "success");
  } catch (err) {
    _setPasswordError("❌ Unlock service unavailable.");
  }
}

function handlePasswordRequired(result, files) {
  _pendingSavedName = result.saved_name;
  protectedDocChk.checked = true;
  localStorage.setItem("protectedDoc", "true");
  // Primary: re-submit File object from browser memory with password
  // Fallback: server will use the already-saved file via /unlock
  showPasswordPrompt(files, result.saved_name);
}

// ── Attachment display ────────────────────────────────────────────────────────

function updateAttachmentDisplay() {
  attachedFilesDiv.innerHTML = "";

  // Group PDF pages by their source PDF
  const pdfGroups = new Map();
  const singles   = [];

  currentAttachments.forEach((file, index) => {
    if (file.pdf_source) {
      if (!pdfGroups.has(file.pdf_source)) pdfGroups.set(file.pdf_source, []);
      pdfGroups.get(file.pdf_source).push({ file, index });
    } else {
      singles.push({ file });
    }
  });

  // PDF groups → one chip with highlighted page-count badge
  pdfGroups.forEach((items, pdfName) => {
    const sanitized = items.some(i => i.file.sanitized);
    const chip      = document.createElement("div");
    chip.className  = "file-chip" + (sanitized ? " file-chip-sanitized" : "");
    const short     = pdfName.length > 22 ? pdfName.substring(0, 20) + "…" : pdfName;
    chip.innerHTML  = `
      <span class="file-icon">📄</span>
      <span title="${escapeHtml(pdfName)}">${escapeHtml(short)}</span>
      <span class="pdf-page-badge">${items.length}&nbsp;pg</span>
      ${sanitized ? '<span class="sanitized-badge" title="Security threats removed before analysis">🛡️</span>' : ""}
      <button class="file-remove">&times;</button>
    `;
    chip.querySelector(".file-remove").onclick = () => {
      currentAttachments = currentAttachments.filter(f => f.pdf_source !== pdfName);
      updateAttachmentDisplay();
    };
    attachedFilesDiv.appendChild(chip);
  });

  // Singles (images, documents, blocked files)
  // Remove by object identity — index would be stale if a PDF group was removed first.
  singles.forEach(({ file }) => {
    const chip = document.createElement("div");
    if (file.type === "security_blocked") {
      chip.className = "file-chip file-chip-blocked";
      const short = file.name.length > 18 ? file.name.substring(0, 16) + "…" : file.name;
      chip.innerHTML = `
        <span class="file-icon">🚫</span>
        <span title="${escapeHtml(file.name + ": " + (file.block_reason || "blocked"))}">${escapeHtml(short)}</span>
        <button class="file-remove">&times;</button>
      `;
    } else {
      chip.className = "file-chip" + (file.sanitized ? " file-chip-sanitized" : "");
      const short = file.name.length > 18 ? file.name.substring(0, 16) + "…" : file.name;
      chip.innerHTML = `
        <span class="file-icon">${getFileIcon(file.name)}</span>
        <span title="${escapeHtml(file.name)}">${escapeHtml(short)}</span>
        ${file.sanitized ? '<span class="sanitized-badge" title="Security threats removed">🛡️</span>' : ""}
        <button class="file-remove">&times;</button>
      `;
    }
    chip.querySelector(".file-remove").onclick = () => {
      currentAttachments = currentAttachments.filter(a => a !== file);
      updateAttachmentDisplay();
    };
    attachedFilesDiv.appendChild(chip);
  });
}

// ── Security banners ──────────────────────────────────────────────────────────
// Routes through showBanner() so deduplication, auto-dismiss and close button
// are inherited automatically.

function _renderSecurityBanners(files) {
  files.forEach(f => {
    if (f.type === "security_blocked") {
      showBanner(
        `🚫 ${f.name} — blocked: ${f.block_reason || "security threat"}`,
        "error"
      );
    } else if (f.security_warning) {
      showBanner(
        `🛡️ ${f.name}: ${f.security_warning} Threats removed before analysis.`,
        "info"
      );
    }
  });
}

function _clearSecurityBanners() {
  // Dismiss all banners — both regular and security warnings.
  // Called on send so the screen clears before the response arrives.
  [..._bannerTimers.keys()].forEach(_dismissBanner);
}

// ---------------------------------------------------------------------------
// Backend status polling
// ---------------------------------------------------------------------------
async function updateStatus() {
  try {
    const data = await apiGet("/status");
    const ready   = data.system_ready;
    const alive   = data.vllm_alive;
    const running = data.running;

    let statusText;
    if (ready) {
      statusText = "✅ ready";
    } else if (running || alive) {
      statusText = "⏳ loading model…";
    } else {
      statusText = "❌ stopped";
    }

    statusDiv.textContent =
      `Backend: ${statusText} | Model: ${data.current_model_id} | Clients: ${data.clients}`;
    identity = data.identity || identity;

    // The send button and prompt are greyed based on system_ready — the same
    // flag that gates /chat server-side.  This means the UI and the server
    // semaphore are always in sync: both open at exactly the same moment.
    const wasEnabled = !promptBox.disabled;
    setControlsEnabled(ready);

    // If the engine just went from ready → not-ready AND the process is not
    // running (i.e. it crashed rather than intentionally loading), show an
    // actionable banner so the user knows to restart rather than wondering why
    // the controls are greyed.
    if (wasEnabled && !ready && !running && !alive) {
      showBanner(
        "⚠️ AI engine stopped unexpectedly (likely OOM). " +
        "Press 🔄 Restart Backend to recover.",
        "error"
      );
    }
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
  const sizeGB    = m.size_bytes ? (m.size_bytes / 1024 ** 3).toFixed(2) : "N/A";
  const caps      = (m.capabilities || []).join(", ");
  // na() renders a visually distinct placeholder for admin-fillable fields
  // that are genuinely absent — makes missing data obvious rather than silent.
  const na = (label) => `<span class="meta-na" title="${label} not set in model JSON">N/A</span>`;
  modelInfoDiv.innerHTML = `
    <div class="model-meta">
      <div><strong>Parameters:</strong> ${m.params_b != null ? m.params_b + "B" : na("params_b")}</div>
      <div><strong>Quantization:</strong> ${escapeHtml(m.quantization || "?")}</div>
      <div><strong>Size:</strong> ${sizeGB} GB</div>
      <div><strong>VRAM (weights):</strong> ${m.model_weights_vram_gb != null ? m.model_weights_vram_gb + " GB" : na("model_weights_vram_gb")}</div>
      <div><strong>GPU util:</strong> ${m.gpu_memory_utilization != null ? (m.gpu_memory_utilization * 100).toFixed(0) + "%" : na("gpu_memory_utilization")}</div>
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

  // Client-side injection pattern check — supplementary UX only.
  // The real gate is server-side (input_guard.py). This gives immediate
  // visual feedback without a round-trip. User can still proceed if they
  // believe the pattern match is a false positive; the server will block it
  // definitively if it truly matches.
  const _INJ_PATTERNS = [
    /ignore\s+(all\s+)?(previous|above|prior)\s+(instructions?|context|prompt)/i,
    /disregard\s+(the\s+)?(above|previous|following|all)/i,
    /forget\s+(all\s+)?(previous\s+)?(instructions?|context|rules?)/i,
    /override\s+(your\s+)?(instructions?|rules?|system\s*prompt)/i,
    /<\|system\|>|<\|assistant\|>|<\|im_start\|>|\[INST\]|<<SYS>>/i,
    /\bDAN\b|\bdeveloper\s+mode\b|\bjailbreak\b|\buncensored\s+mode\b/i,
    /(print|reveal|show)\s+(your\s+)?(system\s+prompt|instructions)/i,
  ];
  if (_INJ_PATTERNS.some(p => p.test(prompt))) {
    showBanner(
      "⚠️ Possible prompt injection pattern detected. " +
      "The server may block this message.",
      "error"
    );
    // Do not return — let server make the final call.
  }

  const now = new Date().toLocaleString("sv-SE").replace("T", " ");

  const attachmentNote = currentAttachments.length
    ? `\n\n*Attachments: ${currentAttachments.map(f => f.name).join(", ")}*`
    : "";
  renderMessage("user", prompt + attachmentNote, null, null, now);

  // Strip security-blocked attachments — they must never reach the model
  const attachmentsSnapshot = [...currentAttachments].filter(
    a => a.type !== "security_blocked"
  );
  const selectedTools = getSelectedTools();

  promptBox.value    = "";
  currentAttachments = [];
  updateAttachmentDisplay();
  _clearSecurityBanners();
  hideDpiSelector();
  hidePasswordPrompt();
  promptBox.disabled = true;
  document.getElementById("send").disabled = true;

  startPhaseIndicator();

  try {
    const response = await fetch(
      `${URL_LAUNCHER}/conversations/${currentConvId}/chat`,
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
      if (err.blocked) {
        // Prompt injection blocked server-side
        showBanner(err.error || "Message blocked by server.", "error");
      } else if (response.status === 503 && err.system_ready === false) {
        // Server semaphore closed — system not ready yet
        showBanner(err.error || "System not ready. Please wait for ✅ ready.", "error");
        // Re-render the user message so it isn't lost — user can resend
      } else {
        renderMessage("assistant", `[Error: ${err.error || response.status}]`, null, null, null);
      }
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

    // ── Multi-batch continuation ─────────────────────────────────────────
    // If the server signals more batches (next_batch_index != null), fire
    // them one by one. Each renders as a separate assistant bubble labelled
    // with its page range, so the user sees progressive output rather than
    // waiting for all batches to complete before seeing anything.
    if (data.next_batch_index != null) {
      let nextBatch      = data.next_batch_index;
      const totalBatches = data.total_batches;
      const pdfName      = data.pdf_name;

      while (nextBatch != null) {
        const batchAtts = attachmentsSnapshot.filter(
          a => (a.batch_index ?? 0) === nextBatch
        );
        if (!batchAtts.length) break;

        const fp = batchAtts[0].pdf_page;
        const lp = batchAtts[batchAtts.length - 1].pdf_page;
        const tp = batchAtts[0].pdf_total_pages;

        showBanner(
          `📄 Analysing pages ${fp}–${lp} of ${tp} (batch ${nextBatch + 1}/${totalBatches})…`,
          "info"
        );

        let contResp;
        try {
          contResp = await fetch(
            `${URL_LAUNCHER}/conversations/${currentConvId}/chat_continue`,
            {
              method:  "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                batch_index:    nextBatch,
                total_batches:  totalBatches,
                attachments:    attachmentsSnapshot,
                pdf_name:       pdfName,
                selected_tools: selectedTools,
                summary:        false,
              }),
            }
          );
        } catch (e) {
          renderMessage("assistant",
            `[Network error on batch ${nextBatch + 1}: ${e.message}]`,
            null, null, null);
          break;
        }

        if (!contResp.ok) {
          const err = await contResp.json().catch(() => ({}));
          renderMessage("assistant",
            `[Error on pages ${fp}–${lp}: ${err.error || contResp.status}]`,
            null, null, null);
          break;
        }

        const contData = await contResp.json();
        renderMessage(
          "assistant",
          `**Pages ${contData.first_page}–${contData.last_page} of ${contData.total_pages}:**\n\n${contData.content}`,
          contData.reasoning_content || null,
          contData.tool_calls        || null,
          contData.ai_timestamp      || null,
        );

        nextBatch = contData.next_batch_index ?? null;

        // All image batches done — fire the executive summary as a dedicated call
        if (nextBatch === null && contData.summary_pending) {
          showBanner("📋 Writing executive summary…", "info");
          try {
            const sumResp = await fetch(
              `${URL_LAUNCHER}/conversations/${currentConvId}/chat_continue`,
              {
                method:  "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                  batch_index:    contData.batch_index,
                  total_batches:  totalBatches,
                  attachments:    attachmentsSnapshot,
                  pdf_name:       pdfName,
                  selected_tools: selectedTools,
                  summary:        true,
                }),
              }
            );
            if (sumResp.ok) {
              const sumData = await sumResp.json();
              renderMessage(
                "assistant",
                `**Executive Summary:**\n\n${sumData.content}`,
                sumData.reasoning_content || null,
                sumData.tool_calls        || null,
                sumData.ai_timestamp      || null,
              );
            } else {
              const err = await sumResp.json().catch(() => ({}));
              renderMessage("assistant",
                `[Executive summary error: ${err.error || sumResp.status}]`,
                null, null, null);
            }
          } catch (e) {
            renderMessage("assistant",
              `[Network error on executive summary: ${e.message}]`,
              null, null, null);
          }
        }
      }
    }

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

// Restore persisted UI preferences
if (localStorage.getItem("protectedDoc") === "true") protectedDocChk.checked = true;
protectedDocChk.onchange = () =>
  localStorage.setItem("protectedDoc", protectedDocChk.checked ? "true" : "false");

refreshConversations();
loadTools();
