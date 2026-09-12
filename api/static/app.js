"use strict";

const NOTICE_TEXT =
  "KrishiGPT-nano is an experimental 1.86M-parameter agriculture language " +
  "model built from scratch. It is a next-token predictor, not a " +
  "question-answering system: answers can be fluent and wrong, may contain " +
  "fabricated words, and must NOT be treated as professional agricultural " +
  "advice. 128-token context: long histories are truncated; the model keeps " +
  "no memory between sessions.";

const chatEl = document.getElementById("chat");
const inputEl = document.getElementById("input");
const sendBtn = document.getElementById("send-btn");
const clearBtn = document.getElementById("clear-btn");
const statusEl = document.getElementById("model-status");
const groundedToggle = document.getElementById("grounded-toggle");

let messages = [];   // {role, content} history sent to /chat each turn
let busy = false;

function el(tag, cls) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  return node;
}

function addNotice() {
  const notice = el("div", "notice");
  notice.textContent = NOTICE_TEXT;
  chatEl.appendChild(notice);
}

function addBubble(role, text, sources) {
  const wrap = el("div", "message " + role);
  const who = el("span", "who");
  who.textContent =
    role === "user" ? "You" : role === "assistant" ? "KrishiGPT-nano" : "Error";
  const body = el("span", "body");
  body.textContent = text;
  wrap.appendChild(who);
  wrap.appendChild(body);
  if (sources && sources.length) {
    const src = el("div", "sources");
    src.textContent = "source: " + sources.join(", ");
    wrap.appendChild(src);
  }
  if (role === "assistant" && groundedToggle.checked) {
    const tag = el("div", "mode-tag");
    tag.textContent = "grounded quote (from training corpus, not generated)";
    wrap.appendChild(tag);
  }
  chatEl.appendChild(wrap);
  chatEl.scrollTop = chatEl.scrollHeight;
  return wrap;
}

let typingEl = null;

function setTyping(on) {
  if (on && !typingEl) {
    typingEl = el("div", "message assistant");
    const dots = el("span", "typing");
    for (let i = 0; i < 3; i++) dots.appendChild(el("span"));
    typingEl.appendChild(dots);
    chatEl.appendChild(typingEl);
    chatEl.scrollTop = chatEl.scrollHeight;
  } else if (!on && typingEl) {
    typingEl.remove();
    typingEl = null;
  }
}

function detailToString(detail) {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail.map((d) => (d && d.msg) || JSON.stringify(d)).join("; ");
  }
  return JSON.stringify(detail);
}

async function send() {
  const text = inputEl.value.trim();
  if (!text || busy) return;

  busy = true;
  sendBtn.disabled = true;
  addBubble("user", text);
  messages.push({ role: "user", content: text });
  inputEl.value = "";
  autoresize();
  setTyping(true);

  try {
    const res = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        messages: messages,
        seed: 0,
        grounded: groundedToggle.checked,
      }),
    });
    let data = null;
    try { data = await res.json(); } catch (_) { /* non-JSON error body */ }
    if (!res.ok) {
      const msg = data ? detailToString(data.detail) : "HTTP " + res.status;
      throw new Error(msg);
    }
    setTyping(false);
    addBubble("assistant", data.response, data.sources);
    messages.push({ role: "assistant", content: data.response });
  } catch (err) {
    setTyping(false);
    messages.pop();                       // drop the failed user turn
    addBubble("error", "Request failed: " + err.message);
  } finally {
    busy = false;
    sendBtn.disabled = false;
    inputEl.focus();
  }
}

function clearChat() {
  messages = [];
  chatEl.innerHTML = "";
  addNotice();
  inputEl.focus();
}

function autoresize() {
  inputEl.style.height = "auto";
  inputEl.style.height = Math.min(inputEl.scrollHeight, 140) + "px";
}

async function checkHealth() {
  try {
    const res = await fetch("/health");
    const data = await res.json();
    if (data.loaded) {
      statusEl.textContent = "model loaded (" + data.params.toLocaleString() + " params)";
      statusEl.className = "status ok";
    } else {
      statusEl.textContent = "model NOT loaded";
      statusEl.className = "status bad";
    }
  } catch (_) {
    statusEl.textContent = "server unreachable";
    statusEl.className = "status bad";
  }
}

sendBtn.addEventListener("click", send);
clearBtn.addEventListener("click", clearChat);
inputEl.addEventListener("input", autoresize);
inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    send();
  }
});

addNotice();
checkHealth();
inputEl.focus();
