#!/usr/bin/env python3
"""
Embedding Quiz — CAPTCHA-style review of card parser embeddings.

Fetches random cards from Scryfall, parses them through the card parser,
and presents a 3x3 grid for human review. Vote thumbs-up/down on whether
the detected mechanics capture the card accurately.

Usage:
    python3 scripts/embedding_quiz.py
    python3 scripts/embedding_quiz.py --port 8787
    python3 scripts/embedding_quiz.py --format modern
    python3 scripts/embedding_quiz.py --set FDN

Then open http://localhost:8787 in your browser.
"""

import argparse
import html
import json
import os
import sys
import time
import urllib.request
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mechanics.card_parser import parse_oracle_text
from src.mechanics.vocabulary import Mechanic

# ---------------------------------------------------------------------------
# Scryfall helpers
# ---------------------------------------------------------------------------

SCRYFALL_RANDOM = "https://api.scryfall.com/cards/random"
SCRYFALL_SEARCH = "https://api.scryfall.com/cards/search"
REQUEST_DELAY = 0.105


def fetch_random_cards(n: int = 9, fmt: str = "standard",
                       set_code: str | None = None) -> list[dict]:
    """Fetch n random cards from Scryfall with oracle text."""
    cards = []
    seen = set()
    attempts = 0
    max_attempts = n * 4

    while len(cards) < n and attempts < max_attempts:
        attempts += 1
        parts = ["has:oracle_text", "-is:digital", "-is:funny", "-is:token"]
        if set_code:
            parts.append(f"set:{set_code}")
        else:
            parts.append(f"f:{fmt}")
        q = " ".join(parts)
        url = f"{SCRYFALL_RANDOM}?q={urllib.parse.quote(q)}"

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "MTG-EmbeddingQuiz/1.0"})
            with urllib.request.urlopen(req) as resp:
                card = json.loads(resp.read().decode())
        except Exception:
            time.sleep(REQUEST_DELAY)
            continue

        name = card.get("name", "")
        if name in seen:
            time.sleep(REQUEST_DELAY)
            continue
        seen.add(name)

        oracle = card.get("oracle_text", "")
        type_line = card.get("type_line", "")
        image_uri = ""

        # Handle DFCs
        if "card_faces" in card and card["card_faces"]:
            front = card["card_faces"][0]
            oracle = front.get("oracle_text", oracle)
            type_line = front.get("type_line", type_line)
            image_uri = front.get("image_uris", {}).get("normal", "")

        if not image_uri:
            image_uri = card.get("image_uris", {}).get("normal", "")

        if not oracle.strip():
            time.sleep(REQUEST_DELAY)
            continue

        # Parse through our parser
        try:
            result = parse_oracle_text(oracle, type_line)
            mechanics = [m.name for m in result.mechanics]
            params = result.parameters
            confidence = result.confidence
            unparsed = result.unparsed_text
        except Exception as e:
            mechanics = []
            params = {}
            confidence = 0.0
            unparsed = str(e)

        cards.append({
            "name": name,
            "set": card.get("set", "???").upper(),
            "type_line": type_line,
            "oracle_text": oracle,
            "image_uri": image_uri,
            "scryfall_uri": card.get("scryfall_uri", ""),
            "mechanics": mechanics,
            "parameters": params,
            "confidence": round(confidence, 3),
            "unparsed_text": unparsed,
        })
        time.sleep(REQUEST_DELAY)

    return cards


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Embedding Quiz</title>
<style>
  :root {
    --bg: #1a1a2e;
    --card-bg: #16213e;
    --card-border: #0f3460;
    --accent: #e94560;
    --good: #00b894;
    --bad: #e17055;
    --text: #eee;
    --text-muted: #999;
    --tag-bg: #0f3460;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
  }
  header {
    text-align: center;
    padding: 24px 16px 12px;
  }
  header h1 { font-size: 1.6rem; font-weight: 700; }
  header p { color: var(--text-muted); font-size: 0.85rem; margin-top: 4px; }
  .stats-bar {
    display: flex;
    justify-content: center;
    gap: 24px;
    padding: 8px 16px 16px;
    font-size: 0.85rem;
    color: var(--text-muted);
  }
  .stats-bar span { font-weight: 600; color: var(--text); }

  .grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
    max-width: 1400px;
    margin: 0 auto;
    padding: 0 16px 24px;
  }
  @media (max-width: 1100px) { .grid { grid-template-columns: repeat(2, 1fr); } }
  @media (max-width: 700px) { .grid { grid-template-columns: 1fr; } }

  .card {
    background: var(--card-bg);
    border: 2px solid var(--card-border);
    border-radius: 12px;
    overflow: hidden;
    transition: border-color 0.2s, transform 0.15s;
    display: flex;
    flex-direction: column;
  }
  .card:hover { transform: translateY(-2px); }
  .card.voted-good { border-color: var(--good); }
  .card.voted-bad { border-color: var(--bad); }

  .card-top {
    display: flex;
    gap: 12px;
    padding: 12px;
    align-items: flex-start;
  }
  .card-img {
    width: 130px;
    min-width: 130px;
    border-radius: 8px;
    cursor: pointer;
    transition: transform 0.2s;
  }
  .card-img:hover { transform: scale(1.05); }
  .card-info { flex: 1; min-width: 0; }
  .card-name {
    font-weight: 700;
    font-size: 0.95rem;
    margin-bottom: 2px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .card-name a { color: var(--text); text-decoration: none; }
  .card-name a:hover { text-decoration: underline; }
  .card-type { font-size: 0.75rem; color: var(--text-muted); margin-bottom: 6px; }
  .card-oracle {
    font-size: 0.75rem;
    color: #ccc;
    line-height: 1.4;
    max-height: 80px;
    overflow-y: auto;
    white-space: pre-wrap;
    padding-right: 4px;
  }

  .card-bottom {
    padding: 0 12px 12px;
    flex: 1;
    display: flex;
    flex-direction: column;
  }
  .conf-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 8px;
  }
  .conf-bar-track {
    flex: 1;
    height: 6px;
    background: #2d3a5a;
    border-radius: 3px;
    overflow: hidden;
  }
  .conf-bar-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.3s;
  }
  .conf-label { font-size: 0.75rem; font-weight: 600; min-width: 36px; text-align: right; }

  .mechanics-list {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    margin-bottom: 8px;
    max-height: 72px;
    overflow-y: auto;
  }
  .mech-tag {
    display: inline-block;
    background: var(--tag-bg);
    color: #8ecae6;
    font-size: 0.65rem;
    font-weight: 600;
    padding: 2px 6px;
    border-radius: 4px;
    white-space: nowrap;
  }
  .param-tag {
    background: #2d1b4e;
    color: #c4a7e7;
  }

  .unparsed {
    font-size: 0.65rem;
    color: var(--bad);
    font-style: italic;
    margin-bottom: 8px;
    max-height: 36px;
    overflow: hidden;
  }

  .vote-row {
    display: flex;
    gap: 8px;
    margin-top: auto;
  }
  .vote-btn {
    flex: 1;
    padding: 8px;
    border: 2px solid transparent;
    border-radius: 8px;
    font-size: 0.85rem;
    font-weight: 700;
    cursor: pointer;
    transition: all 0.15s;
    background: #2d3a5a;
    color: var(--text);
  }
  .vote-btn:hover { transform: scale(1.03); }
  .btn-good { border-color: var(--good); }
  .btn-good:hover, .btn-good.active { background: var(--good); color: #fff; }
  .btn-bad { border-color: var(--bad); }
  .btn-bad:hover, .btn-bad.active { background: var(--bad); color: #fff; }

  .note-input {
    width: 100%;
    margin-top: 6px;
    padding: 6px 8px;
    border-radius: 6px;
    border: 1px solid #2d3a5a;
    background: #0d1b2a;
    color: var(--text);
    font-size: 0.75rem;
    display: none;
  }
  .note-input.visible { display: block; }
  .note-input::placeholder { color: #555; }

  .actions {
    display: flex;
    justify-content: center;
    gap: 12px;
    padding: 16px;
  }
  .action-btn {
    padding: 10px 28px;
    border-radius: 8px;
    border: none;
    font-size: 0.95rem;
    font-weight: 700;
    cursor: pointer;
    transition: all 0.15s;
  }
  .action-btn:hover { transform: scale(1.04); }
  .btn-next { background: var(--accent); color: #fff; }
  .btn-report { background: #0f3460; color: var(--text); }

  /* Report overlay */
  .overlay {
    display: none;
    position: fixed;
    inset: 0;
    background: rgba(0,0,0,0.85);
    z-index: 100;
    justify-content: center;
    align-items: flex-start;
    padding: 40px 16px;
    overflow-y: auto;
  }
  .overlay.visible { display: flex; }
  .report-box {
    background: var(--card-bg);
    border: 2px solid var(--card-border);
    border-radius: 16px;
    padding: 32px;
    max-width: 700px;
    width: 100%;
  }
  .report-box h2 { margin-bottom: 16px; }
  .report-box pre {
    background: #0d1b2a;
    padding: 16px;
    border-radius: 8px;
    font-size: 0.8rem;
    overflow-x: auto;
    white-space: pre-wrap;
    line-height: 1.5;
  }
  .report-box button {
    margin-top: 16px;
    padding: 8px 24px;
    border-radius: 8px;
    border: none;
    background: var(--accent);
    color: #fff;
    font-weight: 700;
    cursor: pointer;
  }
  .report-box .btn-row { display: flex; gap: 10px; }

  .loading {
    text-align: center;
    padding: 80px 16px;
    font-size: 1.1rem;
    color: var(--text-muted);
  }
  .spinner {
    display: inline-block;
    width: 28px; height: 28px;
    border: 3px solid #2d3a5a;
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    margin-bottom: 12px;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>

<header>
  <h1>Embedding Quiz</h1>
  <p>Do the detected mechanics capture this card? Vote on each card below.</p>
</header>

<div class="stats-bar" id="stats-bar">
  <div>Reviewed: <span id="stat-total">0</span></div>
  <div>Approved: <span id="stat-good" style="color:var(--good)">0</span></div>
  <div>Rejected: <span id="stat-bad" style="color:var(--bad)">0</span></div>
  <div>Approval rate: <span id="stat-rate">—</span></div>
</div>

<div id="content">
  <div class="loading">
    <div class="spinner"></div>
    <div>Fetching cards from Scryfall...</div>
  </div>
</div>

<div class="actions" id="actions" style="display:none">
  <button class="action-btn btn-next" onclick="loadBatch()">Next 9 Cards</button>
  <button class="action-btn btn-report" onclick="showReport()">View Report</button>
</div>

<div class="overlay" id="overlay">
  <div class="report-box">
    <h2>Review Report</h2>
    <pre id="report-text"></pre>
    <div class="btn-row">
      <button onclick="closeReport()">Close</button>
      <button onclick="downloadReport()" style="background:#0f3460">Download JSON</button>
    </div>
  </div>
</div>

<script>
const votes = [];       // {name, set, vote, confidence, mechanics, note, oracle_text}
let currentCards = [];
let batchNum = 0;

function confColor(c) {
  if (c >= 0.8) return 'var(--good)';
  if (c >= 0.5) return '#fdcb6e';
  return 'var(--bad)';
}

function renderGrid(cards) {
  currentCards = cards;
  const el = document.getElementById('content');
  if (!cards.length) {
    el.innerHTML = '<div class="loading">No cards returned. Try again.</div>';
    return;
  }
  let html = '<div class="grid">';
  cards.forEach((c, i) => {
    const confPct = Math.round(c.confidence * 100);
    const mechTags = c.mechanics.map(m =>
      `<span class="mech-tag">${esc(m)}</span>`
    ).join('');
    const paramTags = Object.entries(c.parameters || {}).map(([k,v]) =>
      `<span class="mech-tag param-tag">${esc(k)}=${esc(String(v))}</span>`
    ).join('');
    const unparsed = c.unparsed_text
      ? `<div class="unparsed">Unparsed: ${esc(c.unparsed_text.substring(0, 120))}</div>`
      : '';

    html += `
    <div class="card" id="card-${i}">
      <div class="card-top">
        <img class="card-img" src="${esc(c.image_uri)}" alt="${esc(c.name)}"
             onclick="window.open('${esc(c.scryfall_uri)}','_blank')"
             loading="lazy">
        <div class="card-info">
          <div class="card-name" title="${esc(c.name)}">
            <a href="${esc(c.scryfall_uri)}" target="_blank">${esc(c.name)}</a>
            <span style="font-weight:400;font-size:0.7rem;color:var(--text-muted)"> ${esc(c.set)}</span>
          </div>
          <div class="card-type">${esc(c.type_line)}</div>
          <div class="card-oracle">${esc(c.oracle_text)}</div>
        </div>
      </div>
      <div class="card-bottom">
        <div class="conf-row">
          <div class="conf-bar-track">
            <div class="conf-bar-fill" style="width:${confPct}%;background:${confColor(c.confidence)}"></div>
          </div>
          <div class="conf-label" style="color:${confColor(c.confidence)}">${confPct}%</div>
        </div>
        <div class="mechanics-list">${mechTags}${paramTags}</div>
        ${unparsed}
        <div class="vote-row">
          <button class="vote-btn btn-good" onclick="vote(${i},'good')">&#x1F44D; Good</button>
          <button class="vote-btn btn-bad" onclick="vote(${i},'bad')">&#x1F44E; Bad</button>
        </div>
        <input class="note-input" id="note-${i}" placeholder="What's wrong? (optional)"
               onchange="updateNote(${i})">
      </div>
    </div>`;
  });
  html += '</div>';
  el.innerHTML = html;
  document.getElementById('actions').style.display = 'flex';
}

function esc(s) {
  const d = document.createElement('div');
  d.textContent = s || '';
  return d.innerHTML;
}

function vote(idx, v) {
  const card = currentCards[idx];
  const el = document.getElementById(`card-${idx}`);
  const noteEl = document.getElementById(`note-${idx}`);

  // Remove previous vote for this card in this batch
  const existingIdx = votes.findIndex(x => x.name === card.name && x._batch === batchNum);
  if (existingIdx !== -1) votes.splice(existingIdx, 1);

  el.classList.remove('voted-good', 'voted-bad');
  el.classList.add(v === 'good' ? 'voted-good' : 'voted-bad');

  // Toggle button active states
  el.querySelectorAll('.btn-good, .btn-bad').forEach(b => b.classList.remove('active'));
  el.querySelector(v === 'good' ? '.btn-good' : '.btn-bad').classList.add('active');

  // Show note field for bad votes
  if (v === 'bad') {
    noteEl.classList.add('visible');
    noteEl.focus();
  } else {
    noteEl.classList.remove('visible');
  }

  votes.push({
    name: card.name,
    set: card.set,
    type_line: card.type_line,
    oracle_text: card.oracle_text,
    vote: v,
    confidence: card.confidence,
    mechanics: card.mechanics,
    parameters: card.parameters || {},
    unparsed_text: card.unparsed_text || '',
    note: '',
    _batch: batchNum,
  });
  updateStats();
}

function updateNote(idx) {
  const note = document.getElementById(`note-${idx}`).value;
  const card = currentCards[idx];
  const entry = votes.find(x => x.name === card.name && x._batch === batchNum);
  if (entry) entry.note = note;
}

function updateStats() {
  const total = votes.length;
  const good = votes.filter(v => v.vote === 'good').length;
  const bad = total - good;
  document.getElementById('stat-total').textContent = total;
  document.getElementById('stat-good').textContent = good;
  document.getElementById('stat-bad').textContent = bad;
  document.getElementById('stat-rate').textContent = total ? Math.round(good / total * 100) + '%' : '—';
}

async function loadBatch() {
  batchNum++;
  const el = document.getElementById('content');
  el.innerHTML = '<div class="loading"><div class="spinner"></div><div>Fetching cards from Scryfall...</div></div>';
  document.getElementById('actions').style.display = 'none';
  try {
    const resp = await fetch('/api/cards');
    const cards = await resp.json();
    renderGrid(cards);
  } catch (e) {
    el.innerHTML = `<div class="loading">Error fetching cards: ${e.message}</div>`;
    document.getElementById('actions').style.display = 'flex';
  }
}

function buildReportData() {
  const good = votes.filter(v => v.vote === 'good');
  const bad = votes.filter(v => v.vote === 'bad');

  // Mechanic miss frequency (from bad votes)
  const mechMiss = {};
  bad.forEach(v => {
    v.mechanics.forEach(m => { mechMiss[m] = (mechMiss[m] || 0) + 1; });
  });
  // Mechanic hit frequency (from good votes)
  const mechHit = {};
  good.forEach(v => {
    v.mechanics.forEach(m => { mechHit[m] = (mechHit[m] || 0) + 1; });
  });

  // Confidence comparison
  const avgConfGood = good.length ? good.reduce((s, v) => s + v.confidence, 0) / good.length : 0;
  const avgConfBad = bad.length ? bad.reduce((s, v) => s + v.confidence, 0) / bad.length : 0;

  return {
    total_reviewed: votes.length,
    approved: good.length,
    rejected: bad.length,
    approval_rate: votes.length ? +(good.length / votes.length).toFixed(3) : 0,
    avg_confidence_approved: +avgConfGood.toFixed(3),
    avg_confidence_rejected: +avgConfBad.toFixed(3),
    rejected_cards: bad.map(v => ({
      name: v.name,
      set: v.set,
      confidence: v.confidence,
      mechanics: v.mechanics,
      unparsed: v.unparsed_text.substring(0, 100),
      note: v.note,
    })),
    mechanics_on_rejected: Object.entries(mechMiss)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 20)
      .map(([m, c]) => ({mechanic: m, count: c})),
  };
}

function showReport() {
  if (!votes.length) { alert('Vote on some cards first!'); return; }
  const data = buildReportData();
  let text = `EMBEDDING QUIZ REPORT\n${'='.repeat(50)}\n\n`;
  text += `Total reviewed:   ${data.total_reviewed}\n`;
  text += `Approved:         ${data.approved}\n`;
  text += `Rejected:         ${data.rejected}\n`;
  text += `Approval rate:    ${Math.round(data.approval_rate * 100)}%\n\n`;
  text += `Avg confidence (approved): ${(data.avg_confidence_approved * 100).toFixed(1)}%\n`;
  text += `Avg confidence (rejected): ${(data.avg_confidence_rejected * 100).toFixed(1)}%\n`;

  if (data.rejected_cards.length) {
    text += `\nREJECTED CARDS\n${'-'.repeat(50)}\n`;
    data.rejected_cards.forEach(c => {
      text += `  ${c.name} (${c.set}) — ${Math.round(c.confidence * 100)}% conf\n`;
      if (c.note) text += `    Note: ${c.note}\n`;
      if (c.unparsed) text += `    Unparsed: ${c.unparsed}\n`;
      text += `    Mechanics: ${c.mechanics.slice(0, 6).join(', ')}\n`;
    });
  }

  if (data.mechanics_on_rejected.length) {
    text += `\nMECHANICS APPEARING ON REJECTED CARDS\n${'-'.repeat(50)}\n`;
    data.mechanics_on_rejected.forEach(({mechanic, count}) => {
      text += `  [${count}x] ${mechanic}\n`;
    });
  }

  document.getElementById('report-text').textContent = text;
  document.getElementById('overlay').classList.add('visible');
}

function closeReport() {
  document.getElementById('overlay').classList.remove('visible');
}

function downloadReport() {
  const data = buildReportData();
  const blob = new Blob([JSON.stringify(data, null, 2)], {type: 'application/json'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `embedding_quiz_report_${new Date().toISOString().slice(0,10)}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

// Initial load
loadBatch();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------

class QuizHandler(BaseHTTPRequestHandler):
    """Serves the quiz page and API endpoints."""

    fmt = "standard"
    set_code = None

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self._serve_html()
        elif self.path == "/api/cards":
            self._serve_cards()
        else:
            self.send_error(404)

    def _serve_html(self):
        body = HTML_PAGE.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_cards(self):
        cards = fetch_random_cards(9, fmt=self.fmt, set_code=self.set_code)
        body = json.dumps(cards).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        # Suppress request logging noise
        pass


def main():
    parser = argparse.ArgumentParser(description="Embedding Quiz — review card parser accuracy")
    parser.add_argument("--port", "-p", type=int, default=8787)
    parser.add_argument("--format", "-f", type=str, default="standard",
                        help="Card format to sample from (standard, modern, commander)")
    parser.add_argument("--set", "-s", type=str, default=None,
                        help="Specific set code (e.g., FDN, DSK)")
    args = parser.parse_args()

    QuizHandler.fmt = args.format
    QuizHandler.set_code = args.set

    server = HTTPServer(("127.0.0.1", args.port), QuizHandler)
    label = f"set:{args.set}" if args.set else f"format:{args.format}"
    print(f"Embedding Quiz running at http://localhost:{args.port}")
    print(f"Sampling from: {label}")
    print(f"Press Ctrl+C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
