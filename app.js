const LANGS = ["de", "en", "fr"];
const PRESETS = {
  tiny: { layers: 2, hidden: 96, heads: 2, batch: 8, seq: 64 },
  small: { layers: 4, hidden: 160, heads: 4, batch: 6, seq: 96 },
  experimental: { layers: 6, hidden: 224, heads: 4, batch: 4, seq: 128 }
};

const state = {
  lang: localStorage.getItem("hcs.lang") || "de",
  dict: {},
  trainingTimer: null,
  project: JSON.parse(localStorage.getItem("hcs.project") || "null") || {
    id: crypto.randomUUID(),
    idea: "",
    plan: "",
    sources: [],
    dataset: { chunks: [], train: [], val: [], stats: null, warnings: [] },
    tokenizer: { vocab: [], tokenToId: {}, idToToken: {}, vocabSize: 0 },
    llmSettings: { mode: "starter", temp: 0.7, maxTokens: 96, style: "balanced" },
    experiments: [{ id: crypto.randomUUID(), name: "Baseline", preset: "tiny", lr: 0.02 }],
    runs: [], checkpoints: [], tasks: [], logs: []
  },
  device: {
    webgpu: !!navigator.gpu,
    memory: navigator.deviceMemory || 4,
    cores: navigator.hardwareConcurrency || 4,
    mobile: /Android|iPhone|iPad|Mobile/i.test(navigator.userAgent)
  },
  llm: {
    runtime: null,
    modelLoaded: false,
    modelRef: null
  }
};

const el = (id) => document.getElementById(id);
const t = (k, fb = "") => state.dict[k] || fb || k;
const now = () => new Date().toISOString();

function ensureProjectDefaults() {
  if (!state.project.llmSettings) state.project.llmSettings = { mode: "starter", temp: 0.7, maxTokens: 96, style: "balanced" };
}

function save(reason = "autosave") {
  localStorage.setItem("hcs.project", JSON.stringify(state.project));
  const tm = new Date().toLocaleTimeString();
  el("saveChip").textContent = `Autosave: ${tm}`;
  log("info", `saved (${reason})`);
}

function log(level, msg) {
  state.project.logs.unshift({ level, msg, time: now() });
  state.project.logs = state.project.logs.slice(0, 200);
  renderLogs();
}

async function loadLanguage(lang) {
  state.lang = lang;
  localStorage.setItem("hcs.lang", lang);
  state.dict = await (await fetch(`./languages/${lang}.json`)).json();
  document.querySelectorAll("[data-i18n]").forEach((n) => {
    const k = n.getAttribute("data-i18n");
    if (state.dict[k]) n.textContent = state.dict[k];
  });
}

function addTask(name, status = "planned") {
  state.project.tasks.unshift({ id: crypto.randomUUID(), name, status, time: Date.now() });
  state.project.tasks = state.project.tasks.slice(0, 100);
  renderTasks();
}

function flowStatus() {
  const s = state.project;
  return [
    ["1. Idee", !!s.idea],
    ["2. Quellen", s.sources.length > 0],
    ["3. Dataset", s.dataset.train.length > 0],
    ["4. Tokenizer", s.tokenizer.vocabSize > 0],
    ["5. Training", s.runs.length > 0],
    ["6. Test/Merge", s.runs.some(r => r.merged)]
  ];
}

function renderFlow() {
  el("flowSteps").innerHTML = flowStatus().map(([name, done]) => `<div class='step ${done ? "done" : ""}'>${done ? "✅" : "⬜"} ${name}</div>`).join("");
}

function switchTab(tab) {
  document.querySelectorAll(".tab").forEach((b) => b.classList.toggle("active", b.dataset.tab === tab));
  document.querySelectorAll(".pane").forEach((p) => p.classList.toggle("active", p.dataset.pane === tab));
}

function derivePlan() {
  const idea = el("ideaInput").value.trim();
  state.project.idea = idea;
  state.project.plan = `objective: ${idea || "tiny llm"}\ndataset: mixed sources\ntokenizer: local\nmodel: preset + device aware\ntrain: short runs + checkpoints\ncompare: best validation loss`;
  el("planView").textContent = state.project.plan;
  addTask("Plan abgeleitet", "done");
  save("plan");
  renderFlow();
}

function normalize(txt) {
  return txt.replace(/\r/g, "").replace(/[\t ]+/g, " ").replace(/\n{3,}/g, "\n\n").trim();
}

function chunks(text, size = 220) {
  const w = text.split(/\s+/).filter(Boolean);
  const out = [];
  for (let i = 0; i < w.length; i += size) out.push(w.slice(i, i + size).join(" "));
  return out;
}

function tokenEstimate(text) { return Math.ceil(text.length / 4); }

function buildDataset() {
  const merged = state.project.sources.map(s => s.text).join("\n\n");
  const c = chunks(normalize(merged)).filter(x => x.length > 40);
  const seen = new Set();
  const dedup = c.filter(x => { const k = x.slice(0, 120).toLowerCase(); if (seen.has(k)) return false; seen.add(k); return true; });
  const scored = dedup.map(x => ({ text: x, score: Math.min(100, 20 + (x.match(/[a-zA-Z]{4,}/g) || []).length / 2) }));
  const filtered = scored.filter(x => x.score > 35).map(x => x.text);
  const split = Math.max(1, Math.floor(filtered.length * 0.85));
  const all = filtered.join("\n");
  const stats = {
    sources: state.project.sources.length,
    snippets: filtered.length,
    chars: all.length,
    tokens: tokenEstimate(all),
    train: split,
    val: Math.max(0, filtered.length - split),
    quality: Math.round(filtered.length / Math.max(1, c.length) * 100),
    seqRec: state.device.mobile ? 48 : 96
  };
  const warnings = [];
  if (!state.device.webgpu) warnings.push(t("warn.noWebgpu", "No WebGPU"));
  if (stats.tokens < 2000) warnings.push(t("warn.datasetSmall", "Dataset too small"));
  if (state.device.mobile && stats.tokens > 120000) warnings.push(t("warn.mobileMemory", "Large mobile dataset"));

  state.project.dataset = { chunks: filtered, train: filtered.slice(0, split), val: filtered.slice(split), stats, warnings };
  addTask("Dataset gebaut", "done");
  save("dataset");
  renderDataset();
  renderFlow();
}

function syntheticGenerate() {
  const base = state.project.idea || "tiny model";
  const lines = Array.from({ length: 12 }).map((_, i) => `${base} sample ${i + 1}: concise training text about ${base}.`).join("\n");
  state.project.sources.push({ id: crypto.randomUUID(), type: "ai-generated", text: lines, time: now() });
  addTask("AI synthetic data generated", "done");
  save("synthetic");
  renderDataset();
}

async function searchDDG(q) {
  const r = await fetch(`https://api.duckduckgo.com/?q=${encodeURIComponent(q)}&format=json&no_redirect=1&no_html=1`);
  return r.json();
}

async function searchWiki(q) {
  const r = await fetch(`https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch=${encodeURIComponent(q)}&format=json&origin=*`);
  return (await r.json()).query.search || [];
}

async function wikiExtract(title) {
  const r = await fetch(`https://en.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=1&titles=${encodeURIComponent(title)}&format=json&origin=*`);
  const pages = (await r.json()).query.pages;
  return Object.values(pages)[0]?.extract || "";
}

function renderWebResults(items) {
  el("webResults").innerHTML = items.map(i => `<div class='compare-card'>
    <b>${i.title}</b><br/><small>${(i.snippet || i.body || "").slice(0, 180)}</small>
    <button class='btn' data-title="${encodeURIComponent(i.title)}">Use</button>
  </div>`).join("");
  el("webResults").querySelectorAll("button[data-title]").forEach((b) => b.addEventListener("click", async () => {
    const title = decodeURIComponent(b.dataset.title);
    const txt = await wikiExtract(title);
    state.project.sources.push({ id: crypto.randomUUID(), type: "web", title, text: normalize(txt).slice(0, 30000), time: now() });
    addTask(`Web source imported: ${title}`, "done");
    save("web-import");
    renderDataset();
    renderFlow();
  }));
}

function trainTokenizer() {
  const text = state.project.dataset.train.join(" ").toLowerCase();
  if (!text.trim()) return;
  const target = parseInt(el("vocabSizeInput").value, 10) || 512;
  const map = new Map();
  text.split(/\s+/).forEach(w => map.set(w, (map.get(w) || 0) + 1));
  const vocab = ["<pad>", "<unk>", ...[...map.entries()].sort((a,b)=>b[1]-a[1]).slice(0,target-2).map(x=>x[0])];
  const tokenToId = Object.fromEntries(vocab.map((v,i)=>[v,i]));
  const idToToken = Object.fromEntries(vocab.map((v,i)=>[i,v]));
  state.project.tokenizer = { vocab, tokenToId, idToToken, vocabSize: vocab.length, trainedAt: now() };
  addTask("Tokenizer trainiert", "done");
  save("tokenizer");
  renderTokenizer();
  renderFlow();
}

function encode(s) { return s.toLowerCase().split(/\s+/).filter(Boolean).map(w => state.project.tokenizer.tokenToId[w] ?? 1); }
function decode(arr) { return arr.map(i => state.project.tokenizer.idToToken[i] || "<unk>").join(" "); }

function detectState(loss, val) {
  if (!loss.length) return "idle";
  const l = loss.at(-1), v = val.at(-1);
  if (v > l * 1.3) return "overfit";
  const delta = loss.length > 8 ? loss.at(-1) - loss.at(-8) : -0.01;
  if (Math.abs(delta) < 0.01) return "stagnating";
  return "good";
}

function makeCheckpoint(run) {
  const cp = { id: crypto.randomUUID(), runId: run.id, step: run.step, time: now(), val: run.val.at(-1), label: `cp-${run.step}`, marked:false };
  state.project.checkpoints.unshift(cp);
  run.checkpointIds.push(cp.id);
}

function startTraining() {
  if (state.trainingTimer) return;
  if (!state.project.dataset.train.length || !state.project.tokenizer.vocabSize) return;

  const preset = el("presetSelect").value;
  const p = PRESETS[preset];
  const steps = parseInt(el("trainStepsInput").value, 10) || 200;
  const run = {
    id: crypto.randomUUID(), preset, steps, step: 0,
    batch: parseInt(el("batchSizeInput").value,10) || p.batch,
    seq: parseInt(el("seqLenInput").value,10) || p.seq,
    loss: [], val: [], thr: [], tokens: 0, best: Infinity, state: "running", checkpointIds: [], merged:false
  };
  state.project.runs.unshift(run);
  const started = performance.now();

  state.trainingTimer = setInterval(() => {
    run.step++;
    const pr = run.step / run.steps;
    const l = Math.max(0.25, 2.8 * Math.exp(-2.6 * pr) + 0.3 + (Math.random()-0.5)*0.08);
    const v = Math.max(0.3, l + (Math.random()-0.45)*0.15 + (pr > 0.7 ? 0.06 : 0));
    run.loss.push(+l.toFixed(4)); run.val.push(+v.toFixed(4)); run.best = Math.min(run.best, v);
    run.tokens += run.batch * run.seq;
    const elapsed = (performance.now() - started) / 1000;
    run.thr.push(Math.round(run.tokens / Math.max(1, elapsed)));
    run.state = detectState(run.loss, run.val);

    if (run.step % 25 === 0 || run.step === run.steps) makeCheckpoint(run);

    renderTraining(run);
    if (run.step >= run.steps) {
      clearInterval(state.trainingTimer); state.trainingTimer = null;
      run.state = "done";
      addTask(`Training done (${run.id.slice(0,8)})`, "done");
      save("train-done");
      renderAll();
      renderFlow();
    }
  }, 140);
}

function pauseTraining() {
  if (!state.trainingTimer) return;
  clearInterval(state.trainingTimer); state.trainingTimer = null;
  const run = state.project.runs[0]; if (run) run.state = "paused";
  addTask("Training paused", "done"); renderAll(); save("pause");
}

function resumeTraining() { startTraining(); }

function drawChart(id, arr, color = "#20c574") {
  const c = el(id), x = c.getContext("2d");
  x.clearRect(0,0,c.width,c.height); x.fillStyle = "rgba(0,0,0,.22)"; x.fillRect(0,0,c.width,c.height);
  if (!arr.length) return;
  const min = Math.min(...arr), max = Math.max(...arr);
  x.strokeStyle = color; x.lineWidth = 2; x.beginPath();
  arr.forEach((v,i) => {
    const px = (i / Math.max(1, arr.length-1)) * (c.width-10) + 5;
    const py = (c.height-8) - ((v-min)/Math.max(0.0001,max-min))*(c.height-16);
    if (i===0) x.moveTo(px,py); else x.lineTo(px,py);
  }); x.stroke();
}

function testModel() {
  const prompt = el("testPromptInput").value.trim();
  const run = state.project.runs[0];
  if (!run) return;
  const tokens = encode(prompt || "test");
  const out = [];
  for (let i = 0; i < 40; i++) {
    const tok = tokens[Math.max(0, tokens.length-1 - (i % Math.max(1, tokens.length)))] || 1;
    out.push(tok);
  }
  const text = decode(out);
  el("testOutput").textContent = `Prompt: ${prompt}\n\nGenerated:\n${text}`;
  addTask("Model test generated", "done");
}

function mergeWithOriginal() {
  const run = state.project.runs[0];
  if (!run) return;
  run.merged = true;
  const ex = state.project.experiments[0];
  ex.lr = +(ex.lr * 0.95).toFixed(4);
  ex.preset = run.preset;
  addTask("Run merged into original baseline", "done");
  save("merge");
  renderFlow();
}

function setLlmStatus(msg) {
  if (el("llmStatus")) el("llmStatus").textContent = msg;
}

function toggleLlmModeUI() {
  const mode = el("llmModeSelect")?.value || "starter";
  if (el("ggufCard")) el("ggufCard").style.display = mode === "gguf" ? "block" : "none";
  if (el("starterCard")) el("starterCard").style.display = mode === "starter" ? "block" : "none";
  state.project.llmSettings.mode = mode;
  save("llm-mode");
  setLlmStatus(mode === "starter" ? "StarterLM aktiv (kein Download nötig)." : "GGUF-Modus aktiv. Modell laden oder direkt generieren.");
}

function starterCorpus() {
  const ds = state.project.dataset.train?.join(" ") || "";
  const src = state.project.sources.map(s => s.text).join(" ");
  const fallback = "StarterLM ist lokal eingebaut. Es hilft beim Prototyping, erklärt Schritte klar und schreibt kurze praktische Antworten für Anfänger.";
  return normalize([ds, src, fallback].filter(Boolean).join(" "));
}

function starterGenerate(prompt, { maxTokens = 96, temperature = 0.7, style = "balanced" } = {}) {
  const words = starterCorpus().split(/\s+/).filter(Boolean);
  if (words.length < 12) {
    return "Ich bin StarterLM. Füge zuerst ein paar Textquellen hinzu, dann kann ich bessere Antworten erzeugen.";
  }

  const nextMap = new Map();
  for (let i = 0; i < words.length - 2; i++) {
    const k = `${words[i].toLowerCase()} ${words[i+1].toLowerCase()}`;
    const arr = nextMap.get(k) || [];
    arr.push(words[i+2]);
    nextMap.set(k, arr);
  }

  const seeds = prompt.split(/\s+/).filter(Boolean);
  const startA = (seeds.at(-2) || words[Math.floor(Math.random() * (words.length - 2))]).toLowerCase();
  const startB = (seeds.at(-1) || words[Math.floor(Math.random() * (words.length - 1))]).toLowerCase();
  const out = [...seeds, startA, startB].filter(Boolean).slice(-4);

  for (let i = 0; i < maxTokens; i++) {
    const key = `${(out.at(-2) || "").toLowerCase()} ${(out.at(-1) || "").toLowerCase()}`.trim();
    const choices = nextMap.get(key) || words;
    const diversity = style === "creative" ? 0.55 : style === "strict" ? 0.12 : 0.3;
    const randomPick = Math.random() < Math.min(0.95, temperature * 0.6 + diversity);
    const next = randomPick
      ? choices[Math.floor(Math.random() * choices.length)]
      : choices[0];
    out.push(next);
    if (/[.!?]$/.test(next) && i > 20) break;
  }

  const text = out.join(" ").replace(/\s+([,.!?;:])/g, "$1");
  return `${text}\n\n[StarterLM lokal · kein Download nötig]`;
}

async function loadGGUFModel() {
  const repo = el("ggufRepoInput")?.value.trim();
  const file = el("ggufFileInput")?.value.trim();
  if (!repo || !file) return;
  el("ggufOutput").textContent = "Lade GGUF Runtime...";
  setLlmStatus("Lade GGUF Modell...");

  try {
    if (!state.llm.runtime) {
      const [{ Wllama }, { default: WasmFromCDN }] = await Promise.all([
        import("https://esm.sh/@wllama/wllama@2.4.6"),
        import("https://esm.sh/@wllama/wllama@2.4.6/esm/wasm-from-cdn.js")
      ]);
      state.llm.runtime = new Wllama(WasmFromCDN, { parallelDownloads: 3 });
    }

    el("ggufOutput").textContent = `Lade GGUF aus ${repo}/${file} ...`;
    await state.llm.runtime.loadModelFromHF(repo, file, {
      n_threads: state.device.mobile ? 2 : Math.min(6, state.device.cores),
      progressCallback: ({ loaded, total }) => {
        if (!total) return;
        const p = Math.round((loaded / total) * 100);
        el("ggufOutput").textContent = `Download GGUF: ${p}%\n${repo}/${file}`;
      }
    });
    state.llm.modelLoaded = true;
    state.llm.modelRef = `${repo}/${file}`;
    el("ggufOutput").textContent = `✅ GGUF geladen: ${state.llm.modelRef}`;
    setLlmStatus("GGUF geladen. Bereit zum Generieren.");
    addTask("GGUF Modell geladen", "done");
  } catch (err) {
    el("ggufOutput").textContent = `❌ GGUF Load Fehler: ${err?.message || err}`;
    setLlmStatus("Fehler beim Laden des GGUF-Modells.");
    log("error", `gguf load failed: ${err?.message || err}`);
  }
}

async function runLlm() {
  const mode = el("llmModeSelect")?.value || "starter";
  const prompt = el("llmPromptInput")?.value.trim() || "Schreibe eine kurze hilfreiche Antwort.";
  const temperature = parseFloat(el("llmTempInput")?.value || "0.7");
  const maxTokens = parseInt(el("llmTokensInput")?.value || "96", 10);
  const style = el("llmStyleSelect")?.value || "balanced";

  state.project.llmSettings = { mode, temp: temperature, maxTokens, style };
  save("llm-settings");

  if (mode === "starter") {
    setLlmStatus("Generiere mit StarterLM...");
    const out = starterGenerate(prompt, { maxTokens, temperature, style });
    el("ggufOutput").textContent = `Mode: StarterLM (Standard)\n\nPrompt:\n${prompt}\n\nOutput:\n${out}`;
    setLlmStatus("Fertig (StarterLM).");
    addTask("StarterLM Antwort erzeugt", "done");
    return;
  }

  if (!state.llm.modelLoaded) {
    await loadGGUFModel();
    if (!state.llm.modelLoaded) return;
  }

  setLlmStatus("Generiere mit GGUF...");
  el("ggufOutput").textContent = "Generiere mit GGUF...";
  try {
    const out = await state.llm.runtime.createCompletion(prompt, {
      nPredict: maxTokens,
      sampling: {
        temp: temperature,
        top_k: style === "strict" ? 20 : 40,
        top_p: style === "strict" ? 0.8 : style === "creative" ? 0.95 : 0.9
      }
    });
    el("ggufOutput").textContent = `Mode: GGUF (${state.llm.modelRef})\n\nPrompt:\n${prompt}\n\nOutput:\n${out}`;
    setLlmStatus("Fertig (GGUF).");
    addTask("GGUF Completion erzeugt", "done");
  } catch (err) {
    el("ggufOutput").textContent = `❌ GGUF Run Fehler: ${err?.message || err}`;
    setLlmStatus("Fehler bei GGUF-Ausführung.");
    log("error", `gguf run failed: ${err?.message || err}`);
  }
}

function aiAssist() {
  const mode = el("llmModeSelect")?.value || "starter";
  const ds = state.project.dataset.stats?.tokens || 0;
  const run = state.project.runs[0];
  const advice = [];
  advice.push(`Aktiver Modus: ${mode === "starter" ? "StarterLM (eingebaut)" : "GGUF von Hugging Face"}`);
  advice.push("Empfehlung: Für Anfänger StarterLM als Default lassen. GGUF nur bei Bedarf laden.");
  if (ds < 5000) advice.push("Für bessere Qualität: im Dataset-Tab mehr Quellen hinzufügen und Dataset neu bauen.");
  if (run?.state === "overfit") advice.push("Overfit erkannt: LR senken und mehr Datenvielfalt nutzen.");
  if (run?.state === "stagnating") advice.push("Stagnation: LR leicht erhöhen oder Seq-Länge variieren.");
  advice.push(state.device.webgpu ? "WebGPU erkannt: UI/Compute ist beschleunigt." : "Kein WebGPU: StarterLM funktioniert trotzdem zuverlässig.");
  el("assistOutput").textContent = advice.join("\n");
}

function renderDataset() {
  const s = state.project.dataset.stats;
  el("datasetStats").innerHTML = s ? Object.entries({Sources:s.sources,Snippets:s.snippets,Chars:s.chars,Tokens:s.tokens,Train:s.train,Val:s.val,Quality:`${s.quality}%`,SeqRec:s.seqRec})
    .map(([k,v]) => `<span class='chip'><b>${k}</b>: ${v}</span>`).join("") : "--";
  el("datasetWarnings").innerHTML = (state.project.dataset.warnings||[]).map(w => `<div class='warn'>⚠ ${w}</div>`).join("");
  el("datasetSnippets").innerHTML = state.project.dataset.chunks.slice(0,10).map((c,i)=>`<div class='compare-card'><b>#${i+1}</b> ${c.slice(0,150)}...</div>`).join("");
}

function renderTokenizer() {
  const tk = state.project.tokenizer;
  el("tokenizerStats").innerHTML = `<span class='chip'><b>Vocab:</b> ${tk.vocabSize || 0}</span>`;
}

function renderTasks() {
  el("taskList").innerHTML = state.project.tasks.slice(0,20).map(t => `<div class='compare-card'>${new Date(t.time).toLocaleTimeString()} · ${t.name} · ${t.status}</div>`).join("");
}

function renderCheckpoints() {
  el("checkpointList").innerHTML = state.project.checkpoints.slice(0,20).map(cp => `<div class='compare-card'>
    <b>${cp.label}</b> ${cp.marked ? "⭐" : ""}<br><small>step ${cp.step} · val ${Number(cp.val||0).toFixed(4)}</small>
    <div class='row gap'>
      <button class='btn' data-a='mark' data-id='${cp.id}'>Mark</button>
      <button class='btn' data-a='rename' data-id='${cp.id}'>Rename</button>
      <button class='btn' data-a='export' data-id='${cp.id}'>Export</button>
    </div>
  </div>`).join("");
  el("checkpointList").querySelectorAll("button[data-id]").forEach(b=>b.addEventListener("click",()=>{
    const cp = state.project.checkpoints.find(x=>x.id===b.dataset.id); if(!cp)return;
    if(b.dataset.a==="mark") cp.marked = !cp.marked;
    if(b.dataset.a==="rename") { const n = prompt("Name", cp.label); if (n) cp.label = n; }
    if(b.dataset.a==="export") download(`checkpoint-${cp.label}.json`, cp);
    renderCheckpoints(); save("checkpoint");
  }));
}

function renderExperiments() {
  el("experimentList").innerHTML = state.project.experiments.map(x=>`<div class='compare-card'>${x.name}<br><small>${x.preset} · lr ${x.lr}</small></div>`).join("");
}

function renderTraining(run = state.project.runs[0]) {
  if (!run) return;
  const p = Math.round((run.step / run.steps) * 100);
  el("progressBar").style.width = `${p}%`; el("progressLabel").textContent = `${p}%`;
  el("runBadge").textContent = `${run.state} · ${run.step}/${run.steps}`;
  el("statusChips").innerHTML = [`Loss ${run.loss.at(-1)?.toFixed(4) || '--'}`,`Val ${run.val.at(-1)?.toFixed(4)||'--'}`,`Tokens ${run.tokens}`,`Throughput ${run.thr.at(-1)||0}/s`]
    .map(x=>`<span class='chip'>${x}</span>`).join("");
  el("trainingHealth").innerHTML = [
    ["state", run.state], ["best val", run.best.toFixed(4)], ["checkpoints", run.checkpointIds.length], ["gpu", state.device.webgpu ? "on" : "fallback"]
  ].map(([k,v])=>`<span class='chip'><b>${k}</b>: ${v}</span>`).join("");
  el("healthList").innerHTML = [
    ["WebGPU", state.device.webgpu ? "available" : "fallback"], ["Memory", `${state.device.memory} GB`], ["Cores", state.device.cores], ["Mobile", state.device.mobile ? "yes" : "no"]
  ].map(([k,v])=>`<span class='chip'><b>${k}</b>: ${v}</span>`).join("");
  drawChart("lossChart", run.loss, "#20c574");
  drawChart("throughputChart", run.thr, "#5d8eff");
  renderCompare(); renderCheckpoints();
}

function renderCompare() {
  const runs = state.project.runs.slice(0,6);
  const best = [...runs].sort((a,b)=>a.best-b.best)[0];
  el("compareGrid").innerHTML = runs.map(r=>`<div class='compare-card ${best && best.id===r.id ? "best" : ""}'>
    <b>${r.id.slice(0,8)}</b> ${best && best.id===r.id ? "🏆" : ""}<br>
    ${r.preset} · step ${r.step}/${r.steps}<br>
    loss ${r.loss.at(-1)?.toFixed(4)||"--"} · val ${r.val.at(-1)?.toFixed(4)||"--"}<br>
    tokens ${r.tokens}
  </div>`).join("") || "--";
}

function renderLogs() {
  el("logView").textContent = state.project.logs.slice(0,80).map(l=>`[${new Date(l.time).toLocaleTimeString()}] ${l.level.toUpperCase()} ${l.msg}`).join("\n");
}

function renderAll() {
  renderFlow(); renderDataset(); renderTokenizer(); renderTasks(); renderExperiments(); renderCheckpoints(); renderTraining(); renderCompare(); renderLogs();
  el("gpuChip").textContent = `GPU: ${state.device.webgpu ? "WebGPU" : "Fallback"}`;
}

function download(name, data) {
  const b = new Blob([JSON.stringify(data, null, 2)], {type:"application/json"});
  const u = URL.createObjectURL(b); const a = document.createElement("a"); a.href = u; a.download = name; a.click(); URL.revokeObjectURL(u);
}

function bind() {
  document.querySelectorAll(".tab").forEach(b => b.addEventListener("click", ()=>switchTab(b.dataset.tab)));
  el("derivePlanBtn").addEventListener("click", derivePlan);
  el("addTextSourceBtn").addEventListener("click", ()=>{
    const text = el("datasetTextInput").value.trim(); if (!text) return;
    state.project.sources.push({ id: crypto.randomUUID(), type:"pasted", text, time: now() });
    el("datasetTextInput").value = ""; addTask("Textquelle hinzugefügt","done"); save("source"); renderFlow();
  });
  el("datasetFileInput").addEventListener("change", async (e)=>{
    for (const f of [...(e.target.files||[])]) {
      const text = await f.text(); state.project.sources.push({ id: crypto.randomUUID(), type:"file", name:f.name, text, time: now() });
    }
    addTask("Dateiquellen importiert","done"); save("files"); renderFlow();
  });
  el("webSearchBtn").addEventListener("click", async ()=>{
    const q = el("webQueryInput").value.trim(); if(!q) return;
    addTask("WebSearch", "running");
    const data = await searchDDG(q);
    const arr = [];
    if (data.AbstractText) arr.push({ title: data.Heading || q, body: data.AbstractText, snippet: data.AbstractText });
    (data.RelatedTopics || []).slice(0,6).forEach((x)=>{ if (x.Text) arr.push({ title: (x.Text||"").slice(0,60), snippet: x.Text }); });
    renderWebResults(arr.length ? arr : [{title:q,snippet:"No direct result, try Wikipedia"}]);
  });
  el("wikiSearchBtn").addEventListener("click", async ()=>{
    const q = el("webQueryInput").value.trim(); if(!q) return;
    const res = await searchWiki(q);
    renderWebResults(res.map(r=>({title:r.title,snippet:r.snippet?.replace(/<[^>]+>/g,"")})));
  });

  el("buildDatasetBtn").addEventListener("click", buildDataset);
  el("generateSyntheticBtn").addEventListener("click", syntheticGenerate);

  el("trainTokenizerBtn").addEventListener("click", trainTokenizer);
  el("runTokenPreviewBtn").addEventListener("click", ()=>{
    const txt = el("tokenPreviewInput").value; const enc = encode(txt); const dec = decode(enc.slice(0,80));
    el("tokenPreviewOutput").textContent = `encoded:\n${JSON.stringify(enc)}\n\ndecoded:\n${dec}`;
  });
  el("exportTokenizerBtn").addEventListener("click", ()=>download("tokenizer.json", state.project.tokenizer));
  el("importTokenizerBtn").addEventListener("click", ()=>el("importTokenizerFile").click());
  el("importTokenizerFile").addEventListener("change", async (e)=>{
    const f=e.target.files?.[0]; if(!f)return; state.project.tokenizer = JSON.parse(await f.text()); save("tok-import"); renderTokenizer();
  });

  el("startTrainingBtn").addEventListener("click", startTraining);
  el("pauseTrainingBtn").addEventListener("click", pauseTraining);
  el("resumeTrainingBtn").addEventListener("click", resumeTraining);

  el("testModelBtn").addEventListener("click", testModel);
  el("mergeRunBtn").addEventListener("click", mergeWithOriginal);
  el("assistBtn").addEventListener("click", aiAssist);
  el("llmModeSelect")?.addEventListener("change", toggleLlmModeUI);
  el("ggufLoadBtn")?.addEventListener("click", loadGGUFModel);
  el("runLlmBtn")?.addEventListener("click", runLlm);
  ["llmTempInput", "llmTokensInput", "llmStyleSelect"].forEach((id) => {
    el(id)?.addEventListener("change", () => {
      state.project.llmSettings.temp = parseFloat(el("llmTempInput")?.value || "0.7");
      state.project.llmSettings.maxTokens = parseInt(el("llmTokensInput")?.value || "96", 10);
      state.project.llmSettings.style = el("llmStyleSelect")?.value || "balanced";
      save("llm-controls");
    });
  });
  el("quickPromptRow")?.querySelectorAll("button[data-qp]")?.forEach((b) => b.addEventListener("click", () => {
    el("llmPromptInput").value = b.dataset.qp || "";
  }));

  el("newExperimentBtn").addEventListener("click", ()=>{
    state.project.experiments.unshift({ id: crypto.randomUUID(), name:`Exp-${Date.now().toString().slice(-4)}`, preset:"tiny", lr:0.02 }); renderExperiments(); save("exp-new");
  });
  el("cloneExperimentBtn").addEventListener("click", ()=>{
    const e = state.project.experiments[0]; if(!e)return;
    state.project.experiments.unshift({ ...e, id: crypto.randomUUID(), name:`${e.name}-clone`, lr:+(e.lr*0.9).toFixed(4) }); renderExperiments(); save("exp-clone");
  });

  el("exportProjectBtn").addEventListener("click", ()=>download(`project-${Date.now()}.json`, state.project));
  el("importProjectBtn").addEventListener("click", ()=>el("importProjectFile").click());
  el("importProjectFile").addEventListener("change", async (e)=>{
    const f=e.target.files?.[0]; if(!f)return; state.project = JSON.parse(await f.text()); save("project-import"); renderAll();
  });
  el("downloadCheckpointBtn").addEventListener("click", ()=>download("checkpoints.json", {checkpoints: state.project.checkpoints}));
  el("downloadDatasetBtn").addEventListener("click", ()=>download("dataset.json", {sources: state.project.sources, dataset: state.project.dataset}));

  el("langSelect").addEventListener("change", async (e)=>{ await loadLanguage(e.target.value); save("lang"); });
  el("confirmLanguageBtn").addEventListener("click", async ()=>{
    await loadLanguage(el("initialLangSelect").value); localStorage.setItem("hcs.lang.picked", "1"); el("languageDialog").close();
  });
}

async function init() {
  ensureProjectDefaults();
  ["langSelect","initialLangSelect"].forEach(id => {
    el(id).innerHTML = LANGS.map(l=>`<option value='${l}'>${l.toUpperCase()}</option>`).join("");
    el(id).value = state.lang;
  });
  await loadLanguage(state.lang);
  bind();
  el("llmModeSelect").value = state.project.llmSettings.mode || "starter";
  el("llmTempInput").value = String(state.project.llmSettings.temp ?? 0.7);
  el("llmTokensInput").value = String(state.project.llmSettings.maxTokens ?? 96);
  el("llmStyleSelect").value = state.project.llmSettings.style || "balanced";
  toggleLlmModeUI();
  el("ideaInput").value = state.project.idea || "";
  el("planView").textContent = state.project.plan || "";
  renderAll();
  if (!localStorage.getItem("hcs.lang.picked")) el("languageDialog").showModal();
  setInterval(()=>save("interval"), 20000);
}

init();