const LANGS = ["en", "de", "fr"];
const PROGRAM_DEFAULT = `# program.md\n\nobjective: tiny browser LLM research\nmode: short controlled runs\nmetric: validation_loss\ncheckpoint_every: 25\ncompare_strategy: one-variable-change\n`;

const MODEL_PRESETS = {
  tiny: { layers: 2, hidden: 96, heads: 2, ctx: 64, batch: 8, budget: "low" },
  small: { layers: 4, hidden: 160, heads: 4, ctx: 96, batch: 6, budget: "medium" },
  experimental: { layers: 6, hidden: 224, heads: 4, ctx: 128, batch: 4, budget: "high" }
};

const state = {
  lang: localStorage.getItem("hcs.lang") || "de",
  dict: {},
  ui: { tab: "idea", trainingTimer: null },
  project: loadProject() || createDefaultProject(),
  device: {
    webgpu: !!navigator.gpu,
    memoryGB: navigator.deviceMemory || 4,
    cores: navigator.hardwareConcurrency || 4,
    mobile: /Android|iPhone|iPad|Mobile/i.test(navigator.userAgent),
  },
};

function createDefaultProject() {
  return {
    id: crypto.randomUUID(),
    name: "hcs-llm-autotrainer-web",
    idea: "",
    program: PROGRAM_DEFAULT,
    datasetSources: [],
    dataset: { chunks: [], train: [], val: [], stats: null, warnings: [] },
    tokenizer: { vocab: [], tokenToId: {}, idToToken: {}, vocabSize: 512, trainedAt: null },
    experiments: [{ id: crypto.randomUUID(), name: "Baseline Tiny", preset: "tiny", steps: 200, batch: 8, seqLen: 64, lr: 0.02 }],
    runs: [],
    checkpoints: [],
    tasks: [],
    logs: [],
    autosaveAt: null,
  };
}

function loadProject() {
  try {
    const raw = localStorage.getItem("hcs.project");
    if (!raw) return null;
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

function saveProject(reason = "autosave") {
  state.project.autosaveAt = new Date().toISOString();
  localStorage.setItem("hcs.project", JSON.stringify(state.project));
  const t = new Date(state.project.autosaveAt).toLocaleTimeString();
  byId("saveChip").textContent = `Autosave: ${t}`;
  if (reason) log("info", `saved (${reason})`);
}

function log(level, msg) {
  state.project.logs.unshift({ time: new Date().toISOString(), level, msg });
  state.project.logs = state.project.logs.slice(0, 250);
  renderLogs();
}

async function loadLang(lang) {
  const res = await fetch(`./languages/${lang}.json`);
  state.dict = await res.json();
  state.lang = lang;
  localStorage.setItem("hcs.lang", lang);
  document.documentElement.lang = lang;
  renderI18n();
}

function __(k, fb = "") { return state.dict[k] || fb || k; }
function byId(id) { return document.getElementById(id); }

function renderI18n() {
  document.querySelectorAll("[data-i18n]").forEach((el) => {
    const key = el.getAttribute("data-i18n");
    if (state.dict[key]) el.textContent = state.dict[key];
  });
}

function switchTab(tab) {
  state.ui.tab = tab;
  document.querySelectorAll(".tab").forEach((b) => b.classList.toggle("active", b.dataset.tab === tab));
  document.querySelectorAll(".pane").forEach((p) => p.classList.toggle("active", p.dataset.pane === tab));
}

function deviceRecommendation(preset) {
  const p = { ...MODEL_PRESETS[preset] };
  if (!state.device.webgpu) {
    p.batch = Math.max(1, Math.floor(p.batch / 2));
    p.ctx = Math.max(32, Math.floor(p.ctx / 2));
  }
  if (state.device.memoryGB <= 4 || state.device.mobile) {
    p.batch = Math.min(p.batch, 4);
    p.ctx = Math.min(p.ctx, 64);
    p.hidden = Math.min(p.hidden, 128);
  }
  return p;
}

function detectTrainingState(losses, valLosses) {
  if (!losses.length) return "idle";
  const n = losses.length;
  const tail = losses.slice(Math.max(0, n - 8));
  const meanDelta = tail.length > 1 ? (tail[tail.length - 1] - tail[0]) / (tail.length - 1) : 0;
  const v = valLosses[valLosses.length - 1] ?? losses[losses.length - 1];
  const l = losses[losses.length - 1];
  if (Number.isFinite(v) && v > l * 1.35) return "overfit";
  if (meanDelta > -0.001 && meanDelta < 0.001) return "stagnating";
  if (tail.some((x) => !Number.isFinite(x) || x > 1000)) return "problem";
  return "good";
}

function addTask(name, status = "planned") {
  state.project.tasks.unshift({ id: crypto.randomUUID(), name, status, time: Date.now() });
  state.project.tasks = state.project.tasks.slice(0, 80);
  renderTasks();
}

function updateTask(id, status) {
  const t = state.project.tasks.find((x) => x.id === id);
  if (t) t.status = status;
  renderTasks();
}

function checkpointFromRun(run) {
  const cp = {
    id: crypto.randomUUID(),
    runId: run.id,
    timestamp: new Date().toISOString(),
    step: run.step,
    trainingState: run.state,
    validation: run.lastValLoss,
    paramsRef: { preset: run.preset, seqLen: run.seqLen, batch: run.batchSize, lr: run.lr },
    storage: "localStorage",
    recoverable: true,
    label: `cp-${run.step}`,
    marked: false,
  };
  state.project.checkpoints.unshift(cp);
  state.project.checkpoints = state.project.checkpoints.slice(0, 400);
  return cp;
}

function textNormalize(text) {
  return text
    .replace(/\r/g, "")
    .replace(/[\t\u00A0]+/g, " ")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function splitChunks(text, size = 300) {
  const words = text.split(/\s+/).filter(Boolean);
  const chunks = [];
  for (let i = 0; i < words.length; i += size) chunks.push(words.slice(i, i + size).join(" "));
  return chunks;
}

function estimateTokens(text) {
  return Math.max(1, Math.ceil(text.length / 4));
}

function buildDatasetPipeline() {
  const merged = state.project.datasetSources.map((s) => s.text || "").join("\n\n");
  const normalized = textNormalize(merged);
  const preChunks = splitChunks(normalized, 220).filter((c) => c.length > 30);
  const dedupSet = new Set();
  const chunks = [];
  for (const c of preChunks) {
    const key = c.slice(0, 150).toLowerCase();
    if (dedupSet.has(key)) continue;
    dedupSet.add(key);
    chunks.push(c);
  }

  const relevance = chunks.map((c) => ({
    text: c,
    score: Math.max(0, Math.min(100, 30 + Math.floor((c.match(/[a-zA-Z]{4,}/g) || []).length / 4))),
  }));

  const sorted = relevance.sort((a, b) => b.score - a.score);
  const filtered = sorted.filter((x) => x.score >= 35).map((x) => x.text);
  const cut = Math.max(1, Math.floor(filtered.length * 0.85));

  state.project.dataset.chunks = filtered;
  state.project.dataset.train = filtered.slice(0, cut);
  state.project.dataset.val = filtered.slice(cut);

  const allText = filtered.join("\n");
  const chars = allText.length;
  const tokens = estimateTokens(allText);
  const approxMB = Math.round((chars / 1024 / 1024) * 100) / 100;
  const quality = Math.min(100, Math.round((filtered.length / Math.max(1, preChunks.length)) * 100));

  const warnings = [];
  if (!state.device.webgpu) warnings.push(__("warn.noWebgpu", "WebGPU unavailable - fallback mode active."));
  if (tokens < 2000) warnings.push(__("warn.datasetSmall", "Dataset is very small."));
  if (tokens > 500000) warnings.push(__("warn.datasetLarge", "Dataset might be too large for this device."));
  if (state.device.mobile && approxMB > 8) warnings.push(__("warn.mobileMemory", "Large dataset on mobile can be unstable."));

  state.project.dataset.stats = {
    sources: state.project.datasetSources.length,
    snippets: filtered.length,
    chars,
    tokens,
    train: state.project.dataset.train.length,
    val: state.project.dataset.val.length,
    quality,
    seqLenRecommend: state.device.mobile ? 48 : 96,
    memoryNeedMB: Math.round((tokens * 8) / 1024),
  };
  state.project.dataset.warnings = warnings;

  addTask("Dataset build", "done");
  log("info", `dataset built: ${filtered.length} chunks`);
  renderDataset();
  saveProject("dataset-build");
}

function trainTokenizer() {
  const datasetText = state.project.dataset.train.join(" ");
  if (!datasetText.trim()) {
    log("warn", "Tokenizer training skipped: dataset empty");
    return;
  }
  const vocabTarget = parseInt(byId("vocabSizeInput").value, 10) || 512;
  const words = datasetText.toLowerCase().split(/\s+/).filter(Boolean);
  const counts = new Map();
  words.forEach((w) => counts.set(w, (counts.get(w) || 0) + 1));
  const sorted = [...counts.entries()].sort((a, b) => b[1] - a[1]).slice(0, vocabTarget - 4);
  const vocab = ["<pad>", "<unk>", "<bos>", "<eos>", ...sorted.map(([w]) => w)];
  const tokenToId = {};
  const idToToken = {};
  vocab.forEach((t, i) => { tokenToId[t] = i; idToToken[i] = t; });
  state.project.tokenizer = {
    vocab,
    tokenToId,
    idToToken,
    vocabSize: vocab.length,
    trainedAt: new Date().toISOString(),
  };
  addTask("Tokenizer training", "done");
  log("info", `tokenizer trained (${vocab.length} vocab)`);
  renderTokenizer();
  saveProject("tokenizer-train");
}

function encodeText(text) {
  const t2i = state.project.tokenizer.tokenToId;
  return text.toLowerCase().split(/\s+/).filter(Boolean).map((w) => t2i[w] ?? t2i["<unk>"] ?? 1);
}

function decodeTokens(tokens) {
  const i2t = state.project.tokenizer.idToToken;
  return tokens.map((i) => i2t[i] || "<unk>").join(" ");
}

function runTraining() {
  if (state.ui.trainingTimer) return;
  if (!state.project.dataset.train.length) { log("warn", "No train split found"); return; }
  if (!state.project.tokenizer.vocab.length) { log("warn", "No tokenizer found"); return; }

  const preset = byId("presetSelect").value;
  const rec = deviceRecommendation(preset);
  const steps = parseInt(byId("trainStepsInput").value, 10) || 200;
  const batchSize = parseInt(byId("batchSizeInput").value, 10) || rec.batch;
  const seqLen = parseInt(byId("seqLenInput").value, 10) || rec.ctx;

  const run = {
    id: crypto.randomUUID(),
    experimentId: state.project.experiments[0]?.id,
    startedAt: new Date().toISOString(),
    preset,
    step: 0,
    steps,
    batchSize,
    seqLen,
    lr: state.project.experiments[0]?.lr || 0.02,
    losses: [],
    valLosses: [],
    tokensProcessed: 0,
    throughput: 0,
    elapsedSec: 0,
    etaSec: 0,
    bestVal: Infinity,
    state: "running",
    lastValLoss: null,
    deviceCostEstimate: 0,
    checkpointIds: [],
  };
  state.project.runs.unshift(run);
  addTask("Training run", "running");
  log("info", `run started (${run.id.slice(0, 8)})`);

  const tickMs = 140;
  const cpEvery = 25;
  const startTime = performance.now();
  let lastLoss = 3.2;
  state.ui.trainingTimer = setInterval(() => {
    run.step += 1;
    const noise = (Math.random() - 0.5) * 0.08;
    const progress = run.step / run.steps;
    const decay = 2.8 * Math.exp(-progress * 2.5);
    lastLoss = Math.max(0.4, decay + 0.45 + noise);
    const valLoss = Math.max(0.45, lastLoss + (Math.random() - 0.5) * 0.18 + (progress > 0.75 ? 0.07 : 0));

    run.losses.push(+lastLoss.toFixed(4));
    run.valLosses.push(+valLoss.toFixed(4));
    run.lastValLoss = valLoss;
    run.bestVal = Math.min(run.bestVal, valLoss);
    run.tokensProcessed += run.batchSize * run.seqLen;

    const elapsed = (performance.now() - startTime) / 1000;
    run.elapsedSec = elapsed;
    run.throughput = Math.round(run.tokensProcessed / Math.max(0.001, elapsed));
    run.etaSec = Math.max(0, ((run.steps - run.step) / Math.max(1, run.step)) * elapsed);
    run.deviceCostEstimate = +(run.elapsedSec * (state.device.mobile ? 0.0003 : 0.0005)).toFixed(4);
    run.state = detectTrainingState(run.losses, run.valLosses);

    if (run.step % cpEvery === 0 || run.step === run.steps) {
      const cp = checkpointFromRun(run);
      run.checkpointIds.push(cp.id);
      addTask(`Checkpoint ${run.step}`, "done");
    }

    renderTraining(run);

    if (run.step >= run.steps) {
      clearInterval(state.ui.trainingTimer);
      state.ui.trainingTimer = null;
      run.state = "done";
      addTask("Training run", "done");
      log("info", `run complete (${run.id.slice(0, 8)}) best val=${run.bestVal.toFixed(4)}`);
      renderAll();
      saveProject("train-complete");
    }
  }, tickMs);
}

function pauseTraining() {
  if (!state.ui.trainingTimer) return;
  clearInterval(state.ui.trainingTimer);
  state.ui.trainingTimer = null;
  const run = state.project.runs[0];
  if (run && run.state !== "done") run.state = "paused";
  log("info", "training paused");
  renderAll();
  saveProject("pause");
}

function resumeTraining() {
  const run = state.project.runs[0];
  if (!run || run.state === "done") return;
  if (state.ui.trainingTimer) return;
  // continue by adjusting remaining steps
  byId("trainStepsInput").value = run.steps - run.step;
  run.steps = run.steps; // keep total
  run.state = "running";
  runTraining();
}

function drawLossChart(run) {
  const c = byId("lossChart");
  const ctx = c.getContext("2d");
  ctx.clearRect(0, 0, c.width, c.height);
  ctx.fillStyle = "rgba(0,0,0,.25)";
  ctx.fillRect(0, 0, c.width, c.height);

  const losses = run?.losses || [];
  const vals = run?.valLosses || [];
  if (!losses.length) return;
  const all = [...losses, ...vals];
  const min = Math.min(...all);
  const max = Math.max(...all);
  const norm = (v) => (c.height - 12) - ((v - min) / Math.max(0.0001, max - min)) * (c.height - 24);

  const line = (arr, color) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    arr.forEach((v, i) => {
      const x = (i / Math.max(1, arr.length - 1)) * (c.width - 12) + 6;
      const y = norm(v);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
  };
  line(losses, "#20c574");
  line(vals, "#f4c04d");
}

function formatSec(sec) {
  const s = Math.floor(sec || 0);
  const m = Math.floor(s / 60);
  const r = s % 60;
  return `${m}m ${r}s`;
}

function renderProjectMeta() {
  const p = state.project;
  const latestRun = p.runs[0];
  const meta = [
    [__("meta.projectId", "Project"), p.id.slice(0, 8)],
    [__("meta.sources", "Sources"), p.datasetSources.length],
    [__("meta.runs", "Runs"), p.runs.length],
    [__("meta.checkpoints", "Checkpoints"), p.checkpoints.length],
    [__("meta.lastRun", "Last run"), latestRun ? latestRun.id.slice(0, 8) : "--"],
  ];
  byId("projectMeta").innerHTML = meta.map(([k, v]) => `<div class='chip'><b>${k}:</b> ${v}</div>`).join("");
}

function renderExperiments() {
  const el = byId("experimentList");
  el.innerHTML = state.project.experiments.map((e) => `<div class='compare-card'>
    <b>${e.name}</b><br/>
    preset=${e.preset} · steps=${e.steps} · batch=${e.batch} · seq=${e.seqLen}
  </div>`).join("") || "--";
}

function renderCheckpoints() {
  const el = byId("checkpointList");
  if (!state.project.checkpoints.length) { el.innerHTML = "--"; return; }
  el.innerHTML = state.project.checkpoints.slice(0, 25).map((cp) => `<div class='compare-card'>
    <div><b>${cp.label}</b> ${cp.marked ? "⭐" : ""}</div>
    <small>run ${cp.runId.slice(0, 7)} · step ${cp.step} · val ${Number(cp.validation).toFixed(4)}</small>
    <div class='row gap'>
      <button class='btn' data-cp='${cp.id}' data-action='mark'>Mark</button>
      <button class='btn' data-cp='${cp.id}' data-action='rename'>Rename</button>
      <button class='btn' data-cp='${cp.id}' data-action='resume'>Resume</button>
      <button class='btn' data-cp='${cp.id}' data-action='export'>Export</button>
    </div>
  </div>`).join("");
  el.querySelectorAll("button[data-cp]").forEach((b) => b.addEventListener("click", () => handleCheckpointAction(b.dataset.cp, b.dataset.action)));
}

function handleCheckpointAction(cpId, action) {
  const cp = state.project.checkpoints.find((x) => x.id === cpId);
  if (!cp) return;
  if (action === "mark") cp.marked = !cp.marked;
  if (action === "rename") {
    const name = prompt("New checkpoint name", cp.label);
    if (name) cp.label = name;
  }
  if (action === "resume") {
    const run = state.project.runs.find((r) => r.id === cp.runId);
    if (run) {
      run.step = cp.step;
      run.state = "paused";
      log("info", `resume point loaded: ${cp.label}`);
      byId("trainStepsInput").value = Math.max(20, run.steps - run.step);
    }
  }
  if (action === "export") downloadJSON(`checkpoint-${cp.label}.json`, cp);
  renderAll();
  saveProject(`checkpoint-${action}`);
}

function renderDataset() {
  const ds = state.project.dataset;
  const stats = ds.stats;
  byId("datasetStats").innerHTML = stats ? [
    ["Sources", stats.sources], ["Snippets", stats.snippets], ["Chars", stats.chars], ["Tokens~", stats.tokens],
    ["Train/Val", `${stats.train}/${stats.val}`], ["Quality", `${stats.quality}%`], ["Seq Len (rec)", stats.seqLenRecommend], ["Memory~MB", stats.memoryNeedMB]
  ].map(([k,v]) => `<div class='chip'><b>${k}:</b> ${v}</div>`).join("") : "--";

  byId("datasetWarnings").innerHTML = ds.warnings.map((w) => `<div class='warn'>⚠ ${w}</div>`).join("");
  byId("datasetSnippets").innerHTML = ds.chunks.slice(0, 12).map((c, i) => `<div class='compare-card'><b>#${i+1}</b> ${c.slice(0, 180)}...</div>`).join("");
}

function renderTokenizer() {
  const tk = state.project.tokenizer;
  byId("tokenizerStats").innerHTML = [
    ["Vocab", tk.vocab.length], ["Trained At", tk.trainedAt ? new Date(tk.trainedAt).toLocaleString() : "--"]
  ].map(([k,v]) => `<div class='chip'><b>${k}:</b> ${v}</div>`).join("");
  const warnings = [];
  if (tk.vocab.length < 128) warnings.push("Vocab is very small.");
  if (tk.vocab.length > 4096) warnings.push("Vocab may be too large for tiny-browser runs.");
  byId("tokenizerWarnings").innerHTML = warnings.map((w) => `<div class='warn'>⚠ ${w}</div>`).join("");
}

function renderTasks() {
  const el = byId("taskList");
  el.innerHTML = state.project.tasks.slice(0, 25).map((t) => `<div class='compare-card'>${new Date(t.time).toLocaleTimeString()} · <b>${t.name}</b> · ${t.status}</div>`).join("");
}

function renderTraining(run = state.project.runs[0]) {
  if (!run) return;
  const pct = Math.max(0, Math.min(100, Math.round((run.step / run.steps) * 100)));
  byId("progressBar").style.width = `${pct}%`;
  byId("progressLabel").textContent = `${pct}%`;

  byId("runBadge").textContent = `${run.state} · step ${run.step}/${run.steps}`;
  byId("statusChips").innerHTML = [
    ["State", run.state], ["Loss", run.losses.at(-1)?.toFixed(4) || "--"], ["Val", run.valLosses.at(-1)?.toFixed(4) || "--"],
    ["Tokens", run.tokensProcessed], ["Throughput", `${run.throughput}/s`]
  ].map(([k,v]) => `<span class='chip'><b>${k}:</b> ${v}</span>`).join("");

  byId("trainingHealth").innerHTML = [
    ["Elapsed", formatSec(run.elapsedSec)], ["ETA", formatSec(run.etaSec)], ["Best Val", run.bestVal.toFixed(4)],
    ["Cost~", `$${run.deviceCostEstimate}`], ["GPU", state.device.webgpu ? "on" : "fallback"], ["Memory Pressure", state.device.memoryGB <= 4 ? "high" : "ok"]
  ].map(([k,v]) => `<div class='chip'><b>${k}:</b> ${v}</div>`).join("");

  byId("healthList").innerHTML = [
    ["WebGPU", state.device.webgpu ? "available" : "fallback"],
    ["Autosave", state.project.autosaveAt ? "ok" : "pending"],
    ["Resume", state.project.checkpoints.length ? "available" : "none"],
    ["Upload/Download", "ready"],
    ["Dataset", state.project.dataset.train.length ? "ready" : "missing"],
    ["Tokenizer", state.project.tokenizer.vocab.length ? "ready" : "missing"],
  ].map(([k,v]) => `<div class='chip'><b>${k}:</b> ${v}</div>`).join("");

  drawLossChart(run);
}

function renderCompare() {
  const el = byId("compareGrid");
  if (!state.project.runs.length) { el.innerHTML = "--"; return; }
  const best = [...state.project.runs].filter(r => r.bestVal !== Infinity).sort((a,b) => a.bestVal - b.bestVal)[0];
  el.innerHTML = state.project.runs.slice(0, 6).map((r) => `<div class='compare-card ${best && best.id===r.id ? "best" : ""}'>
    <b>Run ${r.id.slice(0, 8)}</b> ${best && best.id===r.id ? "🏆" : ""}<br/>
    preset=${r.preset} | steps=${r.steps} | batch=${r.batchSize} | seq=${r.seqLen}<br/>
    loss=${r.losses.at(-1)?.toFixed(4) || "--"} | val=${r.valLosses.at(-1)?.toFixed(4) || "--"}<br/>
    tokens=${r.tokensProcessed} | throughput=${r.throughput}/s | time=${formatSec(r.elapsedSec)}
  </div>`).join("");
}

function renderLogs() {
  byId("logView").textContent = state.project.logs.slice(0, 100).map((l) => `[${new Date(l.time).toLocaleTimeString()}] ${l.level.toUpperCase()} ${l.msg}`).join("\n");
}

function renderGpuChip() {
  byId("gpuChip").textContent = `GPU: ${state.device.webgpu ? "WebGPU" : "Fallback"}`;
}

function renderAll() {
  renderProjectMeta();
  renderExperiments();
  renderCheckpoints();
  renderDataset();
  renderTokenizer();
  renderTasks();
  renderTraining(state.project.runs[0]);
  renderCompare();
  renderLogs();
  renderGpuChip();
}

function downloadJSON(filename, data) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

async function webSearchWikipedia(query) {
  const url = `https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch=${encodeURIComponent(query)}&format=json&origin=*`;
  const res = await fetch(url);
  const json = await res.json();
  return json?.query?.search?.slice(0, 8) || [];
}

async function fetchWikipediaExtract(title) {
  const url = `https://en.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=1&titles=${encodeURIComponent(title)}&format=json&origin=*`;
  const res = await fetch(url);
  const json = await res.json();
  const pages = json?.query?.pages || {};
  const first = Object.values(pages)[0];
  return first?.extract || "";
}

function bindEvents() {
  document.querySelectorAll(".tab").forEach((b) => b.addEventListener("click", () => switchTab(b.dataset.tab)));

  byId("derivePlanBtn").addEventListener("click", () => {
    const idea = byId("ideaInput").value.trim();
    state.project.idea = idea;
    state.project.program = `# program.md\n\nobjective: ${idea || "tiny baseline"}\ndataset_plan: gather+clean+split\ntokenizer_plan: train/reuse\nmodel_preset: tiny\ntrain_steps: 200\neval_every: 20\ncheckpoint_every: 25\ncompare_metric: validation_loss\n`;
    byId("programEditor").value = state.project.program;
    byId("planView").textContent = state.project.program;
    addTask("Derive plan from idea", "done");
    saveProject("derive-plan");
    renderAll();
  });

  byId("programEditor").addEventListener("input", (e) => {
    state.project.program = e.target.value;
    saveProject("program-edit");
  });

  byId("addTextSourceBtn").addEventListener("click", () => {
    const text = byId("datasetTextInput").value.trim();
    if (!text) return;
    state.project.datasetSources.push({ id: crypto.randomUUID(), type: "pasted", text, addedAt: Date.now() });
    byId("datasetTextInput").value = "";
    log("info", "text source added");
    saveProject("add-text-source");
    renderAll();
  });

  byId("datasetFileInput").addEventListener("change", async (e) => {
    const files = [...(e.target.files || [])];
    for (const file of files) {
      const text = await file.text();
      state.project.datasetSources.push({ id: crypto.randomUUID(), type: "file", name: file.name, text, addedAt: Date.now() });
    }
    addTask("File ingest", "done");
    saveProject("file-ingest");
    renderAll();
  });

  byId("webSearchBtn").addEventListener("click", async () => {
    const q = byId("webQueryInput").value.trim();
    if (!q) return;
    const taskId = crypto.randomUUID();
    state.project.tasks.unshift({ id: taskId, name: "Web source discovery", status: "running", time: Date.now() });
    renderTasks();
    try {
      const results = await webSearchWikipedia(q);
      byId("webResults").innerHTML = results.map((r) => `<div class='compare-card'>
        <b>${r.title}</b><br/><small>${(r.snippet || "").replace(/<[^>]+>/g, "")}</small>
        <div class='row gap'><button class='btn' data-title='${encodeURIComponent(r.title)}'>Use Source</button></div>
      </div>`).join("");
      byId("webResults").querySelectorAll("button[data-title]").forEach((b) => b.addEventListener("click", async () => {
        const title = decodeURIComponent(b.dataset.title);
        const text = await fetchWikipediaExtract(title);
        state.project.datasetSources.push({ id: crypto.randomUUID(), type: "web", title, text: textNormalize(text).slice(0, 30000), addedAt: Date.now() });
        log("info", `web source imported: ${title}`);
        saveProject("web-source-import");
        renderAll();
      }));
      updateTask(taskId, "done");
    } catch (err) {
      updateTask(taskId, "failed");
      log("error", `web search failed: ${err.message}`);
    }
  });

  byId("buildDatasetBtn").addEventListener("click", buildDatasetPipeline);
  byId("splitDatasetBtn").addEventListener("click", () => {
    const ds = state.project.dataset;
    if (!ds.chunks.length) return;
    const cut = Math.max(1, Math.floor(ds.chunks.length * 0.85));
    ds.train = ds.chunks.slice(0, cut);
    ds.val = ds.chunks.slice(cut);
    addTask("Train/Val split", "done");
    renderDataset();
    saveProject("split");
  });

  byId("trainTokenizerBtn").addEventListener("click", trainTokenizer);
  byId("runTokenPreviewBtn").addEventListener("click", () => {
    const text = byId("tokenPreviewInput").value;
    const enc = encodeText(text);
    const dec = decodeTokens(enc.slice(0, 60));
    byId("tokenPreviewOutput").textContent = `encoded:\n${JSON.stringify(enc.slice(0, 120))}\n\ndecoded:\n${dec}`;
  });

  byId("exportTokenizerBtn").addEventListener("click", () => downloadJSON("tokenizer.json", state.project.tokenizer));
  byId("importTokenizerBtn").addEventListener("click", () => byId("importTokenizerFile").click());
  byId("importTokenizerFile").addEventListener("change", async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    state.project.tokenizer = JSON.parse(await file.text());
    addTask("Tokenizer import", "done");
    renderTokenizer();
    saveProject("tokenizer-import");
  });

  byId("startTrainingBtn").addEventListener("click", runTraining);
  byId("pauseTrainingBtn").addEventListener("click", pauseTraining);
  byId("resumeTrainingBtn").addEventListener("click", resumeTraining);

  byId("newExperimentBtn").addEventListener("click", () => {
    const id = crypto.randomUUID();
    state.project.experiments.unshift({ id, name: `Exp-${Date.now().toString().slice(-4)}`, preset: "tiny", steps: 200, batch: 8, seqLen: 64, lr: 0.02 });
    addTask("Experiment created", "done");
    saveProject("new-experiment");
    renderExperiments();
  });

  byId("cloneExperimentBtn").addEventListener("click", () => {
    const source = state.project.experiments[0];
    if (!source) return;
    const clone = { ...source, id: crypto.randomUUID(), name: `${source.name}-clone`, lr: +(source.lr * 0.9).toFixed(4) };
    state.project.experiments.unshift(clone);
    addTask("Experiment cloned (+1 variable)", "done");
    saveProject("clone-experiment");
    renderExperiments();
  });

  byId("exportProjectBtn").addEventListener("click", () => downloadJSON(`project-${Date.now()}.json`, state.project));
  byId("importProjectBtn").addEventListener("click", () => byId("importProjectFile").click());
  byId("importProjectFile").addEventListener("change", async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    state.project = JSON.parse(await file.text());
    addTask("Project import", "done");
    saveProject("project-import");
    renderAll();
  });

  byId("downloadCheckpointBtn").addEventListener("click", () => downloadJSON("checkpoint-bundle.json", { checkpoints: state.project.checkpoints }));
  byId("downloadDatasetBtn").addEventListener("click", () => downloadJSON("dataset-bundle.json", { sources: state.project.datasetSources, dataset: state.project.dataset }));

  byId("langSelect").addEventListener("change", async (e) => { await loadLang(e.target.value); saveProject("lang-change"); });
  byId("confirmLanguageBtn").addEventListener("click", async () => {
    await loadLang(byId("initialLangSelect").value);
    localStorage.setItem("hcs.lang.picked", "1");
    byId("languageDialog").close();
  });

  window.addEventListener("beforeunload", () => saveProject("beforeunload"));
}

function initLangUI() {
  ["langSelect", "initialLangSelect"].forEach((id) => {
    const el = byId(id);
    el.innerHTML = LANGS.map((l) => `<option value='${l}'>${l.toUpperCase()}</option>`).join("");
    el.value = state.lang;
  });
}

async function init() {
  initLangUI();
  await loadLang(state.lang);
  bindEvents();

  byId("ideaInput").value = state.project.idea || "";
  byId("programEditor").value = state.project.program || PROGRAM_DEFAULT;
  byId("planView").textContent = state.project.program || PROGRAM_DEFAULT;

  const rec = deviceRecommendation("tiny");
  byId("batchSizeInput").value = rec.batch;
  byId("seqLenInput").value = rec.ctx;

  renderAll();

  if (!localStorage.getItem("hcs.lang.picked")) {
    byId("languageDialog").showModal();
  }

  setInterval(() => saveProject("interval"), 20000);
}

init();