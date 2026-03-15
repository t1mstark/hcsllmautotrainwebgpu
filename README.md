# hcs-llm-autotrainer-web (final)

Modern browser app to prototype tiny LLM workflows locally.

Built by **@timfromhcs** and **@hcsmediacorp**.

---

## What this is

A static, frontend-only web app for beginner-friendly LLM experimentation:

1. Idea →
2. Data collection/cleanup →
3. Tokenizer training →
4. Tiny training simulation/monitoring →
5. Model testing, comparison, export, backup

No backend is required for core functionality.

---

## Core features

- Beginner workflow (guided steps + help tab)
- Multi-language UI: **DE / EN / FR**
- Dataset tools: paste/files/web/wiki + chunk/dedup/split
- Tokenizer train/import/export + encode/decode preview
- Training monitor with charts:
  - Loss
  - Validation
  - Throughput
- Checkpoints + compare view + model gallery
- LLM playground:
  - **Gemma GGUF quantized** default workflow
  - StarterLM fallback (no model download)
- Export:
  - Project JSON
  - Dataset/checkpoint bundles
  - `.safetensors` metric exports
- Autosave + periodic snapshots

---

## Tech base

- Plain HTML/CSS/JavaScript (ES modules)
- Browser APIs (localStorage, fetch, file APIs)
- Optional GGUF runtime via `wllama`
- GitHub Actions + GitHub Pages deployment

---

## Try it live

GitHub Pages:

**https://t1mstark.github.io/hcsllmautotrainwebgpu/**

If cached, use a cache-buster:

`https://t1mstark.github.io/hcsllmautotrainwebgpu/?v=final`

---

## Run locally

```bash
git clone https://github.com/t1mstark/hcsllmautotrainwebgpu.git
cd hcsllmautotrainwebgpu
python3 -m http.server 8080
```

Open: `http://127.0.0.1:8080`

---

## How to modify

- UI layout: `index.html`
- Styling/theme/responsive: `style.css`
- Logic/features: `app.js`
- Translations: `languages/de.json`, `languages/en.json`, `languages/fr.json`
- WebGPU kernels (prepared): `webgpu/*.wgsl`

Recommended workflow:

```bash
git checkout -b feature/my-change
# edit files
git add .
git commit -m "My change"
git push origin feature/my-change
```

---

## Deploy flow (GitHub Pages)

- Push to `main`
- Workflow `.github/workflows/pages.yml` builds/deploys automatically
- Pages should be configured to **GitHub Actions** source

---

---

## Notes

- GGUF in browser works best via WASM runtime pathways today.
- WebGPU availability depends on browser/device.
- This project focuses on accessible experimentation, not production-scale training.
