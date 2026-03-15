# hcs-llm-autotrainer-web v2 beta

Made with ❤️ by hcsmedia

## Neu in v2
- Anfänger-Workflow (6 Schritte) für einfache Eingaben
- Mehr Quellen: Paste, Files, Wikipedia, WebSearch (DuckDuckGo API)
- Dataset-Builder mit Clean/Dedup/Chunk/Score/Split + Warnungen
- AI-unterstützte Datengenerierung (synthetic data)
- Tokenizer train/import/export + encode/decode preview
- Training mit besseren Visuals:
  - Loss-Chart
  - Throughput-Chart
  - Statuschips (good/stagnating/overfit)
  - Checkpoint-Management
- Trainiertes Modell testen (Generate)
- Run in Baseline „mergen“ (Experiment-Update)
- Compare-View mit Best-Run-Highlight
- EN/DE/FR, Autosave, Import/Export Bundles, mobile-first

## Hinweis zur "AI assisted mit WebGPU LLM"
In dieser Version ist der Assist-Teil device-aware und lokal implementiert.
Die echten WebGPU-Kernel-Dateien (`webgpu/*.wgsl`) bleiben vorbereitet für den nächsten Schritt (echte Compute-Training-Kerne).

## GGUF + WebGPU Research (Stand jetzt)
Kurzfassung:
- **GGUF im Browser**: stabil über **wllama (WASM/CPU)**
- **WebGPU im Browser**: sehr gut für WebLLM/ONNX-Workloads, aber **nicht der Standardpfad für GGUF**
- **"GGUF + WebGPU für alles"** ist aktuell noch nicht überall robust production-ready

Was wir gebaut haben:
1. **Beginner LLM Playground** im Tab **AI Assist**
2. **Gemma GGUF quantized** als Standard-Workflow für Anfänger (aktuell 4B Q4_0 als stabile Nähe zu 3B)
3. Fallback: **StarterLM (eingebaut)**, sofort nutzbar ohne Download
4. Optional: andere GGUF-Modelle direkt aus Hugging Face laden (`repo` + `file`)
5. Volle Basis-Customization (Temperature, max tokens, style) bei einfacher UX + 1-Klick Workflows
6. Device-aware Defaults (Threads, Mobile)

Empfohlener Praxis-Plan:
1. GGUF quantisiert halten (Q4/Q5)
2. große GGUF-Dateien splitten/chunken
3. WebGPU für Training/Visuals/Compute-Pipeline nutzen
4. GGUF-Inferenz lokal über wllama für maximale Browser-Kompatibilität

## Lokal starten
```bash
cd hcs-llm-autotrainer-web-v1-beta
python3 -m http.server 8080
```
Dann öffnen: `http://127.0.0.1:8080`

## Deploy
GitHub Pages (branch `main`, root `/`).
