# hcs-llm-autotrainer-web v1 beta

Train tiny language models directly in your browser.

**Branding:** Made with ❤️ by hcsmedia

## Was ist drin (v1-first functional)
- Mobile-first, dark glass UI with multi-panel desktop layout
- Language Picker + complete UI translation (**EN/DE/FR**)
- `program.md` control layer (single source of truth)
- Dataset Builder:
  - paste text, upload local text files
  - web source discovery (Wikipedia API)
  - clean/normalize/deduplicate/chunk/filter/train-val split
  - token estimation, quality score, warnings
- Tokenizer workflow:
  - train simple local tokenizer from dataset
  - encode/decode preview
  - export/import tokenizer
- Training workflow (browser-safe first version):
  - tiny device-aware presets (tiny/small/experimental)
  - simulated short-run training loop with live loss/val curves
  - task timeline, progress state (good/stagnating/overfit/problem)
  - checkpoint system: create/list/rename/mark/export/resume
- Run comparison:
  - side-by-side metrics and best-run highlight
- Autosave + restore:
  - project, runs, tasks, checkpoints, logs in localStorage
- Full import/export bundles:
  - project bundle, checkpoint bundle, dataset bundle
- Debug/health panel:
  - WebGPU availability, fallback status, autosave, resume readiness

## Static deployment (GitHub Pages)
This project is backend-free and GitHub Pages compatible.

1. Create GitHub repo: `hcs-llm-autotrainer-web-v1-beta`
2. Upload all files from this folder
3. Go to **Settings → Pages**
4. Source: **Deploy from a branch**
5. Branch: `main` and `/ (root)`
6. Save and open the generated Pages URL

## Local run
```bash
cd hcs-llm-autotrainer-web-v1-beta
python3 -m http.server 8080
# open http://localhost:8080
```

## Important note
Current training core is a practical browser-first loop to visualize and manage experiments reliably across devices.
WebGPU shader files are scaffolded in `webgpu/` for progressive replacement with real compute kernels.

## License
MIT
