# program.md — Single Source of Truth

## Project
- name: hcs-llm-autotrainer-web v1 beta
- tagline: Train tiny language models directly in your browser.
- branding: Made with ❤️ by hcsmedia

## Philosophy (from autoresearch spirit)
- short controlled runs
- visible metrics and checkpoints
- transparent pipeline (dataset → tokenizer → model)
- compare runs, change one variable, iterate

## Default experiment template
- objective: tiny LM baseline
- dataset_source: mixed (paste/files/web)
- tokenizer: train local small vocab
- model_preset: tiny
- train_steps: 200
- eval_every: 20
- checkpoint_every: 25
- compare_metric: validation_loss
- resume_allowed: true

## Notes
- Browser-only static app, no backend dependency.
- Device-aware constraints are mandatory.
