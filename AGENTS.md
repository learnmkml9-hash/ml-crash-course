# Codex Instructions

This repository is an ML self-study and research-coding project. Treat it as both a learning record and a place to develop reusable experiment code.

## Working Style

- Prefer clear, educational PyTorch and Hugging Face code.
- Keep existing lesson files readable and approachable.
- For new graduate-level lessons, prefer reusable research-code structure over single-use scripts when it helps clarity: separate configuration, data loading, model definitions, training, evaluation, and utilities.
- Do not modify existing lesson scripts unless explicitly asked.
- Before large refactors, preserve behavior and add a simple smoke test.
- After edits, summarize changed files and how to test them.

## Experiments

- Use `argparse` or YAML configs for experiment settings.
- Add dry-run modes for expensive training scripts.
- Keep training defaults reasonable for local development unless a cluster run is explicitly targeted.

## Artifacts And Data

- Do not commit or track datasets, model checkpoints, Hugging Face caches, virtual environments, Slurm output logs, or large artifacts.
- Use scratch paths for VISION cluster outputs and Hugging Face caches.
- Keep generated artifacts under ignored directories such as `data/`, `checkpoints/`, `hf_outputs/`, `saved_models/`, `runs/`, or cluster scratch paths.

## Cluster Jobs

- For cluster jobs, use Slurm scripts in `scripts/`.
- Prefer explicit environment setup in cluster scripts, including scratch-backed cache paths for Hugging Face and package caches.
- Keep Slurm logs out of version control.
