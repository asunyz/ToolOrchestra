# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

ToolOrchestra trains small "orchestrator" models (Orchestrator-8B) that coordinate a heterogeneous tool set — basic tools (web search, code interpreter), specialized LLMs (coding, math), and generalist LLMs (GPT-5, Nemotron, Claude Opus). The orchestrator is trained end-to-end with RL using outcome, efficiency, and preference rewards. Paper: arXiv:2511.21689.

## Environments

Multiple conda envs are required — evaluation and training are not one-env workflows:

- `toolorchestra` (training) — python 3.12, `pip install -r requirements.txt`, `flash-attn`, `flashinfer-python`, then `pip install -e training/rollout`.
- `retriever` — for the wiki/HLE retrieval server (`faiss-gpu`, `pyserini`, `tavily-python`).
- `vllm1` — for hosting local models via vLLM 0.9.2; also `pip install -e evaluation/tau2-bench`.
- τ²-Bench requires a locally-installed `tau2` package (uninstall the pip one, then `pip install -e evaluation/tau2-bench`).

Required env vars: `HF_HOME`, `REPO_PATH`, `INDEX_DIR`, `CKPT_DIR`, `TAVILY_KEY`, `WANDB_API_KEY`, `OSS_KEY` (NVIDIA NGC), `CLIENT_ID`, `CLIENT_SECRET`.

## Common commands

Training (orchestrator RL — verl-based):
```bash
cd training && python resume_h100.py       # top-level launcher; drives multiple experiments
bash training/train_orchestrator.sh        # underlying training script
```

Evaluation (each in its own env; see README §Setup):
```bash
cd evaluation
python run_hle.py            # HLE — needs vllm1 + retriever
python run_frames.py         # FRAMES — needs vllm1 + retriever
cd tau2-bench && python run.py   # τ²-Bench — needs vllm1
```

For HLE, if host-model connections drop, comment `run_hle.py:248` and run `run_hle.py` and `eval_hle.py --model_name … --model_config model_configs/serve2.json --example_path hle.jsonl` as separate processes.

## Architecture

The system has three layers that are read together for any nontrivial change:

1. **Orchestrator loop** (`evaluation/eval_hle.py`, `eval_frames.py`, `tau2-bench/`) — Alternates reasoning turns with tool calls. Tool schemas are declared in `evaluation/tools.json` (and `training/tools.json` for training-time rollouts). `call_tool` dispatch and prompt strings live inline in `eval_hle.py` / `eval_frames.py`.

2. **LLM backend abstraction** (`LLM_CALL.py`) — Single `get_llm_response` entry point that talks to vLLM, OpenAI, Azure, Anthropic (with OpenAI↔Claude tool-schema conversion), and NGC-hosted models. This is the extension point for adding new providers.

3. **Training stack** (`training/`) — Built on a customized fork of **verl** (`training/verl/`), with orchestrator-specific pieces in `training/recipe/` (algo, DAPO), `training/rollout/` (installable package driving multi-turn tool-calling rollouts, including `tau2` env), and `training/lead_agent/`. `resume_h100.py` orchestrates slurm-style experiment launches by name — experiment names must match filenames in the training directory.

Retrieval for HLE/FRAMES is served separately (`evaluation/retrieval_hle.py`, `retrieval_wiki.py`) and hit as an HTTP service by the orchestrator loop; it uses the `INDEX_DIR` faiss indices.

Data synthesis pipeline (used to produce the ToolScale dataset) is in `data_synthesis/` — a Jupyter notebook (`run.ipynb`) plus prompt templates under `data_synthesis/prompts/`.

## Extension points (from README §Customization)

- New LLM provider → extend `get_llm_response` in `LLM_CALL.py`.
- Change eval prompts → `eval_hle.py:455-458`, `eval_frames.py:506-509`.
- Different tool set → swap `tool_config` at `eval_frames.py:27` / `eval_hle.py:27`; edit `tools.json` and the `call_tool` dispatch.
- Parallel experiments → the `{EXPERIMENT_NAME1..3}` placeholders in `training/resume_h100.py` must match training-directory filenames.
