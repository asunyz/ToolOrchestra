C. Environment setup (for the README runbook)

# venvs on FAST local disk (not the /workspace mount)
python -m venv /opt/vllm1     && /opt/vllm1/bin/pip install -U pip "vllm==0.9.2" "transformers==4.53.3"
python -m venv /opt/retriever && /opt/retriever/bin/pip install -U pip faiss-cpu transformers fastapi uvicorn pyserini tavily-python
/opt/retriever/bin/pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128

# data on the volume
export HF_HOME=/workspace/hf_cache
hf download nvidia/Nemotron-Orchestrator-8B --local-dir /workspace/Orchestrator-8B
hf download multi-train/index --repo-type dataset --include "wiki.*" --local-dir /workspace/index

# config: cp .env.example .env; fill REPO_PATH/HF_HOME/INDEX_DIR/CKPT_DIR/OPENROUTER_API_KEY/TAVILY_KEY; then: source .env
Two version pins were forced by the pod's stack: transformers==4.53.3 (vLLM 0.9.2 breaks on newer — the aimv2 clash) and torch cu128 in the retriever (driver is CUDA 12.8; the default wheel was cu13).

D. Runtime config file (generated, not committed)

evaluation/model_configs/serve_frames.json — real ports for retrieval (1401) and the orchestrator ($CKPT_DIR → 1406); placeholder ports for the OpenRouter-routed models (ignored). Worth saving as a template in the repo.

E. Launch + run

# win1 (retriever venv):  
python -u retrieval_wiki.py --port 1401
# win2 (vllm1 venv):      
vllm serve "$CKPT_DIR" --enable-auto-tool-choice --tool-call-parser hermes --port 1406 --gpu-memory-utilization 0.6
# experiment run:
head -10 frames.jsonl > frames_smoke.jsonl
python eval_frames.py --model_name "$CKPT_DIR" --output_dir outputs/frames_smoke \
--model_config model_configs/serve_frames.json --example_file_path frames_smoke.jsonl
# accuracy:
python -c "import json,glob; r=[json.load(open(f)).get('correct') for f in glob.glob('outputs/frames_smoke/*.json')]; r=[x for x in r if x is not None]; c=sum(bool(x) for x in r); print(f'{c}/{len(r)} = {100*c/max(len(r),1):.1f}%')"

F. Result & environment

- Hardware: single A100 80GB PCIe, 1.5 TB RAM, 320 GB disk.
- Topology: CPU faiss (104 GB index in RAM) + retriever encoder on GPU; orchestrator local vLLM (util 0.6); all other tools via OpenRouter; math swapped to gpt-5-mini for the smoke test.
- Result: 8/10 (80%) on the FRAMES smoke slice → pipeline verified.
- Caveat to note in the README: this is a smoke-test config, not paper-faithful — for real FRAMES numbers, host Coder-32B/Qwen3-32B and the Qwen math models on local vLLM (--max-model-len 65536) so nothing is trimmed or substituted.
