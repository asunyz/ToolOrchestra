# Download models
export HF_HOME=/workspace/hf_cache
export HF_TOKEN=
hf download nvidia/Nemotron-Orchestrator-8B --local-dir /workspace/Orchestrator-8B
hf download multi-train/index --repo-type dataset     --include "wiki.*" --local-dir /workspace/index

# (optional) venv setup (you can use conda)
python -m venv /opt/vllm1     && /opt/vllm1/bin/pip install -U pip "vllm==0.9.2" "transformers==4.53.3"
python -m venv /opt/retriever && /opt/retriever/bin/pip install -U pip faiss-cpu transformers fastapi uvicorn pyserini tavily-python
/opt/retriever/bin/pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128

# If ports need editing
evaluation/model_configs/serve_frames.json 

# Launch + run

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
 
