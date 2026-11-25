#!/bin/sh
# Run the tie check for the given tags on this pod: ./run_tie_some.sh [--setup] [--after-pid PID] tag...
export HF_HOME=/workspace/hf PIP_BREAK_SYSTEM_PACKAGES=1 PIP_ROOT_USER_ACTION=ignore LD_LIBRARY_PATH=/usr/local/cuda/lib64 PYTHONUNBUFFERED=1 HF_XET_HIGH_PERFORMANCE=1 PATH=/usr/local/cuda/bin:$PATH
mkdir -p /workspace/logs /workspace/out/tie; cd /workspace/probe
model_of() { case $1 in
  qwen3) echo "Qwen/Qwen3-32B";; qwen36) echo "Qwen/Qwen3.6-27B";; gensyn) echo "Gensyn/open-1b-base";; marin8b) echo "marin-community/marin-8b-base";;
  pythia12b) echo "EleutherAI/pythia-12b";; pythia6_9b) echo "EleutherAI/pythia-6.9b";; pythia2_8b) echo "EleutherAI/pythia-2.8b";;
  pythia1_4b) echo "EleutherAI/pythia-1.4b";; pythia410m) echo "EleutherAI/pythia-410m";;
  phi4) echo "microsoft/phi-4";; muse30b) echo "meta-models/Muse-Glimmer-30B";; gemma3_27b) echo "unsloth/gemma-3-27b-it";;
  granite42_30b) echo "ibm-granite/granite-4.2-30b";; llama33_70b) echo "unsloth/Llama-3.3-70B-Instruct";; qwen3chk) echo "Qwen/Qwen3-32B";;
  gemma4) echo "google/gemma-4-31B-it";; mistral32) echo "mistralai/Mistral-Small-3.2-24B-Instruct-2506";;
  marin32b) echo "marin-community/marin-32b-base";; olmo7b) echo "allenai/OLMo-2-1124-7B";; olmo13b) echo "allenai/OLMo-2-1124-13B";; olmo32b) echo "allenai/OLMo-2-0325-32B";; esac; }
if [ "$1" = "--setup" ]; then shift
  pip install -q -U "transformers>=5.5" accelerate "huggingface_hub[hf_xet]" safetensors sentencepiece protobuf mistral-common numpy scipy > ../logs/pip.log 2>&1; echo PIP_DONE >> ../logs/pip.log
fi
if [ "$1" = "--after-pid" ]; then while kill -0 $2 2>/dev/null; do sleep 3; done; shift; shift; fi
for tag in "$@"; do
  dlx=""; [ "$tag" = "mistral32" ] && dlx="--exclude consolidated.safetensors"   # the repo holds the weights twice
  if ! grep -q DL_DONE ../logs/dl_$tag.log 2>/dev/null; then (hf download $(model_of $tag) $dlx > ../logs/dl_$tag.log 2>&1; echo DL_DONE >> ../logs/dl_$tag.log) & fi
done
for tag in "$@"; do
  while ! grep -q DL_DONE ../logs/dl_$tag.log 2>/dev/null; do sleep 3; done
  extra=""; [ "$tag" = "gensyn" ] && extra="--trust"
  case $tag in phi4|muse30b|gemma3_27b|granite42_30b|llama33_70b|qwen3chk|gemma4|mistral32) extra="--nodir";; esac
  python3 tiecheck.py --model $(model_of $tag) --hand ../out/corpus.jsonl --docs ../out/blind_corpus.jsonl --out ../out/tie/$tag $extra > ../logs/tie_$tag.log 2>&1; echo EXIT=$? >> ../logs/tie_$tag.log
done
echo POD_DONE > ../logs/pod.done
