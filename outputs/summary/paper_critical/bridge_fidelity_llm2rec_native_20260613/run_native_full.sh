#!/usr/bin/env bash
# Full native run: train SASRec on native embeddings (ABSOLUTE ckpt_dir) + 101-cand score + import.
# Usage: DOMAIN=sports bash run_native_full.sh
set -euo pipefail
DOMAIN="${DOMAIN:?set DOMAIN}"
declare -A DSALIAS=( [tools]=ToolsSameCandidate100Neg [sports]=SportsSameCandidate100Neg [home]=HomeSameCandidate100Neg [toys]=ToysSameCandidate100Neg )
DS="${DSALIAS[$DOMAIN]}"
cd /home/ajifang/projects/pony-rec-rescue-shadow-v6
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$PWD:$PWD/scripts/adapters:$PWD/scripts/audit:$PWD/scripts/build:$PWD/scripts/train"
PY=/home/ajifang/miniconda3/bin/python
EXP="${DOMAIN}_large10000_100neg"
OUT="/home/ajifang/projects/pony-rec-rescue-shadow-v6/outputs/${EXP}_llm2rec_sasrec_NATIVE_llm2vec_reproduction_same_candidate"
NPY="/home/ajifang/projects/LLM2Rec/item_info/${DS}/llm2rec_native_reproduction_title_item_embs.npy"
TASK="/home/ajifang/projects/pony-rec-rescue-shadow-v6/outputs/baselines/external_tasks/${EXP}_test_same_candidate"
VALID="/home/ajifang/projects/pony-rec-rescue-shadow-v6/outputs/baselines/external_tasks/${EXP}_valid_same_candidate"
ADIR="$PWD/outputs/baselines/paper_adapters/${EXP}_llm2rec_NATIVE_adapter"
CKPT="$OUT/seqrec_ckpt"
POLICY=native_llm2vec_qwen2_0.5b_reproduction_same_candidate_v1
VARIANT=native_llm2vec_encoder_default_hparams
mkdir -p "$OUT"
echo "[native-full] DOMAIN=$DOMAIN DS=$DS NPY=$NPY ckpt(abs)=$CKPT"
"$PY" scripts/adapters/main_run_llm2rec_official_same_candidate_adapter.py --stage run --domain "$DOMAIN" --task_dir "$TASK" --valid_task_dir "$VALID" --output_scores_path "$OUT/scores.csv" --provenance_output_path "$OUT/fairness_provenance.json" --fairness_policy_id "$POLICY" --comparison_variant "$VARIANT" --backbone_path /home/ajifang/models/Qwen/Qwen3-8B --llm_adaptation_mode native_item_embedding --embedding_backend hf_mean_pool --embedding_max_length 128 --hf_device_map auto --llm2rec_item_embedding_path "$NPY" --llm2rec_adapter_dir "$ADIR" --llm2rec_ckpt_dir "$CKPT"
echo "[native-full] adapter rc=$?"
if [ -f "$OUT/scores.csv" ]; then
  "$PY" scripts/misc/main_import_same_candidate_baseline_scores.py --baseline_name llm2rec_sasrec --exp_name "${EXP}_llm2rec_sasrec_NATIVE_llm2vec_reproduction_same_candidate" --domain "$DOMAIN" --ranking_input_path "$TASK/ranking_test.jsonl" --scores_path "$OUT/scores.csv" --output_root outputs --ks 5,10,20 --k 10 --artifact_class completed_result --status_label native_encoder_bridge_fidelity --min_score_coverage 1.0 --fairness_policy_id "$POLICY" --comparison_variant "$VARIANT" --provenance_path "$OUT/fairness_provenance.json"
  echo NATIVE_RUN_COMPLETE
else
  echo NO_SCORES; exit 1
fi
