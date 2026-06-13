#!/usr/bin/env bash
set -euo pipefail
cd /home/ajifang/projects/pony-rec-rescue-shadow-v6
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$PWD:$PWD/scripts/adapters:$PWD/scripts/audit:$PWD/scripts/build:$PWD/scripts/train"
PY=/home/ajifang/miniconda3/bin/python
OUT=/home/ajifang/projects/pony-rec-rescue-shadow-v6/outputs/tools_large10000_100neg_llm2rec_sasrec_NATIVE_llm2vec_reproduction_same_candidate
NPY=/home/ajifang/projects/LLM2Rec/item_info/ToolsSameCandidate100Neg/llm2rec_native_reproduction_title_item_embs.npy
PTH=/home/ajifang/projects/LLM2Rec/outputs/tools_large10000_100neg_llm2rec_sasrec_NATIVE_llm2vec_reproduction_same_candidate/seqrec_ckpt/LLM2Rec_official_qwen3base_tools_100neg-evaluate_with_seqrec.py_--model=SASRec_--dataset=T-Jun-13-2026_23-43-09-cba11d.pth
TASK=/home/ajifang/projects/pony-rec-rescue-shadow-v6/outputs/baselines/external_tasks/tools_large10000_100neg_test_same_candidate
VALID=/home/ajifang/projects/pony-rec-rescue-shadow-v6/outputs/baselines/external_tasks/tools_large10000_100neg_valid_same_candidate
POLICY=native_llm2vec_qwen2_0.5b_reproduction_same_candidate_v1
VARIANT=native_llm2vec_encoder_default_hparams
echo "[score] reuse ckpt=$PTH"
"$PY" scripts/adapters/main_run_llm2rec_official_same_candidate_adapter.py --stage run --domain tools --task_dir "$TASK" --valid_task_dir "$VALID" --output_scores_path "$OUT/scores.csv" --provenance_output_path "$OUT/fairness_provenance.json" --fairness_policy_id "$POLICY" --comparison_variant "$VARIANT" --backbone_path /home/ajifang/models/Qwen/Qwen3-8B --llm_adaptation_mode native_item_embedding --embedding_backend hf_mean_pool --embedding_max_length 128 --hf_device_map auto --llm2rec_item_embedding_path "$NPY" --llm2rec_adapter_dir "$PWD/outputs/baselines/paper_adapters/tools_large10000_100neg_llm2rec_NATIVE_adapter" --llm2rec_ckpt_dir "$OUT/seqrec_ckpt" --adapter_or_checkpoint_path "$PTH"
echo "[score] adapter rc=$?"
if [ -f "$OUT/scores.csv" ]; then
  "$PY" scripts/misc/main_import_same_candidate_baseline_scores.py --baseline_name llm2rec_sasrec --exp_name tools_large10000_100neg_llm2rec_sasrec_NATIVE_llm2vec_reproduction_same_candidate --domain tools --ranking_input_path "$TASK/ranking_test.jsonl" --scores_path "$OUT/scores.csv" --output_root outputs --ks 5,10,20 --k 10 --artifact_class completed_result --status_label native_encoder_bridge_fidelity --min_score_coverage 1.0 --fairness_policy_id "$POLICY" --comparison_variant "$VARIANT" --provenance_path "$OUT/fairness_provenance.json"
  echo NATIVE_RUN_COMPLETE
else
  echo NO_SCORES; exit 1
fi
