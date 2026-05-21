#!/bin/bash
set -e
cd ~/projects/pony-rec-rescue-shadow-v6
export PYTHONPATH=.

echo "=== C-CRP v3 All-Domain Run (full scale, matching baselines) ==="
echo "Start time: $(date)"

echo ""
echo "=== BOOKS (10000 users, matching baseline scale) ==="
python experiments/rsc/run_ccrp_v3_domain.py \
  --data outputs/baselines/external_tasks/books_large10000_100neg_test_same_candidate/ranking_test.jsonl \
  --output experiments/rsc/results/ccrp_v3_books

echo ""
echo "=== ELECTRONICS (10000 users) ==="
python experiments/rsc/run_ccrp_v3_domain.py \
  --data outputs/baselines/external_tasks/electronics_large10000_100neg_test_same_candidate/ranking_test.jsonl \
  --output experiments/rsc/results/ccrp_v3_electronics

echo ""
echo "=== MOVIES (10000 users) ==="
python experiments/rsc/run_ccrp_v3_domain.py \
  --data outputs/baselines/external_tasks/movies_large10000_100neg_test_same_candidate/ranking_test.jsonl \
  --output experiments/rsc/results/ccrp_v3_movies

echo ""
echo "=== ALL DONE ==="
echo "End time: $(date)"
