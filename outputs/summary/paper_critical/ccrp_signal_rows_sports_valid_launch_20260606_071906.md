# C-CRP Signal Rows Sports Valid Launch

- Generated local time: `2026-06-06T01:23:19+02:00`
- Server: `pony-rec-gpu`
- Server project: `/home/ajifang/projects/pony-rec-rescue-shadow-v6`
- Domain/split: `sports` / `valid`
- Status: `active_at_launch_audit`
- PID: `3543564`
- Log: `ccrp_signal_rows_sports_valid_20260606_071906.log`
- PID file: `outputs/summary/paper_critical/ccrp_signal_rows_sports_valid_20260606_071906.pid`
- Output dir: `outputs/summary/paper_critical/ccrp_signal_generation_plan_post_performance_gate_20260606/ccrp_signal_rows_sports`
- Expected rows: `10000 * 101 = 1010000`
- Runner: `experiments/rsc/run_ccrp_v3_signal_rows.py`
- Runner sha256: `b3667dfbae798b29371dd0c4b0637ec2318d4b27dc0ea42a240171ec819f8ad7`
- Local runner commit: `70f2f0d`
- Documentation commit before launch: `096f716`
- Python env: `/home/ajifang/miniconda3/envs/qwen_vllm/bin/python`

## Preflight

- Active project processes before launch: `0`
- GPU before launch: `0 %, 15 MiB, 49140 MiB`
- Disk before launch: `/dev/vda5 209709965312 172843282432 26139488256 87% /`
- Target output existed before launch: `false`
- Sports valid inputs existed: `ranking_valid.jsonl` had `10000` lines; `candidate_items.csv` had `1010001` lines including header.

## Launch Audit

- Process uniqueness: `true`
- GPU after launch: `100 %, 42863 MiB, 49140 MiB`
- Disk after launch: `/dev/vda5 209709965312 172999270400 25983500288 87% /`
- Fatal log scan: clean for `Traceback`, `ERROR`, `FAILED`, `Killed`, `OOM`, `CUDA out`, `No space`, and `ImportError`
- Progress lines seen at this checkpoint: `false`

## Failed Attempt

An earlier launch attempt used `/home/ajifang/miniconda3/bin/python` and failed before writing rows:
`ImportError: libcudart.so.13: cannot open shared object file`. Use the `qwen_vllm` environment for this runner.

## Next Gate

Monitor PID `3543564` to completion. After the valid split finishes, audit
`valid_ccrp_signal_rows.csv` against the Sports valid `candidate_items.csv`
before any selector, test, observation, ablation, or hyperparameter use.
