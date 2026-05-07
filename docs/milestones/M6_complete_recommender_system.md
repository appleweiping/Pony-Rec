# M6: Complete Recommendation-System Roadmap

## Role

This milestone is the future complete-system layer. It should guide server work,
but it is not a completed paper claim until the listed modules are evaluated
under the shared protocol.

## Target System

```text
official-code-level external baselines
-> Qwen3-8B base model contract
-> method-specific LoRA/adapter artifacts
-> Shadow large-scale diagnostics
-> Signal LoRA
-> Decision LoRA
-> Generative LoRA
-> catalog-grounded generated-title verification
-> artifact-ready reproducibility package
```

## Completion Gates

- Official LLM-rec baselines have pinned repos, checkpoints, provenance, and
  exact same-candidate score coverage.
- Shadow v1/v6 run under the 101-candidate protocol with validation-selected
  gates.
- Signal/Decision/Generative LoRA teachers are created without test leakage.
- Generated-title recommendation is catalog-grounded and verified.
- Server runbooks produce outputs from a clean pull without local guesswork.
- A reviewer-gate pass confirms the claims do not exceed the completed evidence.

## Safe Claim Before Completion

```text
The repository contains a staged roadmap from controlled uncertainty-aware
candidate ranking to a fuller recommendation system.
```

## Safe Claim After Completion

```text
The framework supports an end-to-end recommendation decision system evaluated
under the same-candidate protocol, with official-code-level external baselines
and artifact-level provenance.
```

## Related Files

- `WEEK8_FUTURE_FRAMEWORK_ROADMAP_2026-05-06.md`
- `configs/week8_large_scale_future_framework.yaml`
- `configs/week8_future_lora/`
- `main_build_week8_same_candidate_pointwise_inputs.py`
- `main_run_week8_shadow_v6_gate_sweep.py`
- `main_build_week8_lora_framework_scaffold.py`
- `main_build_week8_generated_title_verification_scaffold.py`
- `scripts/run_week8_shadow_large_scale_diagnostic.sh`
- `scripts/run_week8_generated_title_verification_scaffold.sh`

