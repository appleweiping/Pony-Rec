# Pre-Submission Gate Refresh

Generated: 2026-06-12T13:22:19.882101+00:00

- OK: `true`
- Final submission ready: `false`
- Final verdict: `LOCAL_PACKAGE_READY_BUT_EXTERNAL_OR_MANUAL_BLOCKED`
- External network mode: `live`
- Stamp: `20260612`
- Git HEAD before refresh: `77b03f28eba465a6029329970bcf90feb39a645b`
- Tracked dirty before refresh: `true`

## Steps

- `external_proceedings_metadata`: ok=`true`, ready=`false`, json=`outputs\summary\paper_critical\external_proceedings_metadata_recheck_20260612.json`
- `submission_package`: ok=`true`, ready=`true`, json=`outputs\summary\paper_critical\submission_package_audit_20260612.json`
- `submission_metadata_packet`: ok=`true`, ready=`true`, json=`outputs\summary\paper_critical\submission_metadata_packet_20260612.json`
- `manual_submission_checklist`: ok=`true`, ready=`true`, json=`outputs\summary\paper_critical\manual_submission_checklist_20260612.json`
- `final_submission_gate`: ok=`true`, ready=`false`, json=`outputs\summary\paper_critical\final_submission_gate_20260612.json`

## Input Fingerprints

- `Paper\main.tex`: `3d4bdc8a2fe0d7c400d72f0c81a39a2a3e6bf28e4d7e482541378c7b383edf60` (840 bytes)
- `Paper\references.bib`: `3633084d6ec06b5615e77b2485c0b0013f86bb72a1e36a314f97d5b032fdd953` (10448 bytes)
- `Paper\main.pdf`: `588afd8311c634569e029e8061c9a8cbefd1d6b2bd53a162b89857b5296cdc86` (546669 bytes)
- `Paper\main.log`: `38fedd0e191668ca890ca4bf0d5b048594adcfd195f36c68366144d377a6abf2` (55332 bytes)
- `Paper\main.blg`: `da6ab355903a0f836bf20461604568a97e46bd57723631cbd77d103771afeafb` (1606 bytes)
- `configs\paper_external_proceedings_metadata_checks.json`: `fac1d47fac2905188d507a546dad4c808071dda7741bddf01c5a5a8367b6483d` (2354 bytes)
- `outputs\summary\paper_critical\final_paper_claim_audit_after_full_panel_review_20260612.json`: `ceb6ff1eb1393580dab990e58b37afe8ff1dcb4324134ca2d04969c6c376a6db` (13282 bytes)
- `outputs\summary\paper_critical\final_full_manuscript_panel_review_20260612.json`: `6cbad714fe6488e649bc580c1e952054ebcc6a37075050ecf0e47c4152fbad84` (4596 bytes)
- `outputs\summary\paper_critical\final_pdf_polish_metadata_followup_20260612.json`: `dea9fc1b54dbe6ca37cc228c303e332a08009cb0d1968e5ae3a16a2d898c7e7d` (3056 bytes)
- `configs\paper_submission_profiles.json`: `00a083d7fca48c521814ce9effe6bf9294c7ddf7e071879edaa4739f39a67d2d` (1276 bytes)
- `configs\paper_submission_metadata.json`: `908484005b171a66d3e9424d36986d26286e86a0272a456b461a9e75b204e1c1` (1005 bytes)
- `configs\paper_manual_submission_checklist.json`: `e8bbef232cc418ca45976cb675411e2128075b6d8eae6d3ce23b561ecf666174` (4732 bytes)
- `scripts\audit\main_refresh_pre_submission_gates.py`: `8814ad9b1571d43e01d1425a23805b84f2487342d05a05bfd1a5be6590b14802` (17527 bytes)
- `scripts\audit\main_audit_external_proceedings_metadata.py`: `0143492e15967b3ee582329934b88f47211ce52454d950fb3af568b5b0e63b9f` (19677 bytes)
- `scripts\audit\main_audit_submission_package.py`: `dba948bb101588d3e4ceb56ae8a8436b4265ffde2de29fb9b363ea39403ff490` (27574 bytes)
- `scripts\audit\main_build_submission_metadata_packet.py`: `01b66efe8a816d5681cef351fdf87a594ad935193e1f9bf2336f5d8a559e513c` (10113 bytes)
- `scripts\audit\main_build_manual_submission_checklist.py`: `c6e50da192f1c8ee970e96e0241d2d614f739b2507b258dec88877d7d0d9dce3` (21881 bytes)
- `scripts\audit\main_build_final_submission_gate.py`: `892a3291518209df27c8b1f2954c34288ba01be539319915adb6dfd94accaeec` (11207 bytes)
- `scripts\audit\main_audit_pre_submission_refresh_freshness.py`: `6c0ebf3d3243b552005226cee4cc5d2065d3b2c3c55dfc95f7016545e0740f91` (13914 bytes)

## Remaining Blockers

- promax:final_page_range_missing_in_bib
- promax:crossref_registry_not_visible:status=404
- promax:doi_resolver_not_visible:status=404
- Final manual submission-system metadata/format checklist is not closed.
- ProMax final ACM page range and ACM/Crossref registry visibility must be rechecked immediately before submission.
- confirm_external_proceedings_metadata:external_proceedings_metadata_ready_not_closed
- manual_submission_system_items_not_confirmed
- external_proceedings_metadata_not_ready
- manual_submission_system_not_ready

## Failures

- None

## Warnings

- `external_proceedings_metadata:proex:crossref_not_visible:status=404`
- `external_proceedings_metadata:proex:doi_resolver_not_visible:status=404`
- `submission_package:underfull_layout_warnings:hbox=6,vbox=8`
- `final_submission_gate:submission_package:underfull_layout_warnings:hbox=6,vbox=8`
- `final_submission_gate:external_proceedings_metadata:proex:crossref_not_visible:status=404`
- `final_submission_gate:external_proceedings_metadata:proex:doi_resolver_not_visible:status=404`

## Next Actions

- Use final_submission_gate as the first-read final submission status.
- Rerun this refresh after any Paper, BibTeX, target profile, or submission metadata change.
- Keep private author/COI/reviewer/declaration fields outside the repository.
