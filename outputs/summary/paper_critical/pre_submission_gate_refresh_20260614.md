# Pre-Submission Gate Refresh

Generated: 2026-06-14T21:21:56.022713+00:00

- OK: `false`
- Final submission ready: `false`
- Final verdict: `FINAL_SUBMISSION_GATE_NEEDS_REPAIR`
- External network mode: `live`
- Stamp: `20260614`
- Git HEAD before refresh: `15499e08e3ab729b40863f16b7f9add59f8a1cf6`
- Tracked dirty before refresh: `true`

## Steps

- `external_proceedings_metadata`: ok=`true`, ready=`false`, json=`outputs\summary\paper_critical\external_proceedings_metadata_recheck_20260614.json`
- `submission_package`: ok=`false`, ready=`false`, json=`outputs\summary\paper_critical\submission_package_audit_20260614.json`
- `submission_source_package`: ok=`false`, ready=`false`, json=`outputs\summary\paper_critical\submission_source_package_20260614.json`
- `submission_source_package_rebuild`: ok=`false`, ready=`false`, json=`outputs\summary\paper_critical\submission_source_package_rebuild_20260614.json`
- `submission_metadata_packet`: ok=`false`, ready=`false`, json=`outputs\summary\paper_critical\submission_metadata_packet_20260614.json`
- `manual_submission_checklist`: ok=`false`, ready=`false`, json=`outputs\summary\paper_critical\manual_submission_checklist_20260614.json`
- `final_submission_gate`: ok=`false`, ready=`false`, json=`outputs\summary\paper_critical\final_submission_gate_20260614.json`

## Input Fingerprints

- `Paper\main.tex`: `c5497f1ae66a0ac69731f7571a020bb1fb27de73afa3c3b6f56244ba5370c28e` (2050 bytes)
- `Paper\references.bib`: `aee26bca68e7421770d957ea2e65882e29994b21038588cdac9969a1a5b8a944` (16810 bytes)
- `Paper\main.pdf`: `be8944fde6bfeacd51f8f72e8bf5c67bb2067fc25c4a0709b9410de99622976f` (850448 bytes)
- `Paper\main.log`: `8d00fc27177f2a129c6412a1d3fbe20472d5f7f767fed2ff6ac3caee13f62f20` (71633 bytes)
- `Paper\main.blg`: `2e5ae2e4d837767f693cf40be1ebfac21fddaa539e03173ba6c6a075214f68a0` (1609 bytes)
- `configs\paper_external_proceedings_metadata_checks.json`: `d9ecfe176f8ed66e4a98964f20987dc422d3a8fd254c0bf48f7956ceb5c52437` (3249 bytes)
- `outputs\summary\paper_critical\final_paper_claim_audit_after_full_panel_review_20260612.json`: `ceb6ff1eb1393580dab990e58b37afe8ff1dcb4324134ca2d04969c6c376a6db` (13282 bytes)
- `outputs\summary\paper_critical\final_full_manuscript_panel_review_20260612.json`: `6cbad714fe6488e649bc580c1e952054ebcc6a37075050ecf0e47c4152fbad84` (4596 bytes)
- `outputs\summary\paper_critical\final_pdf_polish_metadata_followup_20260612.json`: `dea9fc1b54dbe6ca37cc228c303e332a08009cb0d1968e5ae3a16a2d898c7e7d` (3056 bytes)
- `configs\paper_submission_profiles.json`: `00a083d7fca48c521814ce9effe6bf9294c7ddf7e071879edaa4739f39a67d2d` (1276 bytes)
- `configs\paper_submission_metadata.json`: `5e6b232e27ec4fd2fc052e0d60c3e78b1b2077d8c7f4f53888ab304b6265e284` (1088 bytes)
- `configs\paper_manual_submission_checklist.json`: `e8bbef232cc418ca45976cb675411e2128075b6d8eae6d3ce23b561ecf666174` (4732 bytes)
- `outputs\summary\paper_critical\review_continuation_packet_20260613.json`: `f53c64b369c099fa926922864973779844a9abda927a1258334d4b640915631e` (13899 bytes)
- `scripts\audit\main_refresh_pre_submission_gates.py`: `757036636bee7c32932529349793490c3186dd8806d00daa39a1a29ac16162a6` (22396 bytes)
- `scripts\audit\main_audit_external_proceedings_metadata.py`: `0663e10b9d9aefb74346c23f63241167b50654f591acb0c7ec0c6373474389ff` (27205 bytes)
- `scripts\audit\main_audit_submission_package.py`: `f2871b6b18d8433163463547e945dd76ddce314b44d2721238c8355940aa8bc5` (34995 bytes)
- `scripts\audit\main_build_submission_source_package.py`: `9de0d0402285cab61cb6b5acd8a1c6e1870bac5e3365b495d126ddd292ec2796` (18252 bytes)
- `scripts\audit\main_audit_submission_source_package_rebuild.py`: `4c5ae6571bd2ba5e82e88eb43b3aa4aaaef35dee5330b429d0ce6188498dc726` (21775 bytes)
- `scripts\audit\main_build_submission_metadata_packet.py`: `01b66efe8a816d5681cef351fdf87a594ad935193e1f9bf2336f5d8a559e513c` (10113 bytes)
- `scripts\audit\main_build_manual_submission_checklist.py`: `c6e50da192f1c8ee970e96e0241d2d614f739b2507b258dec88877d7d0d9dce3` (21881 bytes)
- `scripts\audit\main_build_final_submission_gate.py`: `47d6bc59bd6c3fa532d424fca0cb61ac5ab7481d5517a5a6531a0f787f21004c` (15125 bytes)
- `scripts\audit\main_build_review_continuation_packet.py`: `41a502d8b13b98d88f0a2394c84fe103d91c03f832747e0a8f3ff6e93d58d7c1` (26878 bytes)
- `scripts\audit\main_audit_pre_submission_refresh_freshness.py`: `6c0ebf3d3243b552005226cee4cc5d2065d3b2c3c55dfc95f7016545e0740f91` (13914 bytes)

## Remaining Blockers

- promax:final_page_range_missing_in_bib
- promax:crossref_registry_not_visible:status=404
- promax:doi_resolver_not_visible:status=404
- Final manual submission-system metadata/format checklist is not closed.
- ProMax final ACM page range and ACM/Crossref registry visibility must be rechecked immediately before submission.
- Final submission package still needs the external submission-target-specific formatting pass.
- confirm_anonymous_shell:target_formatting_profile_not_ok
- confirm_anonymous_shell:target_profile_not_ok
- confirm_external_proceedings_metadata:external_proceedings_metadata_ready_not_closed
- manual_submission_system_items_not_confirmed
- external_proceedings_metadata_not_ready
- manual_submission_system_not_ready
- review_panel_coverage_not_complete
- promax:crossref_registry_not_visible
- promax:doi_resolver_not_visible
- explicit_claude_opus_review

## Failures

- `submission_package:page_count_exceeds_limit:15 > 9`
- `submission_package:overfull_hbox_count:8 > 0`
- `submission_package:target_profile:target_profile_page_count_exceeds_limit:15 > 9`
- `submission_package:target_profile:target_profile_requires_no_overfull_hbox`
- `submission_source_package:submission_package_audit_not_ok`
- `submission_source_package:submission_package_not_ready_for_target_formatting`
- `submission_source_package:submission_package_audit_failures_not_empty_or_missing`
- `submission_source_package_rebuild:source_package_not_ok`
- `submission_source_package_rebuild:source_package_not_ready`
- `submission_source_package_rebuild:source_package_failures_not_empty_or_missing`
- `submission_source_package_rebuild:copied_manifest_missing_files`
- `submission_metadata_packet:submission_package_audit_not_ok`
- `submission_metadata_packet:submission_package_not_ready_for_target_formatting`
- `submission_metadata_packet:target_profile_not_ok`
- `manual_submission_checklist:submission_metadata_packet_not_ok`
- `manual_submission_checklist:submission_metadata_packet_not_ready`
- `manual_submission_checklist:submission_package_audit_not_ok`
- `manual_submission_checklist:submission_package_not_ready_for_target_formatting`
- `final_submission_gate:submission_package:not_ok`
- `final_submission_gate:submission_package:not_ready`
- `final_submission_gate:submission_package:page_count_exceeds_limit:15 > 9`
- `final_submission_gate:submission_package:overfull_hbox_count:8 > 0`
- `final_submission_gate:submission_package:target_profile:target_profile_page_count_exceeds_limit:15 > 9`
- `final_submission_gate:submission_package:target_profile:target_profile_requires_no_overfull_hbox`
- `final_submission_gate:submission_metadata_packet:not_ok`
- `final_submission_gate:submission_metadata_packet:not_ready`
- `final_submission_gate:submission_metadata_packet:submission_package_audit_not_ok`
- `final_submission_gate:submission_metadata_packet:submission_package_not_ready_for_target_formatting`
- `final_submission_gate:submission_metadata_packet:target_profile_not_ok`
- `final_submission_gate:submission_source_package_rebuild:not_ok`
- `final_submission_gate:submission_source_package_rebuild:not_ready`
- `final_submission_gate:submission_source_package_rebuild:source_package_not_ok`
- `final_submission_gate:submission_source_package_rebuild:source_package_not_ready`
- `final_submission_gate:submission_source_package_rebuild:source_package_failures_not_empty_or_missing`
- `final_submission_gate:submission_source_package_rebuild:copied_manifest_missing_files`
- `final_submission_gate:manual_submission_checklist:not_ok`
- `final_submission_gate:manual_submission_checklist:submission_metadata_packet_not_ok`
- `final_submission_gate:manual_submission_checklist:submission_metadata_packet_not_ready`
- `final_submission_gate:manual_submission_checklist:submission_package_audit_not_ok`
- `final_submission_gate:manual_submission_checklist:submission_package_not_ready_for_target_formatting`

## Warnings

- `external_proceedings_metadata:proex:crossref_not_visible:status=404`
- `external_proceedings_metadata:proex:doi_resolver_not_visible:status=404`
- `external_proceedings_metadata:proex:crossref_discovery_alternate_doi_candidates_present`
- `external_proceedings_metadata:promax:crossref_discovery_alternate_doi_candidates_present`
- `submission_package:underfull_layout_warnings:hbox=10,vbox=12`
- `submission_metadata_packet:abstract_word_count_outside_common_range:419`
- `final_submission_gate:underfull_layout_warnings:hbox=10,vbox=12`
- `final_submission_gate:abstract_word_count_outside_common_range:419`
- `final_submission_gate:proex:crossref_not_visible:status=404`
- `final_submission_gate:proex:doi_resolver_not_visible:status=404`
- `final_submission_gate:proex:crossref_discovery_alternate_doi_candidates_present`
- `final_submission_gate:promax:crossref_discovery_alternate_doi_candidates_present`
- `final_submission_gate:underfull_layout_warnings:hbox=6,vbox=8`
- `final_submission_gate:rebuilt_underfull_layout_warnings:hbox=6,vbox=8`
- `final_submission_gate:acm_dl_not_accessible:status=403`
- `final_submission_gate:refresh_recorded_tracked_dirty_inputs_before_generation`

## Next Actions

- Use final_submission_gate as the first-read final submission status.
- Rerun this refresh after any Paper, BibTeX, target profile, or submission metadata change.
- Keep private author/COI/reviewer/declaration fields outside the repository.
