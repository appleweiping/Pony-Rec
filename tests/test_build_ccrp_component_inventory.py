from scripts.audit.main_build_ccrp_component_inventory import build_inventory, write_markdown


def test_inventory_covers_current_ccrp_ablation_handles():
    inventory = build_inventory()

    assert inventory["status_label"] == "paper_critical_ccrp_component_inventory"
    assert inventory["paper_claim_ready"] is False
    assert set(inventory["ablation_handles_from_code"]) == {
        "full",
        "without_boundary_uncertainty",
        "without_calibration_gap",
        "without_counterevidence",
        "without_evidence_support",
        "without_risk_penalty",
    }
    component_ids = {component["id"] for component in inventory["components"]}
    assert {
        "boundary_uncertainty",
        "calibration_gap",
        "evidence_support_insufficiency",
        "counterevidence",
        "risk_penalty",
        "eta_risk_exponent",
        "confidence_weight",
        "uncertainty_weight_triple",
        "raw_vs_calibrated_posterior",
        "temperature_prompt_variants",
    }.issubset(component_ids)


def test_inventory_marks_components_blocked_by_missing_signal_rows():
    inventory = build_inventory()

    assert inventory["blocked_by"] == ["missing_full_scale_uncertainty_or_recomputable_signal_rows"]
    for component in inventory["components"]:
        assert component["paper_execution_status"] == "requires_full_scale_signal_rows"
        assert component["current_blocker"] == "missing_full_scale_uncertainty_or_recomputable_signal_rows"
    conceptual = next(component for component in inventory["components"] if component["id"] == "raw_vs_calibrated_posterior")
    assert conceptual["kind"] == "conceptual_not_currently_executable"
    assert conceptual["selector_handle"] == "no_current_cli_handle"


def test_inventory_detects_formula_alignment_and_risk_notes():
    inventory = build_inventory()

    assert inventory["formula_alignment"]["code_formula"] == "base_score * ((1 - uncertainty) ** eta)"
    assert inventory["formula_alignment"]["figure_formula_contains_multiplicative_form"] is True
    risk_component = next(component for component in inventory["components"] if component["id"] == "risk_penalty")
    assert any("multiplicative risk adjustment" in note for note in risk_component["risk_notes"])
    assert any("Eta has no ranking effect" in note for note in risk_component["risk_notes"])
    counter = next(component for component in inventory["components"] if component["id"] == "counterevidence")
    assert any("no-op" in note for note in counter["risk_notes"])


def test_write_markdown_inventory_summary(tmp_path):
    inventory = build_inventory()
    output = tmp_path / "inventory.md"

    write_markdown(output, inventory)

    text = output.read_text(encoding="utf-8")
    assert "C-CRP Component Inventory" in text
    assert "`risk_penalty`" in text
    assert "missing_full_scale_uncertainty_or_recomputable_signal_rows" in text
