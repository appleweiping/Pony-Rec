import csv
import json

from scripts.analysis.main_export_ccrp_scored_rows_from_signal import export_ccrp_scored_rows


def _write(path, text):
    path.write_text(text, encoding="utf-8")
    return path


def test_export_ccrp_scored_rows_from_saved_signal(tmp_path):
    signal = _write(
        tmp_path / "test_calibrated.jsonl",
        '{"source_event_id":"e1","user_id":"u1","candidate_item_id":"i1",'
        '"relevance_probability":0.9,"calibrated_confidence":0.8,'
        '"evidence_support":0.9,"counterevidence_strength":0.1}\n'
        '{"source_event_id":"e1","user_id":"u1","candidate_item_id":"i2",'
        '"relevance_probability":0.2,"calibrated_confidence":0.3,'
        '"evidence_support":0.2,"counterevidence_strength":0.5}\n',
    )
    candidates = _write(
        tmp_path / "candidate_items.csv",
        "source_event_id,user_id,item_id\n"
        "e1,u1,i1\n"
        "e1,u1,i2\n",
    )
    selected = _write(
        tmp_path / "selected_valid_config.json",
        json.dumps(
            {
                "score_mode": "confidence_plus_evidence",
                "ablation": "full",
                "eta": 1.0,
                "confidence_weight": 0.7,
                "weight_boundary": 0.5,
                "weight_calibration_gap": 0.3,
                "weight_evidence": 0.2,
            }
        ),
    )

    provenance = export_ccrp_scored_rows(
        signal_path=signal,
        candidate_items_path=candidates,
        output_dir=tmp_path / "out",
        domain="toy",
        selected_config_json=selected,
        expected_rows=2,
    )

    assert provenance["audit_ok"] is True
    assert provenance["score_coverage_rate"] == 1.0
    assert provenance["scored_rows"] == 2
    assert provenance["score_rows"] == 2
    assert provenance["score_mode"] == "confidence_plus_evidence"

    scored_rows = list(csv.DictReader((tmp_path / "out" / "ccrp_scored_rows.csv").open(encoding="utf-8")))
    assert {row["candidate_item_id"] for row in scored_rows} == {"i1", "i2"}
    assert "ccrp_uncertainty" in scored_rows[0]
    assert "ccrp_boundary_uncertainty" in scored_rows[0]

    score_rows = list(csv.DictReader((tmp_path / "out" / "ccrp_scores.csv").open(encoding="utf-8")))
    assert [row["item_id"] for row in score_rows] == ["i1", "i2"]
    assert all(row["score"] for row in score_rows)

    saved_prov = json.loads((tmp_path / "out" / "ccrp_scored_rows_provenance.json").read_text(encoding="utf-8"))
    assert saved_prov["status_label"] == "ccrp_scored_rows_rebuilt_from_saved_signal"
    assert saved_prov["candidate_key_count"] == 2
