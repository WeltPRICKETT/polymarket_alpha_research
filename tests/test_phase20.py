import json

import pandas as pd

from src.audit import academic_evidence as ae
from src.audit.phase0 import render_markdown


def test_academic_evidence_pack_writes_research_guardrails(tmp_path, monkeypatch):
    results_dir = tmp_path / "results"
    audit_dir = tmp_path / "docs" / "audit"
    results_dir.mkdir(parents=True)
    audit_dir.mkdir(parents=True)
    monkeypatch.setattr(ae, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(ae, "AUDIT_DIR", audit_dir)
    monkeypatch.setattr(ae, "EVIDENCE_CSV", results_dir / "academic_evidence_table.csv")
    monkeypatch.setattr(ae, "EVIDENCE_JSON", results_dir / "academic_evidence_summary.json")
    monkeypatch.setattr(ae, "EVIDENCE_MD", audit_dir / "phase20_academic_evidence.md")

    (results_dir / "random_baseline_stability_summary.json").write_text(json.dumps({
        "status": "ok",
        "seed_count": 20,
        "model_vs_random": {
            "total_net_profit": {
                "status": "ok",
                "observed": 100.0,
                "random_percentile": 0.95,
                "right_tail_p_value": 0.10,
                "random_seed_count": 20,
            }
        },
    }), encoding="utf-8")
    (results_dir / "stratified_random_baseline_summary.json").write_text(json.dumps({
        "model_vs_matched_random": {}
    }), encoding="utf-8")
    (results_dir / "fold_matched_random_baseline_summary.json").write_text(json.dumps({
        "matched_rows": 15,
        "matched_rank_count": 3,
        "model_vs_fold_matched_random": {
            "total_net_profit": {
                "status": "ok",
                "observed": 100.0,
                "random_percentile": 1.0,
                "right_tail_p_value": 0.25,
                "random_seed_count": 3,
            }
        },
        "bootstrap_model_minus_matched_random": {
            "status": "ok",
            "bootstrap_samples": 1000,
            "intervals": {
                "total_net_profit": {
                    "status": "ok",
                    "p025": -10.0,
                    "p975": 50.0,
                    "probability_model_beats_matched_random": 0.90,
                }
            },
        },
    }), encoding="utf-8")
    (results_dir / "rolling_origin_summary.json").write_text(json.dumps({
        "valid_fold_count": 5,
        "aggregate": {"mean_precision_at_k": 0.4},
    }), encoding="utf-8")
    (results_dir / "label_summary.json").write_text(json.dumps({
        "label_strategy": "future_return",
        "label_independence": "independent_future_return",
        "is_trainable_for_binary_eval": True,
        "future_label_coverage": {"eligible_rows": 123},
    }), encoding="utf-8")
    (results_dir / "phase0_baseline.json").write_text(json.dumps({
        "data_quality": {"transactions": {"row_count": 1000}}
    }), encoding="utf-8")

    summary = ae.build_academic_evidence_pack()

    assert summary["status"] == "ok"
    assert "future_outcome_label" in summary["academic_guardrails"]
    evidence = pd.read_csv(results_dir / "academic_evidence_table.csv")
    assert {"evidence_family", "research_question", "estimand", "metric"}.issubset(evidence.columns)
    assert "fold_matched_bootstrap" in set(evidence["evidence_family"])
    markdown = (audit_dir / "phase20_academic_evidence.md").read_text(encoding="utf-8")
    assert "not a causal" in markdown
    assert "Methodological Guardrails" in markdown


def test_phase0_markdown_includes_phase20_rows_without_external_state():
    markdown = render_markdown({
        "generated_at": "2026-05-14T00:00:00+00:00",
        "project_root": "/tmp/project",
        "environment": {"python": "3.13.0", "platform": "test"},
        "git": {"branch": "main", "head": "abc123", "is_dirty": True},
        "data_quality": {"transactions": {}},
        "label_summary": {},
        "phase_checks": {
            "phase20_academic_evidence_summary": True,
            "phase20_academic_evidence_table": True,
            "phase20_academic_guardrails": True,
            "phase20_docs_exist": True,
            "phase20_academic_evidence_rows": 11,
        },
    })

    assert "Phase 1-24 Gate Snapshot" in markdown
    assert "Phase 20 academic evidence rows: `11`" in markdown
