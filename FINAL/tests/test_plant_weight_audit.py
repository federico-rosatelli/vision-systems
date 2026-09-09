import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.plant_weight_audit import (
    normalized_area_weights,
    summarize_manual_review,
)


@dataclass
class Region:
    green_area: int


def test_normalized_area_weights_follow_mask_pixels():
    weights = normalized_area_weights([Region(100), Region(300), Region(600)])
    np.testing.assert_allclose(weights, [0.1, 0.3, 0.6])
    assert np.isclose(weights.sum(), 1.0)


def test_zero_area_has_no_artificial_weight():
    weights = normalized_area_weights([Region(0), Region(0)])
    np.testing.assert_array_equal(weights, [0.0, 0.0])


def test_manual_summary_requires_complete_review(tmp_path):
    pd.DataFrame({"review_status": ["pass", ""], "area_weight_usable": [True, False]}).to_csv(
        tmp_path / "plant_weight_audit.csv", index=False
    )
    (tmp_path / "plant_weight_audit_summary.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        summarize_manual_review(tmp_path)


def test_manual_summary_applies_ninety_percent_rule(tmp_path):
    pd.DataFrame({
        "review_status": ["pass"] * 9 + ["fail"],
        "area_weight_usable": [True] * 9 + [False],
    }).to_csv(tmp_path / "plant_weight_audit.csv", index=False)
    (tmp_path / "plant_weight_audit_summary.json").write_text(
        json.dumps({"manual_review_complete": False}), encoding="utf-8"
    )
    summary = summarize_manual_review(tmp_path)
    assert summary["usable_rate"] == 0.9
    assert summary["exit_criterion_met"] is True
