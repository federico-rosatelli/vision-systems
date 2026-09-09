from scripts.run_aggregation_experiments import experiment_specs


def test_experiment_matrix_contains_four_methods_and_three_seeds():
    specs = experiment_specs({
        "aggregations": ["weighted", "uniform", "abmil", "gated_abmil"],
        "seeds": [42, 43, 44],
    })
    assert len(specs) == 12
    assert len({spec["run_name"] for spec in specs}) == 12
