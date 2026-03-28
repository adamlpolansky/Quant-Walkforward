from __future__ import annotations

import pytest
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge

from qwf.models import build_model


@pytest.mark.parametrize(
    ("model_name", "model_params", "expected_type"),
    [
        ("linear", {}, LinearRegression),
        ("ridge", {"alpha": 1.0}, Ridge),
        ("lasso", {"alpha": 0.001}, Lasso),
        ("elasticnet", {"alpha": 0.001, "l1_ratio": 0.5}, ElasticNet),
    ],
)
def test_build_model_returns_expected_estimator_type(
    model_name: str,
    model_params: dict[str, float],
    expected_type: type,
) -> None:
    model = build_model(model_name, model_params)
    assert isinstance(model, expected_type)


def test_build_model_rejects_invalid_model_name() -> None:
    with pytest.raises(ValueError, match="Unsupported model_name"):
        build_model("random_forest", {})
