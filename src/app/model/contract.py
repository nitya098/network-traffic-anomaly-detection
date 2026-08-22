from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class InputContract:
    n_features: int
    categorical_features: tuple[str, ...]
    numerical_features: tuple[str, ...]
    feature_columns_in_order: tuple[str, ...]
    output_classes: tuple[str, ...]

    @property
    def categorical_set(self) -> frozenset[str]:
        return frozenset(self.categorical_features)


def load_contract(path: Path) -> InputContract:
    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)

    feature_order = tuple(raw["feature_columns_in_order"])
    n_features = int(raw["n_features"])
    if len(feature_order) != n_features:
        raise ValueError(
            "Input contract n_features does not match feature_columns_in_order length."
        )

    return InputContract(
        n_features=n_features,
        categorical_features=tuple(raw["categorical_features"]),
        numerical_features=tuple(raw["numerical_features"]),
        feature_columns_in_order=feature_order,
        output_classes=tuple(raw["output_classes"]),
    )
