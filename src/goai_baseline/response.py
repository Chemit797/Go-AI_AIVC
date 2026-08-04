"""Features and targets for the response-decomposition experiment."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .controls import exact_control_predictions
from .features import FeatureBuilder, _categories, _one_hot
from .preprocess import PreprocessedData
from .schema import MEDIUM, STRAIN, TEMPERATURE, TIME, control_mask, treatment_mask


BACKGROUND_CATEGORIES = (STRAIN, MEDIUM, TEMPERATURE, TIME)


@dataclass
class ResponseFeatureBuilder:
    """P0 full features plus a chemical-free background feature block."""

    full_builder: FeatureBuilder = field(default_factory=lambda: FeatureBuilder("p0_onehot"))
    background_categories: dict[str, list[str]] = field(default_factory=dict)

    def fit(self, metadata: pd.DataFrame, y_log2: pd.DataFrame, train_ids: pd.Index) -> "ResponseFeatureBuilder":
        self.full_builder.fit(metadata, y_log2, train_ids)
        train_metadata = metadata.loc[train_ids]
        self.background_categories = {name: _categories(train_metadata[name]) for name in BACKGROUND_CATEGORIES}
        return self

    def transform(self, metadata: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if not self.background_categories:
            raise RuntimeError("ResponseFeatureBuilder must be fit before transform")
        full = self.full_builder.transform(metadata)
        background = np.concatenate(
            [_one_hot(metadata[name], self.background_categories[name]) for name in BACKGROUND_CATEGORIES], axis=1, dtype=np.float32
        )
        return full, background

    def state_dict(self) -> dict[str, object]:
        return {"full_feature_state": self.full_builder.state_dict(), "background_categories": self.background_categories}

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "ResponseFeatureBuilder":
        builder = cls(full_builder=FeatureBuilder.from_state_dict(dict(state["full_feature_state"])))
        builder.background_categories = {str(name): list(values) for name, values in dict(state["background_categories"]).items()}
        return builder

    def summary(self) -> dict[str, object]:
        full_summary = self.full_builder.summary()
        return {
            "full_variant": "p0_onehot",
            "full_input_dim": full_summary["total_input_dim"],
            "background_category_dimensions": {name: len(values) for name, values in self.background_categories.items()},
            "background_input_dim": int(sum(len(values) for values in self.background_categories.values())),
        }


def response_targets(data: PreprocessedData) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Create zero-control and matched-treatment FC supervision from training rows only."""
    train_ids = data.train_ids
    target = np.zeros((len(train_ids), len(data.proteins)), dtype=np.float32)
    mask = np.zeros_like(target)
    positions = {sample_id: row for row, sample_id in enumerate(train_ids)}
    controls = train_ids[control_mask(data.metadata.loc[train_ids]).to_numpy()]
    if len(controls):
        rows = [positions[sample_id] for sample_id in controls]
        mask[rows] = data.y_log2.loc[controls].notna().to_numpy(dtype=np.float32)
    treatments = train_ids[treatment_mask(data.metadata.loc[train_ids]).to_numpy()]
    matched = exact_control_predictions(data.metadata, data.y_log2, treatments, train_ids)
    valid = treatments[matched.has_exact_match.to_numpy()]
    if len(valid):
        delta = data.y_log2.loc[valid] - matched.predictions.loc[valid]
        rows = [positions[sample_id] for sample_id in valid]
        target[rows] = delta.fillna(0.0).to_numpy(dtype=np.float32)
        mask[rows] = delta.notna().to_numpy(dtype=np.float32)
    return target, mask, {
        "n_train_rows": int(len(train_ids)),
        "n_control_response_rows": int(len(controls)),
        "n_matched_treatment_response_rows": int(len(valid)),
        "n_response_observed_values": int(mask.sum()),
    }
