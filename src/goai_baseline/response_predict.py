"""Inference and submission validation for response-decomposition runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .audit import assert_allowed_inputs
from .config import load_config
from .model import ResponseDecompositionMLP
from .response import ResponseFeatureBuilder
from .response_train import predict_with_response_model
from .schema import SAMPLE_ID, require_metadata_columns, require_unique_sample_ids
from .submission import verify_submission
from .train import resolve_device


def load_response_checkpoint(path: str | Path, device: torch.device) -> tuple[ResponseDecompositionMLP, ResponseFeatureBuilder, list[str]]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if payload.get("model_type") != "response_decomposition":
        raise ValueError("Checkpoint is not a response-decomposition model")
    if payload.get("target_scale") != "log2":
        raise ValueError("Checkpoint target scale is not log2")
    model = ResponseDecompositionMLP(**payload["model_kwargs"]).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, ResponseFeatureBuilder.from_state_dict(payload["feature_state"]), list(payload["proteins"])


def predict_response_test(config_path: str | Path, run_dir: str | Path, output_csv: str | Path | None = None) -> Path:
    config = load_config(config_path)
    assert_allowed_inputs(config)
    run = Path(run_dir)
    device = resolve_device(config.model.device)
    model, builder, proteins = load_response_checkpoint(run / "checkpoint.pt", device)
    metadata = pd.read_csv(config.data.metadata_test, low_memory=False)
    require_metadata_columns(metadata)
    require_unique_sample_ids(metadata, "submission metadata")
    metadata = metadata.set_index(SAMPLE_ID, verify_integrity=True)
    prediction = predict_with_response_model(model, builder, metadata, metadata.index, proteins, device)
    if not np.isfinite(prediction.to_numpy(dtype=float)).all():
        raise ValueError("Model produced non-finite predictions")
    output = Path(output_csv) if output_csv else run / "prediction.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    prediction.index.name = SAMPLE_ID
    prediction.to_csv(output)
    contract = verify_submission(output, config.data.metadata_test, proteins)
    with (output.parent / "prediction_contract.json").open("w", encoding="utf-8") as handle:
        json.dump(contract, handle, ensure_ascii=False, indent=2)
    print(json.dumps(contract, ensure_ascii=False, indent=2))
    print(f"Wrote submission: {output.resolve()}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a submission from a v2 response-decomposition run")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()
    predict_response_test(args.config, args.run_dir, args.output_csv)


if __name__ == "__main__":
    main()
