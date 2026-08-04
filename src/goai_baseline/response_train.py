"""Train the response-decomposition experiment on P0 condition features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from .audit import audit_inputs
from .config import BaselineConfig, load_config
from .evaluate import evaluate_predictor, write_evaluation
from .loss import masked_mse
from .manifest import write_manifest
from .model import ResponseDecompositionMLP
from .official_metrics import evaluate_official_proxy
from .preprocess import feature_contract, prepare_data
from .response import ResponseFeatureBuilder, response_targets
from .train import default_run_dir, resolve_device, set_seed


VARIANT = "v2_response_decomposition"


def predict_with_response_model(
    model: ResponseDecompositionMLP,
    builder: ResponseFeatureBuilder,
    metadata: pd.DataFrame,
    sample_ids: pd.Index,
    proteins: list[str],
    device: torch.device,
) -> pd.DataFrame:
    model.eval()
    full, background = builder.transform(metadata.loc[sample_ids])
    with torch.no_grad():
        prediction = model(torch.from_numpy(full).to(device), torch.from_numpy(background).to(device)).detach().cpu().numpy()
    return pd.DataFrame(prediction, index=sample_ids, columns=proteins)


def train_response_variant(
    config: BaselineConfig,
    run_dir: str | Path | None = None,
    fc_weight: float = 1.0,
) -> Path:
    if fc_weight < 0:
        raise ValueError("fc_weight must be non-negative")
    audit_inputs(config)
    set_seed(config.model.seed)
    device = resolve_device(config.model.device)
    data = prepare_data(config)
    output = Path(run_dir) if run_dir is not None else default_run_dir(config, VARIANT)
    output.mkdir(parents=True, exist_ok=False)

    builder = ResponseFeatureBuilder().fit(data.metadata, data.y_log2, data.train_ids)
    full, background = builder.transform(data.metadata.loc[data.train_ids])
    absolute_target = data.y_log2.loc[data.train_ids].fillna(0.0).to_numpy(dtype="float32")
    absolute_mask = data.mask.loc[data.train_ids].to_numpy(dtype="float32")
    fc_target, fc_mask, target_summary = response_targets(data)
    full_inputs = torch.from_numpy(full).to(device)
    background_inputs = torch.from_numpy(background).to(device)
    absolute_targets = torch.from_numpy(absolute_target).to(device)
    absolute_masks = torch.from_numpy(absolute_mask).to(device)
    fc_targets = torch.from_numpy(fc_target).to(device)
    fc_masks = torch.from_numpy(fc_mask).to(device)
    hidden_dim = max(1, config.model.hidden_dim // 2)
    model = ResponseDecompositionMLP(full.shape[1], background.shape[1], len(data.proteins), hidden_dim, config.model.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.model.learning_rate)
    history: list[dict[str, float | int]] = []
    for epoch in range(1, config.model.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(full_inputs, background_inputs)
        _, response = model.components(full_inputs, background_inputs)
        absolute_loss = masked_mse(prediction, absolute_targets, absolute_masks)
        response_loss = masked_mse(response, fc_targets, fc_masks)
        loss = absolute_loss + fc_weight * response_loss
        loss.backward()
        optimizer.step()
        history.append({"epoch": epoch, "total_loss": float(loss.detach().cpu()), "absolute_mse": float(absolute_loss.detach().cpu()), "fc_mse": float(response_loss.detach().cpu())})
        if epoch == 1 or epoch % 10 == 0 or epoch == config.model.epochs:
            print(f"epoch={epoch:03d} total={history[-1]['total_loss']:.6f} absolute={history[-1]['absolute_mse']:.6f} fc={history[-1]['fc_mse']:.6f}")

    checkpoint = {
        "model_type": "response_decomposition",
        "variant": VARIANT,
        "model_state_dict": model.state_dict(),
        "model_kwargs": {"full_input_dim": full.shape[1], "background_input_dim": background.shape[1], "output_dim": len(data.proteins), "hidden_dim": hidden_dim, "dropout": config.model.dropout},
        "feature_state": builder.state_dict(),
        "proteins": data.proteins,
        "target_scale": "log2",
        "fc_weight": fc_weight,
    }
    torch.save(checkpoint, output / "checkpoint.pt")
    pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
    with (output / "feature_contract.json").open("w", encoding="utf-8") as handle:
        json.dump(feature_contract(data, config), handle, ensure_ascii=False, indent=2)
    with (output / "feature_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(builder.summary(), handle, ensure_ascii=False, indent=2)
    with (output / "response_target_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(target_summary, handle, ensure_ascii=False, indent=2)

    def predictor(ids: pd.Index) -> pd.DataFrame:
        return predict_with_response_model(model, builder, data.metadata, ids, data.proteins, device)

    report, proteins = evaluate_predictor(data, predictor, VARIANT)
    write_evaluation(output, report, proteins)
    evaluate_official_proxy(data, predictor).to_csv(output / "official_proxy_metrics.csv", index=False)
    write_manifest(output / "manifest.json", config, {"variant": VARIANT, "model_type": "background_plus_response", "fc_weight": fc_weight, "device": str(device), "model_parameter_count": int(sum(parameter.numel() for parameter in model.parameters())), **target_summary})
    print(report.to_string(index=False))
    print(f"Wrote run: {output.resolve()}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the v2 background-plus-response model")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--fc-weight", type=float, default=1.0)
    args = parser.parse_args()
    train_response_variant(load_config(args.config), args.run_dir, args.fc_weight)


if __name__ == "__main__":
    main()
