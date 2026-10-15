"""C-CO12: PGD epsilon=3 vs SVD Gate vs SVD Guided.

Place this script next to the CEBRA-SVD fork directory and run:
    python -u run_CCO12_CEBRA_SVD_verbose_results_v3.py

Show all CEBRA progress, and save R2 metrics as soon as each model finishes.
Results go to a new, unique run directory; no checkpoints or predictions are saved.
"""

from pathlib import Path
import csv
import gc
import json
import inspect
import os
import shutil
import tempfile
import time
from datetime import datetime, timezone
import random
import sys

import numpy as np
import torch
from torch import nn
from sklearn.metrics import r2_score

# Prevent the local fork import from generating __pycache__ files.
sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parent
CEBRA_DIR = ROOT / "CEBRA-SVD"
DATA_DIR = Path("/data/hossein/mm_project/perich_data_valid_final_raw/")
SESSION = "C-CO12"
DATA_PATH = DATA_DIR / f"{SESSION}.npz"
# New name, separate from all prior run output folders.
OUT_ROOT = ROOT / "CCO12_SVD_VERBOSE_RESULTS_V3"

SEED = 42
EPSILON = 3.0
EPS_STEPS = 10
EPS_ALPHA = EPSILON / 5  # 0.6; relevant in continuous mode.
ATTACK_NORM = "l2"
INPUT_DOMAIN = "counts"  # Raw nonnegative integer spike counts.

# SVD parameters: the same in both spectral arms.
SVD_TAU = 0.05
SVD_WEIGHT = 1.0
SVD_CENTER = True

LATENT_DIM = 64
HIDDEN = 64
BATCH_SIZE = 2048
MAX_ITER = 3000
TEMPERATURE = 0.4
ARCH = "offset36-model-more-dropout"
DEVICE = "cuda_if_available"

# EXACT decoder architecture and hyperparameters from the supplied idea-1 runner.
MLP_EPOCHS = 2500
MLP_HIDDEN = 64
MLP_DROP = 0.4
MLP_LR = 1e-3
EVAL_BATCH_SIZE = 8192


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_fork():
    if not (CEBRA_DIR / "cebra" / "__init__.py").is_file():
        raise FileNotFoundError(
            f"Expected fork directory: {CEBRA_DIR}\n"
            "Put this runner next to the folder named CEBRA-SVD."
        )
    # Avoid accidentally using an installed/unrelated version of CEBRA.
    for key in list(sys.modules):
        if key == "cebra" or key.startswith("cebra."):
            del sys.modules[key]
    sys.path.insert(0, str(CEBRA_DIR.resolve()))
    import cebra
    if not Path(cebra.__file__).resolve().is_relative_to(CEBRA_DIR.resolve()):
        raise RuntimeError(f"Incorrect CEBRA package imported: {cebra.__file__}")
    params = inspect.signature(cebra.CEBRA.__init__).parameters
    required = {"adv_svd_mode", "adv_svd_tau", "adv_svd_weight", "adv_input_domain"}
    missing = required - params.keys()
    if missing:
        raise RuntimeError(f"CEBRA-SVD fork lacks expected parameters: {sorted(missing)}")
    return cebra


def load_data():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Session file not found: {DATA_PATH}")
    with np.load(DATA_PATH, allow_pickle=False) as f:
        x_train, x_valid, y_train, y_valid = (
            np.asarray(f[key], dtype=np.float32)
            for key in ("train_data", "valid_data", "train_label", "valid_label")
        )

    if y_train.ndim == 1:
        y_train = y_train[:, None]
    if y_valid.ndim == 1:
        y_valid = y_valid[:, None]

    for key, arr in (('train_data', x_train), ('valid_data', x_valid),
                     ('train_label', y_train), ('valid_label', y_valid)):
        if arr.ndim != 2 or not all(arr.shape) or not np.isfinite(arr).all():
            raise ValueError(f"Invalid {key} shape or NaN/Inf: {arr.shape}")
    if x_train.shape[1] != x_valid.shape[1] or y_train.shape[1] != y_valid.shape[1]:
        raise ValueError("Train/validation feature or label dimensions differ")
    if len(x_train) != len(y_train) or len(x_valid) != len(y_valid):
        raise ValueError("Number of labels and neural samples must match")
    if min(len(x_train), len(x_valid)) < 36:
        raise ValueError("offset36 encoder requires >=36 time bins per split")
    if INPUT_DOMAIN == "counts":
        for name, arr in (("train_data", x_train), ("valid_data", x_valid)):
            if np.any(arr < 0) or not np.array_equal(arr, np.rint(arr)):
                raise ValueError(
                    f"{name} contains non-integer or negative values. "
                    "INPUT_DOMAIN='counts' needs raw binned spike counts. "
                    "If the data were normalized/smoothed, use 'continuous' consistently "
                    "for all three methods instead."
                )
    return tuple(np.ascontiguousarray(a) for a in (x_train, x_valid, y_train, y_valid))


def encoder_config(svd_mode: str) -> dict:
    return dict(
        model_architecture=ARCH,
        batch_size=BATCH_SIZE,
        temperature=TEMPERATURE,
        time_offsets=1,
        max_iterations=MAX_ITER,
        output_dimension=LATENT_DIM,
        num_hidden_units=HIDDEN,
        learning_rate=3e-4,
        temperature_mode="constant",
        training_mode="adversarial",
        adv_epsilon=EPSILON,
        adv_alpha=EPS_ALPHA,
        adv_steps=EPS_STEPS,
        attack_norm=ATTACK_NORM,
        adv_input_domain=INPUT_DOMAIN,
        adv_svd_mode=svd_mode,
        adv_svd_tau=SVD_TAU,
        adv_svd_weight=SVD_WEIGHT,
        adv_svd_center=SVD_CENTER,
        adv_epsilon_factors=(1.0,),
        adv_best_iterate=False,  # Plain PGD baseline; gate/guided keep best feasible internally.
        adv_rate_tau=None,
        device=DEVICE,
        verbose=True,  # CEBRA training progress and iteration/loss logs.
    )


class TwoLayerMLP(nn.Module):
    def __init__(self, dim, out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, MLP_HIDDEN), nn.LayerNorm(MLP_HIDDEN),
            nn.ReLU(), nn.Dropout(MLP_DROP), nn.Linear(MLP_HIDDEN, out),
        )

    def forward(self, x):
        return self.net(x)


def train_decoder(z_train, y_train, seed, device):
    seed_all(seed)  # Same decoder seed in all three arms.
    decoder = TwoLayerMLP(z_train.shape[1], y_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=MLP_LR)
    criterion = nn.MSELoss()
    z = torch.as_tensor(z_train, dtype=torch.float32, device=device)
    y = torch.as_tensor(y_train, dtype=torch.float32, device=device)
    decoder.train()
    print(f"[DECODER] Start {MLP_EPOCHS} epochs; all {y_train.shape[1]} label columns", flush=True)
    decoder_start = time.perf_counter()
    for epoch in range(MLP_EPOCHS):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(decoder(z), y)
        if not torch.isfinite(loss).item():
            raise FloatingPointError(f"Nonfinite decoder loss at epoch {epoch + 1}")
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 250 == 0 or epoch + 1 == MLP_EPOCHS:
            print(
                f"[DECODER] epoch {epoch + 1}/{MLP_EPOCHS} "
                f"| train_MSE={loss.item():.6f} "
                f"| elapsed={time.perf_counter() - decoder_start:.1f}s",
                flush=True,
            )
    return decoder


def predict_decoder(decoder, embeddings):
    decoder.eval()  # Disable Dropout for R2 evaluation.
    device = next(decoder.parameters()).device
    preds = []
    with torch.inference_mode():
        for start in range(0, len(embeddings), EVAL_BATCH_SIZE):
            z = torch.as_tensor(
                embeddings[start:start + EVAL_BATCH_SIZE],
                dtype=torch.float32, device=device,
            )
            preds.append(decoder(z).cpu().numpy())
    return np.concatenate(preds, axis=0)


LABELS = {"none": "PGD_EPS3", "gate": "SVD_GATE_EPS3", "guided": "SVD_GUIDED_EPS3"}
MODES = ("none", "gate", "guided")


def run_model(cebra, mode: str, arrays) -> dict:
    x_train, x_valid, y_train, y_valid = arrays
    encoder = decoder = z_train = z_valid = None
    arm_start = time.perf_counter()
    label = LABELS[mode]
    try:
        seed_all(SEED)  # Same initialization and sampling seed in each arm.
        print(f"\n{'=' * 75}", flush=True)
        print(f"[START] {label} | seed={SEED} | eps={EPSILON} | {MAX_ITER} CEBRA iterations", flush=True)
        print(f"[ENCODER] {label}: training started", flush=True)
        encoder = cebra.CEBRA(**encoder_config(mode))
        encoder.fit(x_train, y_train)
        encoder_seconds = time.perf_counter() - arm_start
        print(f"[ENCODER] {label}: completed in {encoder_seconds / 3600:.2f} hours", flush=True)

        print(f"[EMBEDDING] {label}: transforming train / validation", flush=True)
        z_train = np.asarray(encoder.transform(x_train), dtype=np.float32)
        z_valid = np.asarray(encoder.transform(x_valid), dtype=np.float32)
        if (len(z_train) != len(x_train) or len(z_valid) != len(x_valid)
                or not np.isfinite(z_train).all() or not np.isfinite(z_valid).all()):
            raise ValueError(f"Bad embeddings or alignment for mode={mode}")

        print(f"[DECODER] {label}: training started", flush=True)
        decoder_start = time.perf_counter()
        decoder = train_decoder(z_train, y_train, SEED + 100_000,
                                torch.device(encoder.device_))
        decoder_seconds = time.perf_counter() - decoder_start
        print(f"[DECODER] {label}: completed in {decoder_seconds / 60:.1f} minutes", flush=True)

        predictions = predict_decoder(decoder, z_valid)
        if predictions.shape != y_valid.shape or not np.isfinite(predictions).all():
            raise ValueError(f"Bad decoder prediction for mode={mode}")
        per_label = np.atleast_1d(r2_score(y_valid, predictions, multioutput="raw_values"))
        if not np.isfinite(per_label).all():
            raise ValueError(f"Nonfinite R2 score for mode={mode}: {per_label}")
        row = {
            "model": label, "svd_mode": mode, "seed": SEED,
            "epsilon": EPSILON, "valid_mean_r2": float(per_label.mean()),
            "valid_r2_per_label": [float(v) for v in per_label],
            "encoder_seconds": float(encoder_seconds),
            "decoder_seconds": float(decoder_seconds),
            "total_seconds": float(time.perf_counter() - arm_start),
        }
        print(f"[RESULT] {label}: validation mean R2 = {row['valid_mean_r2']:.6f}", flush=True)
        print(f"[RESULT] {label}: R2 per label = {row['valid_r2_per_label']}", flush=True)
        return row
    finally:
        del encoder, decoder, z_train, z_valid
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def save_results(out: Path, results: list[dict]):
    """Persist all completed arms atomically, even if a later arm is slow/fails."""
    json_path = out / "results_so_far.json"
    temp_json = out / ".results_so_far.json.tmp"
    temp_json.write_text(
        json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temp_json, json_path)

    csv_path = out / "results_so_far.csv"
    temp_csv = out / ".results_so_far.csv.tmp"
    num_labels = len(results[0]["valid_r2_per_label"])
    with temp_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["model", "svd_mode", "seed", "epsilon", "valid_mean_r2"]
            + [f"r2_label_{i}" for i in range(num_labels)]
            + ["encoder_seconds", "decoder_seconds", "total_seconds"]
        )
        for row in results:
            writer.writerow([
                row["model"], row["svd_mode"], row["seed"], row["epsilon"],
                row["valid_mean_r2"], *row["valid_r2_per_label"],
                row["encoder_seconds"], row["decoder_seconds"], row["total_seconds"],
            ])
    os.replace(temp_csv, csv_path)


def main():
    # mkdtemp uses atomic creation; runs cannot overwrite one another.
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = Path(tempfile.mkdtemp(prefix=f"CCO12_seed{SEED}_{stamp}_", dir=OUT_ROOT))
    print(f"[START] New experiment: {SESSION} | output: {out}", flush=True)
    print(f"[START] Modes: {MODES} | epsilon: {EPSILON} | seed: {SEED}", flush=True)
    cebra = load_fork()
    print(f"[FORK] {cebra.__file__}", flush=True)
    arrays = load_data()
    print(f"[DATA] Train X={arrays[0].shape}, Y={arrays[2].shape}", flush=True)
    print(f"[DATA] Valid X={arrays[1].shape}, Y={arrays[3].shape}", flush=True)

    config = {
        "session": SESSION, "data_path": str(DATA_PATH),
        "fork": str(CEBRA_DIR), "cebra_import": str(cebra.__file__),
        "modes": list(MODES), "seed": SEED,
        "encoder_configs": {mode: encoder_config(mode) for mode in MODES},
        "decoder": {
            "architecture": "Linear -> LayerNorm -> ReLU -> Dropout -> Linear",
            "epochs": MLP_EPOCHS, "hidden": MLP_HIDDEN,
            "dropout": MLP_DROP, "learning_rate": MLP_LR,
        },
        "shapes": {"x_train": arrays[0].shape, "x_valid": arrays[1].shape,
                   "y_train": arrays[2].shape, "y_valid": arrays[3].shape},
    }
    (out / "run_config.json").write_text(
        json.dumps(config, indent=2, allow_nan=False), encoding="utf-8"
    )

    results = []
    try:
        for mode in MODES:
            row = run_model(cebra, mode, arrays)
            results.append(row)
            save_results(out, results)  # Save right after each completed decoder.
            print(f"[SAVED] {len(results)}/{len(MODES)} results -> {out / 'results_so_far.csv'}", flush=True)
    except BaseException as exc:
        print(f"[INTERRUPTED] {type(exc).__name__}: {exc}", flush=True)
        print(f"[RECOVERY] Completed models' results (if any) remain in: {out}", flush=True)
        raise

    # Canonical final files; partial files remain available from earlier completions.
    shutil.copyfile(out / "results_so_far.csv", out / "results.csv")
    shutil.copyfile(out / "results_so_far.json", out / "results.json")
    print("\n" + "=" * 75, flush=True)
    print("FINAL VALIDATION R2 (all label columns)", flush=True)
    for row in results:
        print(f"{row['model']}: R2 = {row['valid_mean_r2']:.6f} | "
              f"per_label = {row['valid_r2_per_label']}", flush=True)
    print(f"[COMPLETE] Results saved in: {out}", flush=True)


if __name__ == "__main__":
    main()

