#!/usr/bin/env python3
"""Benchmark clean/noisy simulation for real-channel prediction.

What this script compares:
1) Domain similarity to real data:
   - mean PDP correlation / NMSE
   - temporal energy correlation / NMSE
2) Prediction quality on real data:
   - Traditional: Persistence, Linear AR (multi-output Ridge)
   - AI: MLP regressor (TensorFlow/Keras)

By default, it evaluates two simulation sources from cir_frame files:
- h_cir_clean (no synthetic noise)
- h_cir_noisy (with synthetic measurement noise)
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


@dataclass
class Standardizer:
    mean: np.ndarray | None = None
    std: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "Standardizer":
        self.mean = np.mean(x, axis=0, keepdims=True)
        self.std = np.std(x, axis=0, keepdims=True)
        self.std = np.where(self.std < 1e-12, 1.0, self.std)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Standardizer not fitted")
        return (x - self.mean) / self.std

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Standardizer not fitted")
        return x * self.std + self.mean


class MultiOutputRidge:
    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)
        self.w: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "MultiOutputRidge":
        if x.ndim != 2 or y.ndim != 2:
            raise ValueError("x/y must be 2D")
        if x.shape[0] != y.shape[0]:
            raise ValueError("x/y sample mismatch")
        xb = np.concatenate([x, np.ones((x.shape[0], 1), dtype=x.dtype)], axis=1)
        xtx = xb.T @ xb
        reg = self.alpha * np.eye(xtx.shape[0], dtype=x.dtype)
        reg[-1, -1] = 0.0
        rhs = xb.T @ y
        self.w = np.linalg.solve(xtx + reg, rhs)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise RuntimeError("model not fitted")
        xb = np.concatenate([x, np.ones((x.shape[0], 1), dtype=x.dtype)], axis=1)
        return xb @ self.w


def _resolve_real_path(real_path_arg: str) -> str:
    if real_path_arg:
        return real_path_arg
    candidates = (
        "/data/kmh/sionna-rt/mh/workspace/data_analyze/data/CIR.mat",
        "/home/mh/kmh/sionna-rt/mh/workspace/data_analyze/data/CIR.mat",
        "/home/mh/kmh/sionna-rt/mh/workspace/data_analyze/simulation_frames/CIR.mat",
    )
    for p in candidates:
        if Path(p).exists():
            print(f"[real] auto-detected real-path: {p}")
            return p
    return ""


def _layout_to_time_delay(arr: np.ndarray, layout: str) -> np.ndarray:
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got {arr.shape}")
    if layout == "time_delay":
        return arr
    if layout == "delay_time":
        return arr.T
    if arr.shape[0] > arr.shape[1]:
        return arr.T
    return arr


def _select_key(npz: np.lib.npyio.NpzFile, preferred: str | None, candidates: Iterable[str]) -> str:
    if preferred and preferred in npz.files:
        return preferred
    for c in candidates:
        if c in npz.files:
            return c
    if len(npz.files) == 1:
        return npz.files[0]
    raise KeyError(f"Failed to choose key from npz: {npz.files}")


def _load_frame_dir(frame_dir: Path, key: str | None) -> tuple[np.ndarray, str]:
    files = sorted(frame_dir.glob("cir_frame_*.npz"))
    if not files:
        raise FileNotFoundError(f"No cir_frame_*.npz in {frame_dir}")
    seq: list[np.ndarray] = []
    used = ""
    for p in files:
        with np.load(p) as data:
            k = _select_key(data, key, ("h_cir_noisy", "h_cir_clean", "pdp_noisy", "pdp_clean"))
            if not used:
                used = k
            seq.append(np.asarray(data[k]).reshape(-1))
    arr = np.stack(seq, axis=0)
    if not np.iscomplexobj(arr):
        arr = arr.astype(np.float64) + 0j
    return arr, used


def _load_file(path: Path, key: str | None, layout: str) -> tuple[np.ndarray, str]:
    ext = path.suffix.lower()
    if ext == ".npy":
        arr = np.squeeze(np.load(path))
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        arr = _layout_to_time_delay(arr, layout)
        if not np.iscomplexobj(arr):
            arr = arr.astype(np.float64) + 0j
        return arr, "npy_array"

    if ext == ".npz":
        with np.load(path) as data:
            sel = _select_key(data, key, ("CIR", "cir", "h_cir_noisy", "h_cir_clean"))
            arr = np.squeeze(np.asarray(data[sel]))
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        arr = _layout_to_time_delay(arr, layout)
        if not np.iscomplexobj(arr):
            arr = arr.astype(np.float64) + 0j
        return arr, sel

    if ext == ".mat":
        from scipy.io import loadmat

        mat = loadmat(path)
        if key and key in mat:
            sel = key
        elif "CIR" in mat:
            sel = "CIR"
        else:
            sel = next((k for k, v in mat.items() if not k.startswith("__") and np.ndim(v) >= 2), "")
            if not sel:
                raise KeyError(f"No 2D array found in {path}")
        arr = np.squeeze(np.asarray(mat[sel]))
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        arr = _layout_to_time_delay(arr, layout)
        if not np.iscomplexobj(arr):
            arr = arr.astype(np.float64) + 0j
        return arr, sel

    raise ValueError(f"Unsupported file type: {path}")


def load_cir(path: Path, key: str | None, layout: str) -> tuple[np.ndarray, str]:
    if path.is_dir():
        return _load_frame_dir(path, key)
    if path.is_file():
        return _load_file(path, key, layout)
    raise FileNotFoundError(path)


def complex_to_features(c: np.ndarray) -> np.ndarray:
    return np.concatenate([c.real, c.imag], axis=1).astype(np.float64)


def features_to_complex(x: np.ndarray) -> np.ndarray:
    if x.shape[1] % 2 != 0:
        raise ValueError("feature dim must be even")
    d = x.shape[1] // 2
    return x[:, :d] + 1j * x[:, d:]


def make_windows(seq: np.ndarray, window: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    t = seq.shape[0]
    need = window + horizon
    if t < need:
        raise ValueError(f"need >= {need} frames, got {t}")
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for end in range(window, t - horizon + 1):
        xs.append(seq[end - window : end].reshape(-1))
        ys.append(seq[end + horizon - 1])
    return np.asarray(xs), np.asarray(ys)


def nmse(a: np.ndarray, b: np.ndarray) -> float:
    num = np.mean(np.abs(a - b) ** 2)
    den = np.mean(np.abs(b) ** 2) + 1e-12
    return float(num / den)


def corrcoef_safe(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    sa = float(np.std(a))
    sb = float(np.std(b))
    if sa < 1e-12 or sb < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def similarity_metrics(sim_c: np.ndarray, real_c: np.ndarray) -> dict[str, float]:
    n = min(sim_c.shape[0], real_c.shape[0])
    sim = sim_c[:n]
    real = real_c[:n]

    sim_pdp_mean = np.mean(np.abs(sim) ** 2, axis=0)
    real_pdp_mean = np.mean(np.abs(real) ** 2, axis=0)

    sim_energy = np.sum(np.abs(sim) ** 2, axis=1)
    real_energy = np.sum(np.abs(real) ** 2, axis=1)

    return {
        "pdp_corr": corrcoef_safe(sim_pdp_mean, real_pdp_mean),
        "pdp_nmse": nmse(sim_pdp_mean, real_pdp_mean),
        "energy_corr": corrcoef_safe(sim_energy, real_energy),
        "energy_nmse": nmse(sim_energy, real_energy),
    }


def eval_prediction(y_true_c: np.ndarray, y_pred_c: np.ndarray) -> dict[str, float]:
    err = y_pred_c - y_true_c
    abs_err = np.abs(err)
    return {
        "mse": float(np.mean(abs_err**2)),
        "mae": float(np.mean(abs_err)),
        "nmse": float(np.mean(abs_err**2) / (np.mean(np.abs(y_true_c) ** 2) + 1e-12)),
    }


def train_ridge_predict(
    x_sim: np.ndarray,
    y_sim: np.ndarray,
    x_cal: np.ndarray,
    y_cal: np.ndarray,
    x_eval: np.ndarray,
    alpha: float,
) -> np.ndarray:
    sx = Standardizer().fit(x_sim)
    sy = Standardizer().fit(y_sim)
    model = MultiOutputRidge(alpha=alpha).fit(sx.transform(x_sim), sy.transform(y_sim))
    x_eval_n = sx.transform(x_eval)
    y_eval = sy.inverse_transform(model.predict(x_eval_n))

    if x_cal.shape[0] > 0:
        x_cal_n = sx.transform(x_cal)
        y_cal_base = sy.inverse_transform(model.predict(x_cal_n))
        residual = y_cal - y_cal_base
        rsx = Standardizer().fit(x_cal)
        rsy = Standardizer().fit(residual)
        rmodel = MultiOutputRidge(alpha=alpha).fit(rsx.transform(x_cal), rsy.transform(residual))
        y_eval += rsy.inverse_transform(rmodel.predict(rsx.transform(x_eval)))
    return y_eval


def train_mlp_predict(
    x_sim: np.ndarray,
    y_sim: np.ndarray,
    x_cal: np.ndarray,
    y_cal: np.ndarray,
    x_eval: np.ndarray,
    hidden: int,
    epochs: int,
    fine_tune_epochs: int,
    batch_size: int,
    lr: float,
    clip: float,
    seed: int,
) -> np.ndarray:
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()

    sx = Standardizer().fit(x_sim)
    sy = Standardizer().fit(y_sim)
    xn = np.clip(sx.transform(x_sim), -clip, clip).astype(np.float32)
    yn = sy.transform(y_sim).astype(np.float32)
    xe = np.clip(sx.transform(x_eval), -clip, clip).astype(np.float32)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(xn.shape[1],)),
            tf.keras.layers.Dense(
                hidden,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(1e-5),
            ),
            tf.keras.layers.Dense(
                max(32, hidden // 2),
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(1e-5),
            ),
            tf.keras.layers.Dense(yn.shape[1], activation="linear"),
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="loss",
            patience=max(3, epochs // 10),
            min_delta=1e-6,
            restore_best_weights=True,
        )
    ]
    model.fit(
        xn,
        yn,
        epochs=epochs,
        batch_size=min(batch_size, xn.shape[0]),
        verbose=0,
        shuffle=True,
        callbacks=callbacks,
    )

    if x_cal.shape[0] > 0 and fine_tune_epochs > 0:
        xc = np.clip(sx.transform(x_cal), -clip, clip).astype(np.float32)
        yc = sy.transform(y_cal).astype(np.float32)
        model.fit(
            xc,
            yc,
            epochs=fine_tune_epochs,
            batch_size=min(batch_size, xc.shape[0]),
            verbose=0,
            shuffle=True,
        )

    y_eval_n = model.predict(xe, verbose=0)
    y_eval = sy.inverse_transform(y_eval_n.astype(np.float64))
    if not np.all(np.isfinite(y_eval)):
        raise RuntimeError("MLP prediction produced non-finite values.")
    if np.max(np.abs(y_eval)) > 1e6:
        raise RuntimeError("MLP prediction exploded (>1e6); reduce lr/hidden or increase clipping.")
    return y_eval


def run(args: argparse.Namespace) -> None:
    np.random.seed(args.seed)

    real_path = _resolve_real_path(args.real_path)
    if not real_path:
        raise FileNotFoundError(
            "--real-path was empty and no default CIR.mat was found. "
            "Pass --real-path explicitly."
        )

    real_c, real_key_used = load_cir(Path(real_path), args.real_key, args.real_layout)
    print(f"[real] loaded shape={real_c.shape}, key={real_key_used}")

    sim_keys = [x.strip() for x in args.sim_keys.split(",") if x.strip()]
    if not sim_keys:
        raise ValueError("--sim-keys is empty")

    all_rows: list[dict[str, object]] = []

    for sim_key in sim_keys:
        sim_c, sim_used = load_cir(Path(args.sim_path), sim_key, args.sim_layout)
        d = min(args.max_delay, sim_c.shape[1], real_c.shape[1])
        sim_c = sim_c[:, :d]
        real_cd = real_c[:, :d]

        print()
        print(f"=== scenario: sim_key={sim_used} (delay_taps={d}) ===")

        simm = similarity_metrics(sim_c, real_cd)
        print(
            "[similarity] "
            f"pdp_corr={simm['pdp_corr']:.4f}, pdp_nmse={simm['pdp_nmse']:.4e}, "
            f"energy_corr={simm['energy_corr']:.4f}, energy_nmse={simm['energy_nmse']:.4e}"
        )

        sim_f = complex_to_features(sim_c)
        real_f = complex_to_features(real_cd)
        x_sim, y_sim = make_windows(sim_f, args.window, args.horizon)
        x_real, y_real = make_windows(real_f, args.window, args.horizon)

        n = x_real.shape[0]
        n_cal = int(round(n * args.calib_ratio))
        n_cal = max(0, min(n - 1, n_cal))
        x_cal = x_real[:n_cal]
        y_cal = y_real[:n_cal]
        x_eval = x_real[n_cal:]
        y_eval = y_real[n_cal:]
        if x_eval.shape[0] == 0:
            raise ValueError("No eval samples left; reduce --calib-ratio.")

        y_true_c = features_to_complex(y_eval)
        fdim = y_real.shape[1]

        y_persist_eval = x_eval.reshape(-1, args.window, fdim)[:, -1, :]
        m_persist = eval_prediction(y_true_c, features_to_complex(y_persist_eval))

        y_ridge_eval = train_ridge_predict(
            x_sim=x_sim,
            y_sim=y_sim,
            x_cal=x_cal,
            y_cal=y_cal,
            x_eval=x_eval,
            alpha=args.ridge_alpha,
        )
        m_ridge = eval_prediction(y_true_c, features_to_complex(y_ridge_eval))

        mlp_ok = True
        mlp_err = ""
        try:
            y_mlp_eval = train_mlp_predict(
                x_sim=x_sim,
                y_sim=y_sim,
                x_cal=x_cal,
                y_cal=y_cal,
                x_eval=x_eval,
                hidden=args.mlp_hidden,
                epochs=args.mlp_epochs,
                fine_tune_epochs=args.mlp_fine_tune_epochs,
                batch_size=args.mlp_batch_size,
                lr=args.mlp_lr,
                clip=args.mlp_input_clip,
                seed=args.seed,
            )
            m_mlp = eval_prediction(y_true_c, features_to_complex(y_mlp_eval))
        except Exception as exc:
            mlp_ok = False
            mlp_err = str(exc)
            m_mlp = {"mse": float("nan"), "mae": float("nan"), "nmse": float("nan")}

        print(
            f"[prediction] windows(real)={n}, cal={n_cal}, eval={x_eval.shape[0]}, "
            f"window={args.window}, horizon={args.horizon}"
        )
        print(
            f"  Persistence      MSE={m_persist['mse']:.6e} MAE={m_persist['mae']:.6e} NMSE={m_persist['nmse']:.6e}"
        )
        print(f"  LinearRidge      MSE={m_ridge['mse']:.6e} MAE={m_ridge['mae']:.6e} NMSE={m_ridge['nmse']:.6e}")
        if mlp_ok:
            print(f"  AI-MLP           MSE={m_mlp['mse']:.6e} MAE={m_mlp['mae']:.6e} NMSE={m_mlp['nmse']:.6e}")
        else:
            print(f"  AI-MLP           FAILED ({mlp_err})")

        base_nmse = m_persist["nmse"]
        impr_ridge = 100.0 * (1.0 - m_ridge["nmse"] / base_nmse)
        impr_mlp = 100.0 * (1.0 - m_mlp["nmse"] / base_nmse) if mlp_ok else float("nan")
        print(
            f"[improvement vs Persistence] "
            f"LinearRidge={impr_ridge:+.2f}% AI-MLP={impr_mlp:+.2f}%"
        )

        for model_name, mm in (
            ("Persistence", m_persist),
            ("LinearRidge", m_ridge),
            ("AI-MLP", m_mlp),
        ):
            all_rows.append(
                {
                    "sim_key": sim_used,
                    "model": model_name,
                    "window": args.window,
                    "horizon": args.horizon,
                    "max_delay": d,
                    "real_windows": n,
                    "cal_windows": n_cal,
                    "eval_windows": int(x_eval.shape[0]),
                    "pdp_corr": simm["pdp_corr"],
                    "pdp_nmse": simm["pdp_nmse"],
                    "energy_corr": simm["energy_corr"],
                    "energy_nmse": simm["energy_nmse"],
                    "mse": mm["mse"],
                    "mae": mm["mae"],
                    "nmse": mm["nmse"],
                    "nmse_improve_vs_persistence_pct": (
                        0.0
                        if model_name == "Persistence"
                        else 100.0 * (1.0 - mm["nmse"] / (m_persist["nmse"] + 1e-12))
                    ),
                }
            )

    if args.out_csv:
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        cols = list(all_rows[0].keys()) if all_rows else []
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            writer.writerows(all_rows)
        print()
        print(f"[save] {out_csv.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sim(noise/no-noise) vs real similarity + AI/traditional benchmark")
    p.add_argument("--sim-path", type=str, default="/home/mh/kmh/sionna-rt/jinsup/cir_pdp_exports")
    p.add_argument("--real-path", type=str, default="")
    p.add_argument("--real-key", type=str, default="CIR")
    p.add_argument("--sim-keys", type=str, default="h_cir_clean,h_cir_noisy")
    p.add_argument("--sim-layout", choices=("auto", "time_delay", "delay_time"), default="auto")
    p.add_argument("--real-layout", choices=("auto", "time_delay", "delay_time"), default="auto")

    p.add_argument("--window", type=int, default=8)
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--max-delay", type=int, default=128)
    p.add_argument("--calib-ratio", type=float, default=0.25)

    p.add_argument("--ridge-alpha", type=float, default=1.0)
    p.add_argument("--mlp-hidden", type=int, default=256)
    p.add_argument("--mlp-epochs", type=int, default=120)
    p.add_argument("--mlp-fine-tune-epochs", type=int, default=40)
    p.add_argument("--mlp-batch-size", type=int, default=32)
    p.add_argument("--mlp-lr", type=float, default=1e-3)
    p.add_argument("--mlp-input-clip", type=float, default=8.0)
    p.add_argument("--seed", type=int, default=1234)

    p.add_argument("--out-csv", type=str, default="/home/mh/kmh/sionna-rt/jinsup/cir_pdp_exports/ai_vs_traditional_results.csv")
    return p


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
