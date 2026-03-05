#!/usr/bin/env python3
"""Train on simulation CIR and predict real CIR with calibration.

Supported inputs
- Simulation: directory with `cir_frame_XXXX.npz` exported by `extract_cir_data.py`
- Real: `.mat` / `.npy` / `.npz` / same `cir_frame_*.npz` directory

This script uses only NumPy for modeling (closed-form multi-output Ridge).
If `.mat` is used, SciPy is required only for loading the file.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


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
            raise RuntimeError("Standardizer not fitted.")
        return (x - self.mean) / self.std

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Standardizer not fitted.")
        return x * self.std + self.mean


class MultiOutputRidge:
    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)
        self.w: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "MultiOutputRidge":
        if x.ndim != 2 or y.ndim != 2:
            raise ValueError("x and y must be 2D arrays.")
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y sample counts must match.")

        xb = np.concatenate([x, np.ones((x.shape[0], 1), dtype=x.dtype)], axis=1)
        xtx = xb.T @ xb
        reg = self.alpha * np.eye(xtx.shape[0], dtype=x.dtype)
        reg[-1, -1] = 0.0  # do not regularize bias
        rhs = xb.T @ y
        self.w = np.linalg.solve(xtx + reg, rhs)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise RuntimeError("Model not fitted.")
        xb = np.concatenate([x, np.ones((x.shape[0], 1), dtype=x.dtype)], axis=1)
        return xb @ self.w


def _layout_to_time_delay(arr: np.ndarray, layout: str) -> np.ndarray:
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D CIR array, got shape {arr.shape}.")

    if layout == "time_delay":
        return arr
    if layout == "delay_time":
        return arr.T

    # auto: usually T << D for CIR (e.g., 281 x 4095)
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
    raise KeyError(f"Could not select key from npz. available={npz.files}")


def _load_from_frame_dir(frame_dir: Path, key: str | None) -> tuple[np.ndarray, str]:
    files = sorted(frame_dir.glob("cir_frame_*.npz"))
    if not files:
        raise FileNotFoundError(f"No cir_frame_*.npz found in {frame_dir}")

    seq: list[np.ndarray] = []
    used_key = ""
    for p in files:
        with np.load(p) as data:
            k = _select_key(
                data,
                key,
                ("h_cir_noisy", "h_cir_clean", "pdp_noisy", "pdp_clean"),
            )
            if not used_key:
                used_key = k
            seq.append(np.asarray(data[k]).reshape(-1))

    cir = np.stack(seq, axis=0)
    # If power domain key is used, keep it as real-valued complex for shared pipeline.
    if not np.iscomplexobj(cir):
        cir = cir.astype(np.float64) + 0j
    return cir, used_key


def _load_from_file(path: Path, key: str | None, layout: str) -> tuple[np.ndarray, str]:
    ext = path.suffix.lower()
    if ext == ".npy":
        arr = np.squeeze(np.load(path))
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        cir = _layout_to_time_delay(np.asarray(arr), layout)
        if not np.iscomplexobj(cir):
            cir = cir.astype(np.float64) + 0j
        return cir, "npy_array"

    if ext == ".npz":
        with np.load(path) as data:
            selected = _select_key(data, key, ("CIR", "cir", "h_cir_noisy", "h_cir_clean"))
            arr = np.squeeze(np.asarray(data[selected]))
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        cir = _layout_to_time_delay(arr, layout)
        if not np.iscomplexobj(cir):
            cir = cir.astype(np.float64) + 0j
        return cir, selected

    if ext == ".mat":
        try:
            from scipy.io import loadmat  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "SciPy is required to load .mat files. "
                "Install it (`pip install scipy`) or convert .mat to .npy/.npz."
            ) from exc
        mat = loadmat(path)
        if key is not None:
            if key not in mat:
                raise KeyError(f"Key '{key}' not found in mat file. available={list(mat.keys())}")
            selected = key
        else:
            selected = "CIR" if "CIR" in mat else next(
                (k for k, v in mat.items() if not k.startswith("__") and np.ndim(v) >= 2),
                "",
            )
            if not selected:
                raise KeyError(f"No 2D array key found in {path}")
        arr = np.squeeze(np.asarray(mat[selected]))
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        cir = _layout_to_time_delay(arr, layout)
        if not np.iscomplexobj(cir):
            cir = cir.astype(np.float64) + 0j
        return cir, selected

    raise ValueError(f"Unsupported file extension: {ext}")


def load_cir(path: Path, key: str | None, layout: str) -> tuple[np.ndarray, str]:
    if path.is_dir():
        return _load_from_frame_dir(path, key)
    if path.is_file():
        return _load_from_file(path, key, layout)
    raise FileNotFoundError(path)


def complex_to_real_features(c: np.ndarray) -> np.ndarray:
    return np.concatenate([c.real, c.imag], axis=1).astype(np.float64)


def real_features_to_complex(x: np.ndarray) -> np.ndarray:
    d2 = x.shape[1]
    if d2 % 2 != 0:
        raise ValueError("Expected even feature size for real/imag split.")
    d = d2 // 2
    return x[:, :d] + 1j * x[:, d:]


def make_windows(seq: np.ndarray, window: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    t = seq.shape[0]
    needed = window + horizon
    if t < needed:
        raise ValueError(f"Not enough frames: got {t}, need at least {needed}.")

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for end in range(window, t - horizon + 1):
        xs.append(seq[end - window : end].reshape(-1))
        ys.append(seq[end + horizon - 1])
    return np.asarray(xs), np.asarray(ys)


def eval_complex(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    abs_err = np.abs(err)
    return {
        "mse": float(np.mean(abs_err**2)),
        "mae": float(np.mean(abs_err)),
        "nmse": float(np.mean(abs_err**2) / np.mean(np.abs(y_true) ** 2 + 1e-12)),
    }


def print_metrics(title: str, m: dict[str, float]) -> None:
    print(f"{title}: MSE={m['mse']:.6e}, MAE={m['mae']:.6e}, NMSE={m['nmse']:.6e}")


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


def run(args: argparse.Namespace) -> None:
    sim_cir, sim_key_used = load_cir(Path(args.sim_path), args.sim_key, args.sim_layout)
    print(f"[sim] loaded shape={sim_cir.shape}, key={sim_key_used}")

    real_path = _resolve_real_path(args.real_path)
    real_cir: np.ndarray | None = None
    real_key_used = ""
    if real_path:
        real_cir, real_key_used = load_cir(Path(real_path), args.real_key, args.real_layout)
        print(f"[real] loaded shape={real_cir.shape}, key={real_key_used}")

    max_delay = int(args.max_delay)
    if max_delay <= 0:
        raise ValueError("--max-delay must be > 0")

    sim_d = min(sim_cir.shape[1], max_delay)
    if real_cir is not None:
        sim_d = min(sim_d, real_cir.shape[1])

    sim_cir = sim_cir[:, :sim_d]
    if real_cir is not None:
        real_cir = real_cir[:, :sim_d]
    print(f"[common] using delay taps={sim_d}")

    sim_seq = complex_to_real_features(sim_cir)
    x_sim, y_sim = make_windows(sim_seq, args.window, args.horizon)

    x_scaler = Standardizer().fit(x_sim)
    y_scaler = Standardizer().fit(y_sim)
    x_sim_n = x_scaler.transform(x_sim)
    y_sim_n = y_scaler.transform(y_sim)

    base_model = MultiOutputRidge(alpha=args.alpha).fit(x_sim_n, y_sim_n)
    print(
        f"[base] trained with samples={x_sim.shape[0]}, "
        f"in_dim={x_sim.shape[1]}, out_dim={y_sim.shape[1]}"
    )

    if real_cir is None:
        print("No --real-path supplied. Training on simulation only is complete.")
        return

    real_seq = complex_to_real_features(real_cir)
    x_real, y_real = make_windows(real_seq, args.window, args.horizon)
    n = x_real.shape[0]
    n_cal = int(round(n * args.calib_ratio))
    n_cal = max(1, min(n - 1, n_cal))

    x_cal, y_cal = x_real[:n_cal], y_real[:n_cal]
    x_eval, y_eval = x_real[n_cal:], y_real[n_cal:]

    # Base prediction
    y_base_cal = y_scaler.inverse_transform(base_model.predict(x_scaler.transform(x_cal)))
    y_base_eval = y_scaler.inverse_transform(base_model.predict(x_scaler.transform(x_eval)))

    # Residual calibration model: learn (y_real - y_base) on real calibration split.
    residual = y_cal - y_base_cal
    rx_scaler = Standardizer().fit(x_cal)
    rr_scaler = Standardizer().fit(residual)
    residual_model = MultiOutputRidge(alpha=args.alpha).fit(
        rx_scaler.transform(x_cal), rr_scaler.transform(residual)
    )
    y_res_eval = rr_scaler.inverse_transform(residual_model.predict(rx_scaler.transform(x_eval)))
    y_pred_eval = y_base_eval + y_res_eval

    # Persistence baseline: predict next as last frame in window.
    f = y_real.shape[1]
    y_persist_eval = x_eval.reshape(-1, args.window, f)[:, -1, :]

    y_true_c = real_features_to_complex(y_eval)
    y_base_c = real_features_to_complex(y_base_eval)
    y_pred_c = real_features_to_complex(y_pred_eval)
    y_persist_c = real_features_to_complex(y_persist_eval)

    m_base = eval_complex(y_true_c, y_base_c)
    m_pred = eval_complex(y_true_c, y_pred_c)
    m_pers = eval_complex(y_true_c, y_persist_c)

    print(
        f"[split] real windows={n}, calibration={n_cal}, eval={n - n_cal}, "
        f"window={args.window}, horizon={args.horizon}"
    )
    print_metrics("Base(sim-only)", m_base)
    print_metrics("Sim2Real(calibrated)", m_pred)
    print_metrics("Persistence", m_pers)

    if args.out_npz:
        out_path = Path(args.out_npz)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path,
            y_true=y_true_c,
            y_base=y_base_c,
            y_pred=y_pred_c,
            y_persistence=y_persist_c,
            metrics_base=np.array([m_base["mse"], m_base["mae"], m_base["nmse"]], dtype=np.float64),
            metrics_pred=np.array([m_pred["mse"], m_pred["mae"], m_pred["nmse"]], dtype=np.float64),
            metrics_persistence=np.array([m_pers["mse"], m_pers["mae"], m_pers["nmse"]], dtype=np.float64),
            sim_key=np.array([sim_key_used]),
            real_key=np.array([real_key_used]),
            delay_taps=np.int32(sim_d),
            window=np.int32(args.window),
            horizon=np.int32(args.horizon),
            calib_ratio=np.float64(args.calib_ratio),
            alpha=np.float64(args.alpha),
        )
        print(f"[save] {out_path.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Predict real channel from simulation channel sequence.")
    p.add_argument(
        "--sim-path",
        type=str,
        default="/home/mh/kmh/sionna-rt/jinsup/cir_pdp_exports",
        help="Simulation CIR source: frame directory, .npy, .npz, or .mat",
    )
    p.add_argument(
        "--real-path",
        type=str,
        default="",
        help="Real CIR source: frame directory, .npy, .npz, or .mat",
    )
    p.add_argument("--sim-key", type=str, default="h_cir_noisy", help="Preferred key for simulation input.")
    p.add_argument("--real-key", type=str, default="CIR", help="Preferred key for real input.")
    p.add_argument(
        "--sim-layout",
        choices=("auto", "time_delay", "delay_time"),
        default="auto",
        help="Layout of loaded 2D simulation array.",
    )
    p.add_argument(
        "--real-layout",
        choices=("auto", "time_delay", "delay_time"),
        default="auto",
        help="Layout of loaded 2D real array.",
    )
    p.add_argument("--window", type=int, default=8, help="Input window length (frames).")
    p.add_argument("--horizon", type=int, default=1, help="Prediction horizon in frames.")
    p.add_argument("--max-delay", type=int, default=256, help="Use first N delay taps for training.")
    p.add_argument("--alpha", type=float, default=1.0, help="Ridge regularization.")
    p.add_argument("--calib-ratio", type=float, default=0.25, help="Fraction of real windows for calibration.")
    p.add_argument("--out-npz", type=str, default="", help="Optional output .npz path for predictions and metrics.")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
