"""
M1 (3D): Extract bf_grid CSV + manifest JSON from the dense 3D grid.

Input:
  Artur_code/joint_both_nueLowE_noreactor_solar_systs_realdata/
      merged_ninja_grid_output_forweb.csv

Output:
  data/full/bf_grid.csv
  data/full/manifest.json

Grid: 11 (lambda) × 61 (dm32) × 76 (precision) = 50,996 points.
"""

import csv
import json
import math
import os

import numpy as np
from scipy.stats import norm


SIGMA_CLAMP = 8.0

SRC_CSV = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", "Artur_code",
    "joint_both_nueLowE_noreactor_solar_systs_realdata",
    "merged_ninja_grid_output_forweb.csv"))

OUT_DIR = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", "data", "full"))


def bf_to_sigma(bf, two_sided=True):
    bf = np.asarray(bf, dtype=float)
    p_h1 = bf / (1.0 + bf)
    p = 1.0 - p_h1
    if two_sided:
        return norm.isf(p / 2.0)
    return norm.isf(p)


def _exact_index(value, axis_min, step, n):
    idx = int(round((value - axis_min) / step))
    if idx < 0 or idx >= n:
        raise ValueError(f"value {value} out of axis range "
                         f"[{axis_min}, {axis_min + (n-1)*step}]")
    return idx


def main():
    print(f"Reading {SRC_CSV} ...")
    rows = []
    with open(SRC_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "io_no_offset": float(r["io_no_offset"]),
                "juno_no":      float(r["juno_no"]),
                "pct_precision": float(r["pct_precision"]),
                "bf":           float(r["bayes_factor"]),
                "rmse":         float(r["rmse"]),
            })
    n_rows = len(rows)
    print(f"  {n_rows} rows")

    # Derive sorted unique axis arrays.
    lambda_set = sorted({r["io_no_offset"] for r in rows})
    dm32_set   = sorted({r["juno_no"]      for r in rows})
    prec_set   = sorted({r["pct_precision"] for r in rows})

    n_l, n_i, n_j = len(lambda_set), len(dm32_set), len(prec_set)
    print(f"  axis sizes: lambda={n_l}, dm32={n_i}, precision={n_j}")
    assert n_l * n_i * n_j == n_rows, \
        f"Grid not dense: {n_l}*{n_i}*{n_j}={n_l*n_i*n_j} != {n_rows}"

    # Uniform-step assumption.
    lam_step  = (lambda_set[-1] - lambda_set[0]) / (n_l - 1)
    dm32_step = (dm32_set[-1]   - dm32_set[0])   / (n_i - 1)
    prec_step = (prec_set[-1]   - prec_set[0])   / (n_j - 1)

    # Sanity-check uniformity (looser tol for round-trip float noise).
    def _check_uniform(vals, step, name):
        for k in range(1, len(vals)):
            if not math.isclose(vals[k] - vals[k-1], step,
                                rel_tol=0, abs_tol=step * 1e-6):
                raise ValueError(f"{name} not uniform near index {k}: "
                                 f"{vals[k-1]} -> {vals[k]} (step {step})")
    _check_uniform(lambda_set, lam_step,  "lambda")
    _check_uniform(dm32_set,   dm32_step, "dm32")
    _check_uniform(prec_set,   prec_step, "precision")

    os.makedirs(OUT_DIR, exist_ok=True)

    csv_path      = os.path.join(OUT_DIR, "bf_grid.csv")
    manifest_path = os.path.join(OUT_DIR, "manifest.json")

    bfs_arr   = np.array([r["bf"]   for r in rows], dtype=float)
    rmses_arr = np.array([r["rmse"] for r in rows], dtype=float)

    sigmas = bf_to_sigma(np.abs(bfs_arr)) * np.sign(bfs_arr)
    sigmas = np.clip(sigmas, -SIGMA_CLAMP, SIGMA_CLAMP)
    log10_bf = np.log10(np.abs(bfs_arr)) * np.sign(bfs_arr)
    rmse_sigmas = bf_to_sigma(rmses_arr)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac_rmse  = np.abs(rmses_arr / bfs_arr)
        frac_sigma = np.abs(rmse_sigmas / sigmas)

    manifest = []
    with open(csv_path, "w", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["l", "i", "j", "lambda", "dm32_no", "precision",
                    "bf", "log10_bf", "sigma", "rmse",
                    "rmse_sigma", "frac_rmse", "frac_sigma"])
        for idx, r in enumerate(rows):
            l = _exact_index(r["io_no_offset"],  lambda_set[0], lam_step,  n_l)
            i = _exact_index(r["juno_no"],       dm32_set[0],   dm32_step, n_i)
            j = _exact_index(r["pct_precision"], prec_set[0],   prec_step, n_j)

            lam  = lambda_set[l]
            dm32 = dm32_set[i]
            prec = prec_set[j]

            bf  = float(bfs_arr[idx])
            lb  = float(log10_bf[idx]) if math.isfinite(log10_bf[idx]) else 0.0
            sg  = float(sigmas[idx])
            rm  = float(rmses_arr[idx])
            rsg = float(rmse_sigmas[idx]) if math.isfinite(rmse_sigmas[idx]) else 0.0
            fr  = float(frac_rmse[idx])   if math.isfinite(frac_rmse[idx])   else 0.0
            fs  = float(frac_sigma[idx])  if math.isfinite(frac_sigma[idx])  else 0.0

            w.writerow([l, i, j,
                        f"{lam:.6f}", f"{dm32:.6f}", f"{prec:.6f}",
                        f"{bf:.6f}", f"{lb:.6f}", f"{sg:.6f}",
                        f"{rm:.6f}", f"{rsg:.6f}", f"{fr:.6f}", f"{fs:.6f}"])

            png = (f"triangles/v9/triangle_i{i:02d}_j{j:02d}_l{l:02d}.jpg")
            manifest.append({
                "l": l, "i": i, "j": j,
                "lambda":   round(lam,  6),
                "dm32_no":  round(dm32, 6),
                "precision": round(prec, 6),
                "bf":       round(bf,   6),
                "log10_bf": round(lb,   6),
                "sigma":    round(sg,   6),
                "png": png,
            })

    print(f"Wrote {csv_path}")

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {manifest_path}  ({len(manifest)} entries)")

    # Summary
    print()
    print("=== Summary ===")
    print(f"Grid: {n_l} × {n_i} × {n_j}  ({n_l*n_i*n_j} points)")
    print(f"lambda    : {lambda_set[0]:.4f} – {lambda_set[-1]:.4f}  step {lam_step:.4f}")
    print(f"dm32_no   : {dm32_set[0]:.4f} – {dm32_set[-1]:.4f}  step {dm32_step:.4f}")
    print(f"precision : {prec_set[0]:.4f} – {prec_set[-1]:.4f}  step {prec_step:.4f}")
    print(f"BF        : {bfs_arr.min():.3g} – {bfs_arr.max():.3g}")
    finite_lb = log10_bf[np.isfinite(log10_bf)]
    print(f"log10_bf  : {finite_lb.min():.3f} – {finite_lb.max():.3f}")
    print(f"sigma     : {sigmas.min():.3f} – {sigmas.max():.3f}  (clamped to ±{SIGMA_CLAMP})")


if __name__ == "__main__":
    main()
