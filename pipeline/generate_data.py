"""
M1: Extract CSVs and manifest from npz.
Input:  Artur_code/bayes_factors_and_rmse_prel.npz
Output: data/bf_grid.csv, data/manifest.json
"""

import json
import os
import sys

import numpy as np
from scipy.stats import norm


def bf_to_sigma(bf, two_sided=True):
    bf = np.asarray(bf, dtype=float)
    p_h1 = bf / (1.0 + bf)
    p = 1.0 - p_h1
    if two_sided:
        sigma = norm.isf(p / 2.0)
    else:
        sigma = norm.isf(p)
    return sigma


NPZ_PATH = os.path.join(os.path.dirname(__file__), "..", "Artur_code", "bayes_factors_and_rmse_prel.npz")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def main():
    npz_path = os.path.normpath(NPZ_PATH)
    data_dir = os.path.normpath(DATA_DIR)

    print(f"Loading {npz_path} ...")
    d = np.load(npz_path)

    bfs = d["bfs"]                         # (32, 32)
    rmses = d["rmses"]                     # (32, 32)
    dm32_values = d["dm32_values"] * 1e3  # convert eV² → 10⁻³ eV²
    precision_values = d["precision_values"]

    n_dm32, n_prec = bfs.shape

    # Derived quantities
    sigmas = bf_to_sigma(np.abs(bfs)) * np.sign(bfs)
    log10_bf = np.log10(np.abs(bfs)) * np.sign(bfs)
    rmse_sigmas = bf_to_sigma(rmses)
    frac_rmse = np.abs(rmses / bfs)
    frac_sigma = np.abs(rmse_sigmas / sigmas)

    os.makedirs(data_dir, exist_ok=True)

    # --- bf_grid.csv ---
    csv_path = os.path.join(data_dir, "bf_grid.csv")
    with open(csv_path, "w") as f:
        f.write("i,j,dm32_no,precision,bf,log10_bf,sigma,rmse,rmse_sigma,frac_rmse,frac_sigma\n")
        for i in range(n_dm32):
            for j in range(n_prec):
                f.write(
                    f"{i},{j},"
                    f"{dm32_values[i]:.6f},"
                    f"{precision_values[j]:.6f},"
                    f"{bfs[i, j]:.6f},"
                    f"{log10_bf[i, j]:.6f},"
                    f"{sigmas[i, j]:.6f},"
                    f"{rmses[i, j]:.6f},"
                    f"{rmse_sigmas[i, j]:.6f},"
                    f"{frac_rmse[i, j]:.6f},"
                    f"{frac_sigma[i, j]:.6f}\n"
                )
    print(f"Wrote {csv_path}")

    # --- manifest.json ---
    manifest = []
    for i in range(n_dm32):
        for j in range(n_prec):
            pad = "02d" if max(n_dm32, n_prec) <= 99 else "03d"
            png = f"triangles/triangle_i{i:{pad}}_j{j:{pad}}.png"
            manifest.append({
                "i": int(i),
                "j": int(j),
                "dm32_no": round(float(dm32_values[i]), 6),
                "precision": round(float(precision_values[j]), 6),
                "bf": round(float(bfs[i, j]), 6),
                "log10_bf": round(float(log10_bf[i, j]), 6),
                "sigma": round(float(sigmas[i, j]), 6),
                "png": png,
            })

    manifest_path = os.path.join(data_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {manifest_path}  ({len(manifest)} entries)")

    # --- Summary ---
    print()
    print("=== Summary ===")
    print(f"Grid shape:      {n_dm32} × {n_prec}  ({n_dm32 * n_prec} points)")
    print(f"dm32_no range:   {dm32_values[0]:.4f} – {dm32_values[-1]:.4f}  [×10⁻³ eV²]")
    print(f"precision range: {precision_values[0]:.4f} – {precision_values[-1]:.4f}  [%]")
    print(f"BF range:        {bfs.min():.3g} – {bfs.max():.3g}")
    print(f"log10_bf range:  {log10_bf.min():.3f} – {log10_bf.max():.3f}")
    print(f"sigma range:     {sigmas.min():.3f} – {sigmas.max():.3f}")
    print(f"frac_rmse range: {frac_rmse.min():.4f} – {frac_rmse.max():.4f}")


if __name__ == "__main__":
    main()
