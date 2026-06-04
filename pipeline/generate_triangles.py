"""
M2: Render triangle JPEGs for every grid point in the npz.

Input:  Artur_code/bayes_factors_and_rmse_prel.npz
Output: triangles/triangle_i{i:02d}_j{j:02d}.jpg

Matches the physics and rendering of draw_triangle() in Artur_code/plotting.py.
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm import tqdm
from matplotlib.ticker import MaxNLocator


PARAMETERS = ["delta_in_pi", "ss2th13", "ssth23", "absdm32"]
PARAMETERS_LATEX = {
    "delta_in_pi": r"$\delta_{CP}$ (in $\pi$)",
    "ss2th13":     r"$\sin^2 2\theta_{13}$",
    "ssth23":      r"$\sin^2\theta_{23}$",
    "absdm32":     r"$|\Delta m^2_{32}|$ [$\times 10^{-3}$ eV$^2$]",
}

AXIS_LABEL_FONT_SIZE = 13
TICK_LABEL_FONT_SIZE = 11

NPZ_PATH = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", "Artur_code", "bayes_factors_and_rmse_prel.npz"))
TRIANGLES_DIR = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", "triangles", "v7"))


def get_quantile_threshold(hist, quantile=0.6827):
    hist = np.asarray(hist)
    flat = hist.ravel()
    sorted_hist = np.sort(flat)[::-1]
    cumsum = np.cumsum(sorted_hist)
    cumsum /= cumsum[-1]
    idx = np.searchsorted(cumsum, quantile)
    return sorted_hist[idx]


def bf_to_sigma(bf, two_sided=True):
    bf = np.asarray(bf, dtype=float)
    p_h1 = bf / (1.0 + bf)
    p = 1.0 - p_h1
    if two_sided:
        return norm.isf(p / 2.0)
    return norm.isf(p)


def draw_triangle(data, i_idx, j_idx, fig, panel_axes, out_path):
    postfix = f"i{i_idx}_j{j_idx}"
    n = len(PARAMETERS)

    for i, pari in enumerate(PARAMETERS):
        for j, parj in enumerate(PARAMETERS):
            if i < j:
                continue

            ax = panel_axes[(i, j)]
            ax.cla()

            if i == j:
                histno = data[f"plot_{pari}_hist_no_{postfix}"]
                histio = data[f"plot_{pari}_hist_io_{postfix}"]
                edges = data[f"plot_{pari}_edges_{postfix}"][0]

                hist_joint = np.concatenate([histno, histio])
                thr_joint = get_quantile_threshold(hist_joint, quantile=0.6827)

                mask_no = histno >= thr_joint
                mask_io = histio >= thr_joint

                ax.stairs(histno, edges, color="blue", alpha=0.7, linewidth=1.2)
                ax.stairs(histio, edges, color="red", alpha=0.7, linewidth=1.2)
                ax.stairs(histno * mask_no, edges, fill=True, color="blue", alpha=0.3)
                ax.stairs(histio * mask_io, edges, fill=True, color="red", alpha=0.3)

                if j == 0:
                    ax.set_ylabel("Posterior Density", fontsize=AXIS_LABEL_FONT_SIZE)
                if i == n - 1:
                    ax.set_xlabel(PARAMETERS_LATEX[pari], fontsize=AXIS_LABEL_FONT_SIZE)

                ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
                ax.set_xlim(edges[0], edges[-1])
                ax.set_ylim(0, max(histno.max(), histio.max()) * 1.2)

            else:
                histno = data[f"plot_{pari}_{parj}_hist_no_{postfix}"]
                histio = data[f"plot_{pari}_{parj}_hist_io_{postfix}"]
                edges = data[f"plot_{pari}_{parj}_edges_{postfix}"]
                xedges = edges[1]
                yedges = edges[0]

                xcenters = 0.5 * (xedges[:-1] + xedges[1:])
                ycenters = 0.5 * (yedges[:-1] + yedges[1:])
                X, Y = np.meshgrid(xcenters, ycenters)

                hist_joint = np.concatenate([histno, histio])
                thr_joint = get_quantile_threshold(
                    hist_joint, quantile=[0.9973, 0.9545, 0.6827])

                ax.contour(X, Y, histno, levels=thr_joint,
                           colors="blue", linestyles=["-", "-.", ":"])
                ax.contour(X, Y, histio, levels=thr_joint,
                           colors="red", linestyles=["-", "-.", ":"])

                if j == 0:
                    ax.set_ylabel(PARAMETERS_LATEX[pari], fontsize=AXIS_LABEL_FONT_SIZE)
                if i == n - 1:
                    ax.set_xlabel(PARAMETERS_LATEX[parj], fontsize=AXIS_LABEL_FONT_SIZE)

                ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
                ax.set_xlim(xedges[0], xedges[-1])
                ax.set_ylim(yedges[0], yedges[-1])

            if j > 0:
                ax.set_yticks([])
            else:
                ax.yaxis.set_major_locator(MaxNLocator(nbins=3, prune='both'))
                ax.tick_params(axis="y", labelsize=TICK_LABEL_FONT_SIZE)
            if i < n - 1:
                ax.set_xticks([])
            else:
                ax.xaxis.set_major_locator(MaxNLocator(nbins=3, prune='both'))
                ax.tick_params(axis="x", labelsize=TICK_LABEL_FONT_SIZE)

    fig.savefig(out_path, dpi=80, format='jpeg', bbox_inches='tight', pil_kwargs={'quality': 85})


def main():
    print(f"Loading {NPZ_PATH} ...")
    _npz = np.load(NPZ_PATH)
    print("Pre-loading arrays into memory ...")
    data = {k: _npz[k] for k in _npz.files}

    bfs = data["bfs"]
    rmses = data["rmses"]
    dm32_values = data["dm32_values"] * 1e3
    precision_values = data["precision_values"]

    n_dm32, n_prec = bfs.shape
    os.makedirs(TRIANGLES_DIR, exist_ok=True)

    # Create figure and all panel axes once; reuse across iterations
    n = len(PARAMETERS)
    fig = plt.figure(figsize=(8, 8))
    panel_axes = {}
    for pi in range(n):
        for pj in range(n):
            if pi < pj:
                continue
            panel_axes[(pi, pj)] = fig.add_axes(
                [pj / n, 1 - (pi + 1) / n, 1 / n, 1 / n])

    total = n_dm32 * n_prec
    with tqdm(total=total, desc="Triangles") as pbar:
        for i in range(n_dm32):
            for j in range(n_prec):
                out_path = os.path.join(
                    TRIANGLES_DIR, f"triangle_i{i:02d}_j{j:02d}.jpg")
                if os.path.exists(out_path):
                    pbar.update(1)
                    continue
                draw_triangle(data, i, j, fig, panel_axes, out_path)
                pbar.update(1)

    plt.close(fig)
    print(f"Done. JPEGs in {TRIANGLES_DIR}")


if __name__ == "__main__":
    main()
