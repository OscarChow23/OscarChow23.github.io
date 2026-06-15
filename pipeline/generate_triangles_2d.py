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
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
from scipy.stats import norm
from tqdm import tqdm
from matplotlib.ticker import MaxNLocator


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.linewidth"] = 2
plt.rcParams["xtick.major.size"] = 5
plt.rcParams["xtick.major.width"] = 1
plt.rcParams["xtick.minor.size"] = 3
plt.rcParams["xtick.minor.width"] = 1
plt.rcParams["ytick.major.size"] = 5
plt.rcParams["ytick.major.width"] = 1
plt.rcParams["ytick.minor.size"] = 3
plt.rcParams["ytick.minor.width"] = 1

COLOR_LINE_NO = "#006DFF"
COLOR_LINE_IO = "#CC2020"
# index 0 -> 90% (outer, lighter), index 1 -> 68% (inner, darker)
COLOR_FILL_NO = ["#1CA4F7", "#65cbfe"][::-1]
COLOR_FILL_IO = ["#ED4040", "#fe7f7e"][::-1]

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
    os.path.dirname(__file__), "..", "triangles", "v8"))


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


def _contour_levels(thr):
    """Return strictly increasing contour levels (matplotlib requirement)."""
    thr = np.sort(np.atleast_1d(np.asarray(thr, dtype=float)))
    out = []
    for v in thr:
        if out and v <= out[-1]:
            v = out[-1] + 1e-12
        out.append(v)
    return out


def _add_legend(ax):
    handle_no_68 = (Line2D([], [], color=COLOR_LINE_NO, lw=2, ls="-"),
                    mpatches.Patch(color=COLOR_FILL_NO[1], alpha=0.7, lw=2, ec=COLOR_LINE_NO))
    handle_no_90 = (Line2D([], [], color=COLOR_LINE_NO, lw=2, ls="--"),
                    mpatches.Patch(color=COLOR_FILL_NO[0], alpha=0.5, lw=2, ec=COLOR_LINE_NO))
    handle_io_68 = (Line2D([], [], color=COLOR_LINE_IO, lw=2, ls="-"),
                    mpatches.Patch(color=COLOR_FILL_IO[1], alpha=0.7, lw=2, ec=COLOR_LINE_IO))
    handle_io_90 = (Line2D([], [], color=COLOR_LINE_IO, lw=2, ls="--"),
                    mpatches.Patch(color=COLOR_FILL_IO[0], alpha=0.5, lw=2, ec=COLOR_LINE_IO))

    ax.legend(
        [handle_no_68, handle_no_90, handle_io_68, handle_io_90],
        ["Normal MO 68%", "Normal MO 90%", "Inverted MO 68%", "Inverted MO 90%"],
        loc="center", fontsize=9.5, handleheight=2.5, handlelength=4.1,
        frameon=False, labelspacing=1.5,
        handler_map={tuple: HandlerTuple(ndivide=None)},
    )


def draw_triangle(data, i_idx, j_idx, out_path, fmt="jpeg", dpi=80):
    postfix = f"i{i_idx}_j{j_idx}"
    n = len(PARAMETERS)

    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(n, n, wspace=0.0, hspace=0.0)
    axes = gs.subplots(sharex="col")

    for i, pari in enumerate(PARAMETERS):
        for j, parj in enumerate(PARAMETERS):
            ax = axes[i, j]

            if i < j:
                ax.set_axis_off()
                continue

            if i == j:
                histno = data[f"plot_{pari}_hist_no_{postfix}"]
                histio = data[f"plot_{pari}_hist_io_{postfix}"]
                edges = data[f"plot_{pari}_edges_{postfix}"][0]

                hist_joint = np.concatenate([histno, histio])
                thr = get_quantile_threshold(hist_joint, quantile=[0.90, 0.68])

                mask_no = histno[:, np.newaxis] >= thr
                mask_io = histio[:, np.newaxis] >= thr

                ax.stairs(histno, edges, color=COLOR_LINE_NO, lw=1.5)
                for k in range(len(thr)):
                    ax.stairs(histno * mask_no[:, k], edges, color=COLOR_FILL_NO[k],
                              lw=1.5, alpha=0.5, fill=True, ec=COLOR_LINE_NO)
                ax.stairs(histio, edges, color=COLOR_LINE_IO, lw=1.5)
                for k in range(len(thr)):
                    ax.stairs(histio * mask_io[:, k], edges, color=COLOR_FILL_IO[k],
                              lw=1.5, alpha=0.5, fill=True, ec=COLOR_LINE_IO)

                if j == 0:
                    ax.set_ylabel("Posterior Density", fontsize=AXIS_LABEL_FONT_SIZE)
                if i == n - 1:
                    ax.set_xlabel(PARAMETERS_LATEX[pari], fontsize=AXIS_LABEL_FONT_SIZE)

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
                thr = _contour_levels(
                    get_quantile_threshold(hist_joint, quantile=[0.90, 0.68]))

                ax.contour(X, Y, histno, levels=thr, colors=COLOR_LINE_NO,
                           linewidths=1.5, linestyles=["--", "-"])
                ax.contour(X, Y, histio, levels=thr, colors=COLOR_LINE_IO,
                           linewidths=1.5, linestyles=["--", "-"])

                if j == 0:
                    ax.set_ylabel(PARAMETERS_LATEX[pari], fontsize=AXIS_LABEL_FONT_SIZE)
                if i == n - 1:
                    ax.set_xlabel(PARAMETERS_LATEX[parj], fontsize=AXIS_LABEL_FONT_SIZE)

                ax.set_xlim(xedges[0], xedges[-1])
                ax.set_ylim(yedges[0], yedges[-1])

            ax.xaxis.set_major_locator(MaxNLocator(nbins=3, prune="both"))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3, prune="both"))
            ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_FONT_SIZE)

    # NO/IO legend in the empty upper-right region
    _add_legend(axes[1, n - 1])

    for ax in axes.flatten():
        if not ax.get_visible():
            continue
        ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
        ax.minorticks_on()
        ax.label_outer()

    save_kwargs = {"dpi": dpi, "format": fmt, "bbox_inches": "tight"}
    if fmt == "jpeg":
        save_kwargs["pil_kwargs"] = {"quality": 85}
    fig.savefig(out_path, **save_kwargs)
    plt.close(fig)


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

    total = n_dm32 * n_prec
    with tqdm(total=total, desc="Triangles") as pbar:
        for i in range(n_dm32):
            for j in range(n_prec):
                out_path = os.path.join(
                    TRIANGLES_DIR, f"triangle_i{i:02d}_j{j:02d}.jpg")
                if os.path.exists(out_path):
                    pbar.update(1)
                    continue
                draw_triangle(data, i, j, out_path)
                pbar.update(1)

    print(f"Done. JPEGs in {TRIANGLES_DIR}")


if __name__ == "__main__":
    main()
