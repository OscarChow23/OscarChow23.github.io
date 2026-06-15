"""
M2 (3D): Render triangle JPEGs for every grid point in the new dense npz.

Input:
  Artur_code/joint_both_nueLowE_noreactor_solar_systs_realdata/
      merged_ninja_grid_output_forweb.npz
  Axis arrays come from the companion CSV (avoids loading the 13 GB npz key list).

Output:
  triangles/v9/triangle_i{i:02d}_j{j:02d}_l{l:02d}.jpg

CLI flags:
  --lambda-index N   render only the slice l == N (0..n_l-1)
  --only I,J         render only (i, j) at every λ (or together with --lambda-index)
  --out-dir PATH     override default triangles/v9/
  --npz PATH         override default npz path
  --csv PATH         override default companion csv path

Skips existing JPEGs to allow interrupted reruns.
"""

import argparse
import csv
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm


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

DEFAULT_NPZ = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", "Artur_code",
    "joint_both_nueLowE_noreactor_solar_systs_realdata",
    "merged_ninja_grid_output_forweb.npz"))
DEFAULT_CSV = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", "Artur_code",
    "joint_both_nueLowE_noreactor_solar_systs_realdata",
    "merged_ninja_grid_output_forweb.csv"))
DEFAULT_OUT = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", "triangles", "v9"))


def get_quantile_threshold(hist, quantile=0.6827):
    hist = np.asarray(hist)
    flat = hist.ravel()
    sorted_hist = np.sort(flat)[::-1]
    cumsum = np.cumsum(sorted_hist)
    cumsum /= cumsum[-1]
    idx = np.searchsorted(cumsum, quantile)
    return sorted_hist[idx]


def _contour_levels(thr):
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


def _grid_key_prefix(io_offset, precision, dm32):
    # The npz keys were written using Python's repr() of the raw float values
    # from the source CSV. repr(float(x)) round-trips bit-exactly even for
    # round-9 artefacts like 2.2549999999999. Do NOT format these as fixed dp.
    return (f"noio{repr(float(io_offset))}"
            f"_precision{repr(float(precision))}"
            f"_dm32no{repr(float(dm32))}_plot_")


def draw_triangle(npz, prefix, out_path, fmt="jpeg", dpi=80):
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
                histno = npz[f"{prefix}{pari}_hist_no"]
                histio = npz[f"{prefix}{pari}_hist_io"]
                edges  = npz[f"{prefix}{pari}_edges"][0]

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
                histno = npz[f"{prefix}{pari}_{parj}_hist_no"]
                histio = npz[f"{prefix}{pari}_{parj}_hist_io"]
                edges  = npz[f"{prefix}{pari}_{parj}_edges"]
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


def _load_axes(csv_path):
    """Return (lambda_list, dm32_list, prec_list) as sorted unique floats from the CSV."""
    lam, dm, pr = set(), set(), set()
    with open(csv_path, newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            lam.add(float(row["io_no_offset"]))
            dm.add(float(row["juno_no"]))
            pr.add(float(row["pct_precision"]))
    return sorted(lam), sorted(dm), sorted(pr)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lambda-index", type=int, default=None,
                   help="Render only the slice l == N (0-indexed).")
    p.add_argument("--only", type=str, default=None,
                   help="Render only point i,j (e.g. '0,0'). Combined with "
                        "--lambda-index renders a single (i,j,l) triangle.")
    p.add_argument("--out-dir", type=str, default=DEFAULT_OUT)
    p.add_argument("--npz", type=str, default=DEFAULT_NPZ)
    p.add_argument("--csv", type=str, default=DEFAULT_CSV)
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Axes from {args.csv} ...")
    lambdas, dm32s, precs = _load_axes(args.csv)
    n_l, n_i, n_j = len(lambdas), len(dm32s), len(precs)
    print(f"  grid {n_l} × {n_i} × {n_j}")

    only_ij = None
    if args.only is not None:
        parts = args.only.split(",")
        only_ij = (int(parts[0]), int(parts[1]))

    if args.lambda_index is not None:
        if not (0 <= args.lambda_index < n_l):
            raise SystemExit(f"--lambda-index out of range [0, {n_l-1}]")
        l_range = [args.lambda_index]
    else:
        l_range = list(range(n_l))

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Opening {args.npz} (mmap) ...")
    npz = np.load(args.npz, mmap_mode="r")

    total_tasks = []
    for l in l_range:
        for i in range(n_i):
            for j in range(n_j):
                if only_ij is not None and (i, j) != only_ij:
                    continue
                total_tasks.append((l, i, j))

    t0 = time.time()
    rendered = 0
    skipped = 0
    with tqdm(total=len(total_tasks), desc="Triangles") as pbar:
        for (l, i, j) in total_tasks:
            out_path = os.path.join(
                args.out_dir,
                f"triangle_i{i:02d}_j{j:02d}_l{l:02d}.jpg")
            if os.path.exists(out_path):
                skipped += 1
                pbar.update(1)
                continue
            prefix = _grid_key_prefix(lambdas[l], precs[j], dm32s[i])
            draw_triangle(npz, prefix, out_path)
            rendered += 1
            pbar.update(1)

    dt = time.time() - t0
    print(f"Done. Rendered {rendered}, skipped {skipped}. Took {dt:.1f}s "
          f"({dt / max(rendered, 1):.2f}s/triangle when rendering).")
    print(f"Output: {args.out_dir}")


if __name__ == "__main__":
    main()
