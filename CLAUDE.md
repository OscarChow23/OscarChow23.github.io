# NOvA+JUNO Crystal Ball — Interactive WebsiteA
## CLAUDE.md — Read this file before doing anything else

---

## Project Overview

This project builds a static interactive website that reproduces and extends the "crystal ball" plot from the NOvA+JUNO neutrino mass ordering paper (arXiv, May 2026). The site is intended for public release on a NOvA github.io page.

The site shows:
- **Left panel**: the "crystal ball" 2D heatmap — for each possible JUNO measurement outcome (central value × precision), what would the combined NOvA+JUNO Bayes factor for mass ordering be?
- **Right panel**: the corresponding NOvA posterior triangle plot (corner plot) for the oscillation parameters, at the selected JUNO constraint

User interaction is via **sliders** (not hover):
- Slider 1: JUNO Δm²₃₂ central value (NO assumption), in units of 10⁻³ eV²
- Slider 2: JUNO precision on |Δm²₃₂|, in %
- A **"Generate" button** triggers display of the triangle plot for the current slider position (snaps to nearest available grid point)

The crystal ball heatmap updates in real time as sliders move. The triangle plot only updates on button press.

**No backend. Fully static site. No server-side compute.**

---

## Physics Context

NOvA measures neutrino oscillations over an 810 km baseline. It has a ~1.6σ preference for normal mass ordering (NO) over inverted ordering (IO) when combined with Daya Bay reactor constraints.

The reactor complementarity: both reactor experiments (JUNO, Daya Bay) and long-baseline experiments (NOvA) measure |Δm²₃₂|, but the effective Δm² each extracts is a weighted combination of Δm²₃₂ and Δm²₂₁ with weights that differ between NO and IO. This causes the extracted central value to shift by an offset λ ≈ 0.12 × 10⁻³ eV² when the assumed ordering changes. If the correct ordering is assumed, NOvA and reactor values agree; if the wrong ordering is assumed, they disagree.

The degree of agreement/disagreement is quantified as a Bayes factor (BF), the ratio of posterior probabilities for NO vs IO.

The crystal ball plot maps: for each possible JUNO outcome (CV × precision), what would the combined NOvA+JUNO Bayes factor be?

**Key axes:**
- X: JUNO Δm²₃₂ CV under NO assumption, in 10⁻³ eV², range ~2.25–2.55
- Y: JUNO fractional precision on |Δm²₃₂|, in %, range ~0.2–2.0
- Colour (Z): log₁₀(Bayes Factor), positive = NO preference, negative = IO preference
- Top x-axis: corresponding IO central value (offset by λ ≈ 0.12 × 10⁻³ eV²)
- Iso-contours at σ = ±1, ±2, ±3 equivalent significance

**Triangle plot parameters** (4 oscillation parameters NOvA is sensitive to):
- δCP (in π)
- sin²θ₂₃
- sin²(2θ₁₃)
- |Δm²₃₂| (in 10⁻³ eV²)

Diagonal panels: 1D marginalised posteriors. Off-diagonal: 2D credible interval contours.
Red = IO, Blue = NO.

---

## Sigma Conversion (bf_to_sigma)

This is the exact conversion used throughout. Do not deviate from it.

```python
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
```

Signed sigma: `sigmas = bf_to_sigma(np.abs(bfs)) * np.sign(bfs)`

Iso-contour thresholds on the crystal ball plot are at σ = ±1, ±2, ±3, computed by inverting this function:
- σ=1 → BF ≈ 6.83
- σ=2 → BF ≈ 21.98
- σ=3 → BF ≈ 369.4

For the website, precompute log₁₀(BF) thresholds corresponding to σ = 1, 2, 3 and draw contour lines at those values.

---

## Data Source

**Primary data file:** `Artur_code/bayes_factors_and_rmse_prel.npz`
**Size:** ~2 GB. Never commit this to git. It stays local.
**Format:** NumPy .npz archive

### Key arrays in the npz

| Key | Shape | Description |
|-----|-------|-------------|
| `bfs` | (32, 32) | Bayes factors, indexed [i_dm32, j_precision] |
| `rmses` | (32, 32) | RMSE on Bayes factors |
| `dm32_values` | (32,) | Δm²₃₂ central values in eV² (multiply by 1e3 for 10⁻³ eV²) |
| `precision_values` | (32,) | Precision values in % |

### Triangle plot histogram data (per grid point)

For each grid point (i, j), the npz contains pre-binned posterior histograms keyed as:

**1D histograms (diagonal panels):**
- `plot_{param}_hist_no_i{i}_j{j}` — NO posterior histogram
- `plot_{param}_hist_io_i{i}_j{j}` — IO posterior histogram
- `plot_{param}_edges_i{i}_j{j}` — bin edges

Where `{param}` is one of: `delta_in_pi`, `ss2th13`, `ssth23`, `absdm32`

**2D histograms (off-diagonal panels):**
- `plot_{param_row}_{param_col}_hist_no_i{i}_j{j}`
- `plot_{param_row}_{param_col}_hist_io_i{i}_j{j}`
- `plot_{param_row}_{param_col}_edges_i{i}_j{j}` — shape (2, nbins+1): [y_edges, x_edges]

**Triangle plot parameter order (top-left to bottom-right):**
```python
parameters = ["delta_in_pi", "ss2th13", "ssth23", "absdm32"]
```

Off-diagonal key construction: row param first, then column param.
Example: row=ssth23, col=delta_in_pi → `plot_ssth23_delta_in_pi_hist_no_i{i}_j{j}`

### LaTeX labels for parameters
```python
parameters_latex = {
    "delta_in_pi": r"$\delta_{CP}$ (in $\pi$)",
    "ss2th13":     r"$\sin^2 2\theta_{13}$",
    "ssth23":      r"$\sin^2\theta_{23}$",
    "absdm32":     r"$|\Delta m^2_{32}|$ [$\times 10^{-3}$ eV$^2$]",
}
```

---

## HPD Contour Method (for triangle plots)

```python
def get_quantile_threshold(hist, quantile=0.6827):
    hist = np.asarray(hist)
    flat = hist.ravel()
    sorted_hist = np.sort(flat)[::-1]
    cumsum = np.cumsum(sorted_hist)
    cumsum /= cumsum[-1]
    idx = np.searchsorted(cumsum, quantile)
    return sorted_hist[idx]
```

For 2D panels, contour levels are computed on the **joint** histogram (histno + histio):
```python
hist_joint = histno + histio
thresholds = get_quantile_threshold(hist_joint, quantile=[0.9973, 0.9545, 0.6827])
```
Then draw separate contours for NO and IO at those thresholds.

For 1D panels, shade the region above the 1σ threshold (68.27% HPD) of the joint histogram.

---

## Directory Structure

```
NOvA+JUNO_Website_Release/
├── CLAUDE.md                             ← this file
├── README.md                             ← public-facing description
├── .gitignore                            ← must exclude *.npz and triangles/ if large
├── Artur_code/
│   ├── bayes_factors_and_rmse_prel.npz   ← SOURCE DATA (never commit, ~2GB)
│   └── plotting.py                       ← original PoC script (reference only)
├── Reference_material/
│   └── asztuc_3flav_ninja_20260528.pdf
├── pipeline/                             ← offline build scripts (Python)
│   ├── generate_data.py                  ← M1: extract CSVs + manifest from npz
│   └── generate_triangles.py            ← M2: render all triangle PNGs from npz
├── data/                                 ← build outputs (committed to git)
│   ├── bf_grid.csv                       ← full BF/sigma/RMSE table
│   └── manifest.json                     ← grid point index + PNG filenames
├── triangles/                            ← pre-generated PNGs (git-lfs or Zenodo)
│   └── triangle_i{i:02d}_j{j:02d}.png
└── website/
    ├── v1/                               ← UI variant 1: Scientific/Publication
    │   └── index.html
    ├── v2/                               ← UI variant 2: Dashboard/Dark theme
    │   └── index.html
    └── v3/                               ← UI variant 3: Outreach/Public-facing
        └── index.html
```

---

## Data Contract

### bf_grid.csv columns

| Column | Type | Description |
|--------|------|-------------|
| `i` | int | dm32 grid index |
| `j` | int | precision grid index |
| `dm32_no` | float | Δm²₃₂ CV under NO (× 10⁻³ eV²) |
| `precision` | float | Precision in % |
| `bf` | float | Raw Bayes factor |
| `log10_bf` | float | log₁₀(|BF|) × sign(BF) |
| `sigma` | float | Signed Gaussian-equivalent σ |
| `rmse` | float | Raw RMSE on BF |
| `rmse_sigma` | float | bf_to_sigma(rmse) |
| `frac_rmse` | float | |rmse / bf| |
| `frac_sigma` | float | |rmse_sigma / sigma| |

### manifest.json schema

```json
[
  {
    "i": 0,
    "j": 0,
    "dm32_no": 2.205,
    "precision": 0.40,
    "bf": 15514.96,
    "log10_bf": 4.19,
    "sigma": 8.32,
    "png": "triangles/triangle_i00_j00.png"
  }
]
```

### PNG naming convention

`triangles/triangle_i{i:02d}_j{j:02d}.png`

Zero-padded to 2 digits. Extend to 3 digits (:03d) if grid exceeds 99 in either dimension — update both generate_triangles.py and manifest.json generation consistently.

---

## Milestones

### M1 — pipeline/generate_data.py
**Input:** `Artur_code/bayes_factors_and_rmse_prel.npz`
**Output:** `data/bf_grid.csv`, `data/manifest.json`
**Task:** Load npz. Compute all derived quantities using bf_to_sigma above. Write CSV and manifest JSON.
**Requirements:**
- Multiply dm32_values by 1e3 (convert from eV² to 10⁻³ eV²)
- Create `data/` directory if it does not exist
- All 11 CSV columns as specified in the data contract above
- Manifest must contain one entry per grid point with all fields in the schema above
- Print a summary: grid shape, dm32 range, precision range, BF range, sigma range

**Acceptance check:** A few spot values from manifest.json should match the PoC interactive plot when run against the same npz.

---

### M2 — pipeline/generate_triangles.py
**Input:** `Artur_code/bayes_factors_and_rmse_prel.npz`
**Output:** `triangles/triangle_i{i:02d}_j{j:02d}.png` for all grid points

**Task:** Loop over all (i, j) grid points. For each, extract histogram data from npz, render matplotlib triangle plot, save PNG.

**Requirements:**
- Match the draw_triangle() function in Artur_code/plotting.py exactly for physics and rendering
- Parameters in order: ["delta_in_pi", "ss2th13", "ssth23", "absdm32"]
- Blues colourmap for NO (pcolormesh, alpha=0.5), Reds for IO (pcolormesh, alpha=0.5)
- HPD contours at 1/2/3σ on 2D panels using get_quantile_threshold() above
- 1D panels: stairs plot NO (blue) and IO (red), fill HPD region above 1σ threshold
- Grid on all panels, axis labels from parameters_latex dict, ticks only on leftmost/bottom panels
- Title: `dm32: {val:.2f} [e-3] eV^2, prec: {val:.2f}%`
- Text box with BF, RMSE, Frac RMSE, Sigma values (top-right of figure)
- Add tqdm progress bar
- Skip if PNG already exists (allows interrupted reruns)
- Create `triangles/` directory if it does not exist
- Figure size: (10, 10), dpi=150 (balance quality vs file size)

**Acceptance check:** Spot-check 5 PNGs at known (i, j) values against the PoC interactive_explorer() for the same npz.

---

### M3 — website/v1, v2, v3
**Input:** `data/bf_grid.csv`, `data/manifest.json`, `triangles/` folder
**Output:** `website/v1/index.html`, `website/v2/index.html`, `website/v3/index.html`

Each variant is a **single self-contained HTML file** with all CSS and JS inline. No shared code between variants. No build step required — open index.html in a browser directly.

**All variants must implement this functionality (identical across all three):**

1. Load `../../data/bf_grid.csv` and `../../data/manifest.json` on page load (relative paths)
2. Render crystal ball heatmap from bf_grid.csv:
   - Colour field: log10_bf (default), switchable to sigma or frac_rmse via radio/toggle
   - RdBu colourmap: red = IO (negative BF), blue = NO (positive BF), white = 0
   - Colourbar labelled appropriately for the selected field
3. Dual x-axis: bottom = NO central value (dm32_no), top = IO central value (dm32_no + 0.12)
4. Iso-contours at σ = ±1, ±2, ±3 overlaid on heatmap (use precomputed log10_bf thresholds from bf_to_sigma inverse: ±0.834, ±1.342, ±2.567 in log10_bf units)
5. Slider 1: Δm²₃₂ CV (NO), continuous range spanning manifest min/max dm32_no
6. Slider 2: Precision (%), continuous range spanning manifest min/max precision
7. Crosshair or marker on heatmap at current slider position
8. "Generate" button: find nearest (i, j) in manifest by minimising (dm32_no - slider1)² + (precision - slider2)²; display PNG at `../../triangles/triangle_i{ii}_j{jj}.png`
9. Text overlay showing: selected (CV, precision) from sliders, BF value at nearest grid point, σ significance, nearest grid point label
10. Indicator if slider position differs from nearest grid point: "Showing nearest grid point: dm32={x}, prec={y}%"
11. Field toggle: radio buttons or segmented control to switch heatmap colour between log₁₀(BF), σ, Frac RMSE

**Technology:** Use D3.js v7 (CDN) for the heatmap. Vanilla JS for sliders, button, image display. No React, no build step.

**Variants differ only in aesthetics and layout:**

## Visual Design Reference (v2 — Dashboard Style)

Reference image: Wellmetrix dashboard (Dribbble)

Extract and apply these specific aesthetic elements for v2:

**Background & surface:**
- Page background: warm off-white, e.g. #f5f0eb or #f0ebe4
- Cards: white or #ffffff with soft shadow: 
  box-shadow: 0 2px 16px rgba(0,0,0,0.07)
- Card border-radius: 16px
- No hard borders between cards — separation via shadow and 
  whitespace only

**Typography:**
- Sans-serif only — suggest DM Sans or Plus Jakarta Sans 
  (Google Fonts, free)
- Large bold numerical readouts for BF and sigma values 
  (font-size: 2.5–3rem, font-weight: 700)
- Small muted labels above/below the readouts 
  (font-size: 0.75rem, color: #999, letter-spacing: 0.05em, 
  uppercase)
- Body text: #333, comfortable line-height

**Layout:**
- Slim fixed header bar: logo left, links right, 
  background white, bottom border 1px #eee
- Main content: CSS grid, two columns
  - Left column: crystal ball heatmap card + slider card 
    below it
  - Right column: triangle plot as hero card (tall, 
    full column height) with BF/sigma metric cards 
    stacked above it
- Metric cards (BF value, sigma value) sit above the 
  triangle plot like the "+0.34" and "64%" cards in 
  the reference — small, wide, showing one number each
- Bottom: slim footer card, full width

**Controls:**
- Sliders: custom styled, accent colour #7c6f5e (warm brown) 
  for the filled track portion
- Generate button: filled, rounded-full, same accent colour, 
  white text, subtle hover darkening
- Field toggle: pill-style segmented control, not radio buttons

**Depth & detail:**
- Subtle inner padding on all cards: 24px
- Hover state on cards: very slight shadow increase
- No gradients, no glassmorphism, no blur effects
- Warm neutral accent palette throughout — no loud blues 
  or purples

#### v1 — Scientific / Publication Style
- White background, clean minimal layout
- Font: Georgia or similar serif, or a web-hosted scientific font
- Tight layout matching a paper figure supplement aesthetic
- Thin contour lines with small inline labels (1σ, 2σ, 3σ)
- Sliders as plain range inputs with precise decimal readouts
- Triangle PNG displayed at natural size, bordered with thin rule
- Colourbar matches matplotlib RdBu as closely as possible

#### v2 — Dashboard / Warm Neutral (Wellmetrix-inspired)
Follow the Visual Design Reference section above exactly.
Warm off-white background, card-based layout, large 
numerical readouts for BF and sigma, DM Sans typography,
custom slider styling in warm brown accent.

#### v3 — Outreach / Public-Facing
- Warm, accessible colour palette (light background, friendly typography)
- Large readable font (sans-serif), generous spacing
- Each slider has a one-sentence plain-English tooltip explaining its physical meaning
- Simple legend clearly explaining NO vs IO colour convention and what the Bayes factor means
- Animated fade-in when triangle plot updates (CSS transition)
- "What am I looking at?" collapsible section with 2–3 sentence physics explanation

---

## Extensibility Notes

- **Larger grids:** Re-run M1 and M2 with the new npz. Website code does not change. Replace `data/` and `triangles/`. Manifest auto-encodes the new grid size.
- **λ slider (future extension):** λ is currently hardcoded at 0.12 × 10⁻³ eV² 
  for the IO x-axis offset. Adding λ as a third slider dimension is non-trivial: 
  λ enters the Stan HMC inference itself, not just the axis labels. A different λ 
  means different posteriors, which means a different npz must be generated by 
  running Artur's full pipeline again at each λ value. The data structure becomes 
  a 3D grid: (dm32 CV) × (precision) × (λ). When this is implemented:
  - The manifest gains a third index `l` and a `lambda_val` field
  - PNGs are named `triangle_i{i:02d}_j{j:02d}_l{l:02d}.png`
  - The nearest-neighbour lookup becomes 3D
  - M1 and M2 loop over the λ dimension
  - The current 32×32 grid is the l=0 slice of the eventual 3D grid
  - Do not attempt to implement this by shifting axis labels — it will be wrong
- **PNG hosting for large grids:** If the grid exceeds ~5000 points, github.io will not support the triangles/ folder (~1.5 GB+). In that case, host triangles/ on Zenodo or a NOvA server and update the `png` field in manifest.json to absolute URLs. Website JS does not change.
- **PNG zero-padding:** If grid exceeds 99 in either dimension, change :02d to :03d in both generate_triangles.py and manifest.json generation simultaneously.
- **PNG caching (deployment):** Triangle PNGs should be served with 
  `Cache-Control: public, max-age=31536000, immutable` headers. Since PNG 
  filenames encode the grid indices and the grid is fixed at build time, the 
  files never change — aggressive caching means concurrent users hitting the 
  same grid point only generate one network request between them. This is a 
  web server / github Pages configuration detail, not a code change. On 
  github.io this is handled automatically. On a NOvA server it needs to be 
  set explicitly.

## JUNO result marker (all variants)

At the top of each index.html, include this config block 
exactly as written — do not hardcode the values elsewhere 
in the code:

const JUNO_RESULT = {
  dm32_no: null,
  precision: null,
  label: "JUNO 2026"
};

When both dm32_no and precision are non-null:
- Render a star marker (✦) on the crystal ball heatmap at 
  that (dm32_no, precision) coordinate
- The marker should be visually prominent — white fill, 
  dark stroke, size ~12px
- On hover: tooltip showing label, BF at nearest grid point, 
  σ significance
- The marker sits on top of the heatmap, above the contour 
  lines

When either is null: render nothing. No placeholder, no 
empty marker.

This is the mechanism for publishing the NOvA+JUNO result 
the moment JUNO announces. It must work with a one-line 
change and a git push.
---

## Known Issues (to address before final release)

- **Infinity sigma values:** pipeline/generate_data.py produces inf 
  sigma for extreme BF values (when p-value underflows). Fix: clamp 
  sigma to ±8.0 at generation time. Then update website display to 
  show ">8σ" for any |sigma| ≥ 8.

## What NOT to Do

- Do not load the npz file in the browser. It is 2 GB and will not work client-side.
- Do not implement a backend or server-side rendering. Everything must be static.
- Do not commit `bayes_factors_and_rmse_prel.npz` to git (add to .gitignore).
- Do not hardcode grid dimensions (32, 32) in the website — read them from manifest.json.
- Do not recompute sigma or BF in the browser from scratch — all derived quantities are precomputed in bf_grid.csv.
- Do not use hover to trigger triangle plot display — it must be the Generate button.
- Do not share CSS or JS between website variants — each must be independently deployable.
- Do not use React or any framework requiring a build step for the website variants.

---

## Reference Files

- `Artur_code/plotting.py` — original PoC. The `draw_triangle()`, `get_quantile_threshold()`, `bf_to_sigma()`, and `interactive_explorer()` functions are the reference implementation. Match their physics and rendering logic exactly.
- `Reference_material/asztuc_3flav_ninja_20260528.pdf` — Artur Sztuc's data release talk (28 May 2026). Slide 4 shows the target interactive layout. Slide 3 shows the BF table column schema.BB