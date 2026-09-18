#!/usr/bin/env python3
"""DEB 2026 coronal isophote geometry — preliminary.

Fits ellipses to total-intensity coronal brightness isophotes and measures
ellipse ellipticity epsilon = 1-b/a and semi-major-axis angle in IMAGE
coordinates.  The diagnostic overlay deliberately shows the measured
isophote and the ideal fitted geometric ellipse separately.

Reference: Penn et al. (2026), doi:10.3847/2041-8213/ae8797

Important: this does NOT calculate the Ludendorff flattening parameter.
Solar position angle is not claimed until image-to-solar orientation is
calibrated. The solar center/radius defaults are provisional and should be
replaced with DEB-calibrated 2026 values before final scientific use.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import Ellipse
from astropy.io import fits
from scipy.ndimage import gaussian_filter
from skimage import measure

DEFAULT_CX = 1551.5
DEFAULT_CY = 1145.5
DEFAULT_RSUN_PX = 351.0
R_MIN, R_MAX = 1.40, 2.80
N_LEVELS = 60
SMOOTH_SIGMA = 0.0
MIN_CONTOUR_POINTS = 150
BOUNDARY_MARGIN_PX = 5
MAX_AXIS_RATIO = 4.0
MAX_CENTER_OFFSET_RSUN = 0.20
MAX_FIT_RMS_REL = 0.05
DISPLAY_RADII = np.array([1.45, 1.65, 1.85, 2.05, 2.25, 2.45, 2.65, 2.78])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
    "fits_path",
    nargs="?",
    default=str(
        Path(__file__).resolve().parent.parent
        / "data" / "processed" / "hdr"
        / "solar_corona_hdr_linear.fits"
    )
)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--cx", type=float, default=DEFAULT_CX)
    p.add_argument("--cy", type=float, default=DEFAULT_CY)
    p.add_argument("--rsun", type=float, default=DEFAULT_RSUN_PX)
    p.add_argument("--sigma", type=float, default=SMOOTH_SIGMA)
    p.add_argument("--levels", type=int, default=N_LEVELS)
    return p.parse_args()


def collapse_identical_cube(data, rtol=1e-6):
    data = np.asarray(data)
    data = np.squeeze(data)
    if data.ndim != 3:
        return data
    channel_axis = next((ax for ax, n in enumerate(data.shape) if n <= 4), None)
    if channel_axis is None:
        return data
    moved = np.moveaxis(data, channel_axis, 0)
    if moved.shape[0] == 1:
        return moved[0]
    ref = moved[0]
    identical = all(np.allclose(moved[i], ref, rtol=rtol, atol=1e-9, equal_nan=True)
                    for i in range(1, moved.shape[0]))
    return ref if identical else data


def load_fits(path):
    data = collapse_identical_cube(fits.getdata(path))
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(f"Expected one monochrome 2-D image; got {data.shape}")
    data[~np.isfinite(data)] = np.nan
    return data


def choose_centered_contour(data, level, cx, cy):
    """Largest closed contour at level that encloses the supplied reference center."""
    contours = measure.find_contours(data, level)
    center = np.array([[cx, cy]], dtype=float)
    best = None
    for contour in contours:
        if len(contour) < MIN_CONTOUR_POINTS:
            continue
        xy = np.column_stack((contour[:, 1], contour[:, 0]))
        if (np.any(xy[:, 0] <= BOUNDARY_MARGIN_PX) or
            np.any(xy[:, 0] >= data.shape[1] - 1 - BOUNDARY_MARGIN_PX) or
            np.any(xy[:, 1] <= BOUNDARY_MARGIN_PX) or
            np.any(xy[:, 1] >= data.shape[0] - 1 - BOUNDARY_MARGIN_PX)):
            continue
        if not MplPath(xy).contains_points(center)[0]:
            continue
        x, y = xy[:, 0], xy[:, 1]
        area = 0.5 * abs(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))
        if best is None or area > best[0]:
            best = (area, xy)
    return best


def fit_ellipse(contour_xy, cx, cy, rsun):
    model = measure.EllipseModel()
    if not model.estimate(contour_xy):
        return None
    xc, yc, a, b, theta = model.params
    a, b = float(abs(a)), float(abs(b))
    theta = float(theta)
    if b > a:
        a, b = b, a
        theta += np.pi / 2.0
    if a <= 0 or b <= 0:
        return None
    axis_ratio = a / b
    center_offset = np.hypot(xc - cx, yc - cy) / rsun
    if axis_ratio > MAX_AXIS_RATIO or center_offset > 0.20:
        return None
    r_mean = np.sqrt(a * b) / rsun
    epsilon = 1.0 - b / a
    residuals = model.residuals(contour_xy)
    rms_px = float(np.sqrt(np.mean(residuals ** 2)))
    return {
        "xc_px": float(xc), "yc_px": float(yc),
        "a_px": a, "b_px": b,
        "r_mean_rsun": float(r_mean),
        "epsilon": float(epsilon),
        "axis_ratio": float(axis_ratio),
        "theta_image_deg": float(np.degrees(theta) % 180.0),
        "center_offset_rsun": float(center_offset),
        "fit_rms_px": rms_px,
        "fit_rms_rel": float(rms_px / np.sqrt(a * b)),
        "contour": contour_xy,
    }


def make_analysis(data, cx, cy, rsun, sigma, n_levels):
    work = data.copy()
    if sigma > 0:
        work = gaussian_filter(np.nan_to_num(work, nan=0.0), sigma=sigma)
    yy, xx = np.indices(work.shape)
    radial_r = np.hypot(xx - cx, yy - cy) / rsun

    r_grid = np.arange(R_MIN, R_MAX + 0.005, 0.005)
    radial_i = []
    for r0 in r_grid:
        sel = ((radial_r >= r0 - 0.0025) &
               (radial_r < r0 + 0.0025) & np.isfinite(work))
        vals = work[sel]
        radial_i.append(np.nanmedian(vals) if vals.size else np.nan)
    radial_i = np.asarray(radial_i)
    good = np.isfinite(radial_i) & (radial_i > 0)
    if good.sum() < 10:
        raise RuntimeError("Could not construct a usable radial intensity profile.")

    monotonic = np.minimum.accumulate(radial_i[good])
    rg = r_grid[good]
    target_r = np.linspace(R_MIN, R_MAX, n_levels)
    levels = np.interp(target_r, rg, monotonic)

    records = []
    for target, level in zip(target_r, levels):
        found = choose_centered_contour(work, level, cx, cy)
        if found is None:
            continue
        _, contour = found
        fit = fit_ellipse(contour, cx, cy, rsun)
        if fit is None or not (R_MIN <= fit["r_mean_rsun"] <= R_MAX):
            continue
        fit["target_r_rsun"] = float(target)
        fit["isophote_level"] = float(level)
        fit["quality"] = "good" if fit["fit_rms_rel"] <= MAX_FIT_RMS_REL else "review"
        records.append(fit)
    if len(records) < 10:
        raise RuntimeError(f"Only {len(records)} valid ellipse fits.")
    return records


def choose_display_records(records):
    available = np.array([r["r_mean_rsun"] for r in records])
    chosen, used = [], set()
    for target in DISPLAY_RADII:
        idx = int(np.argmin(np.abs(available - target)))
        if idx in used:
            continue
        used.add(idx)
        chosen.append(records[idx])
    return sorted(chosen, key=lambda r: r["r_mean_rsun"])


def plot_results(data, records, cx, cy, rsun, output_dir, stem):
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "contour"} for r in records])
    df = df.sort_values("r_mean_rsun")
    df.to_csv(output_dir / f"{stem}_ellipse_measurements.csv", index=False)

    # The actual numerical results Matt asked about: ellipticity and axis angle.
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 7.0), dpi=300, sharex=True)
    ax1.plot(df["r_mean_rsun"], df["epsilon"], "o-", ms=3, lw=1.1)
    ax1.set_ylabel(r"Ellipse ellipticity  $\epsilon = 1-b/a$")
    ax1.grid(True, alpha=0.25)
    ax2.plot(df["r_mean_rsun"], df["theta_image_deg"], "o-", ms=3, lw=1.1)
    ax2.set_ylabel("Semi-major-axis angle (image coordinates, °)")
    ax2.set_xlabel(r"Mean isophote height  ($R_\odot$)")
    ax2.grid(True, alpha=0.25)
    fig.suptitle("DEB 2026 Coronal Isophote Geometry — Preliminary", y=0.98)
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}_geometry_preliminary.png", bbox_inches="tight")
    plt.close(fig)

    # Clear diagnostic: observed contour versus *ideal fitted ellipse*.
    finite = np.isfinite(data)
    lo, hi = np.nanpercentile(data[finite], [5, 99.7])
    fig, ax = plt.subplots(figsize=(10, 8), dpi=250)
    ax.imshow(np.clip(data, lo, hi), cmap="gray", origin="upper")

    display_records = choose_display_records(records)
    for i, rec in enumerate(display_records, start=1):
        contour = rec["contour"]
        ax.plot(contour[:, 0], contour[:, 1], color="cyan", lw=0.8, alpha=0.45,
                label="Observed isophote" if i == 1 else None)

        patch = Ellipse((rec["xc_px"], rec["yc_px"]),
                        width=2 * rec["a_px"], height=2 * rec["b_px"],
                        angle=rec["theta_image_deg"], fill=False,
                        edgecolor="magenta", lw=2.8, alpha=1.0,
                        label="Fitted geometric ellipse" if i == 1 else None)
        ax.add_patch(patch)

        # Explicit semi-major axis, so the angle has a visible geometric meaning.
        ang = np.radians(rec["theta_image_deg"])
        dx, dy = rec["a_px"] * np.cos(ang), rec["a_px"] * np.sin(ang)
        ax.plot([rec["xc_px"] - dx, rec["xc_px"] + dx],
                [rec["yc_px"] - dy, rec["yc_px"] + dy],
                color="yellow", lw=1.0, alpha=0.95)

        tx = rec["xc_px"] + rec["a_px"] * np.cos(ang)
        ty = rec["yc_px"] + rec["a_px"] * np.sin(ang)
        ax.text(tx + 12, ty - 12, f'{rec["r_mean_rsun"]:.2f} $R_\\odot$',
                color="white", fontsize=9, weight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black",
                          edgecolor="none", alpha=0.65))

    ax.plot(cx, cy, "+", color="white", ms=13, mew=2.2, label="Reference center")
    ax.set_title("Observed isophotes vs. fitted geometric ellipses\n"
                 "cyan = measured contour   magenta = fitted ellipse   yellow = major axis")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.75)
    ax.set_aspect("equal")
    ax.set_xlim(0, data.shape[1])
    ax.set_ylim(data.shape[0], 0)
    ax.set_xlabel("Image X (px)")
    ax.set_ylabel("Image Y (px)")
    overlay_path = output_dir / f"{stem}_ellipse_overlay_clear.png"
    fig.savefig(overlay_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    metadata = {
        "input": stem,
        "solar_center_x_px": cx,
        "solar_center_y_px": cy,
        "solar_radius_px": rsun,
        "analysis_range_rsun": [R_MIN, R_MAX],
        "n_levels_fitted": len(records),
        "displayed_radii_rsun": [float(r["r_mean_rsun"]) for r in display_records],
        "smoothing_sigma_px": 0.0,
        "pa_status": "IMAGE COORDINATES ONLY; not solar PA",
        "note": "Magenta curves are analytic fitted ellipses drawn from fit parameters; cyan curves are measured brightness isophotes.",
    }
    (output_dir / f"{stem}_metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"[+] {len(df)} ellipse measurements saved")
    print(f"[+] Geometry plot: {output_dir / (stem + '_geometry_preliminary.png')}")
    print(f"[+] Clear overlay: {overlay_path}")
    return overlay_path


def main():
    args = parse_args()
    fits_path = Path(args.fits_path).expanduser().resolve()
    if not fits_path.exists():
        raise FileNotFoundError(f"FITS file not found: {fits_path}")
    output_dir = (Path(args.output_dir).expanduser().resolve()
                  if args.output_dir else fits_path.parent / "ellipse_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_fits(fits_path)
    print(f"[+] FITS: {fits_path}")
    print(f"[+] Image: {data.shape[1]} x {data.shape[0]} px")
    print(f"[+] Center used: ({args.cx:.2f}, {args.cy:.2f}) px")
    print(f"[+] R_sun used: {args.rsun:.2f} px (provisional — verify with DEB)")

    records = make_analysis(data, args.cx, args.cy, args.rsun, args.sigma, args.levels)
    plot_results(data, records, args.cx, args.cy, args.rsun, output_dir, fits_path.stem)


if __name__ == "__main__":
    main()