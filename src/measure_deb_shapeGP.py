#!/usr/bin/env python3
"""DEB 2026 coronal isophote geometry — preliminary/reproducible analysis.

Fits ellipses to total-intensity coronal brightness isophotes and measures
ellipse ellipticity epsilon = 1-b/a and semi-major-axis direction.

Based on Penn et al. (2026), doi:10.3847/2041-8213/ae8797.
"""
import argparse, json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import Ellipse
from astropy.io import fits
from skimage import measure
from scipy.ndimage import gaussian_filter
from lunar_limb import find_lunar_limb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "hdr" / "solar_corona_hdr_linear.fits"
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
MAX_QUALITY_CENTER_OFFSET_RSUN = 0.12
NORTH_ANGLE_IMAGE_DEG = None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("fits_path", nargs="?", default=None,
                   help="2-D monochrome HDR FITS (default: project linear HDR FITS)")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--cx", type=float, default=None, help="solar center x [px]")
    p.add_argument("--cy", type=float, default=None, help="solar center y [px]")
    p.add_argument("--rsun", type=float, default=None, help="solar radius [px]")
    p.add_argument("--bootstrap", type=int, default=20,
                   help="bootstrap samples for fit uncertainties (default: 20)")
    p.add_argument("--sigma", type=float, default=SMOOTH_SIGMA)
    p.add_argument("--profile-mode", choices=("monotonic", "raw"), default="monotonic",
                   help="radial level profile used for contours (default: monotonic)")
    return p.parse_args()


def load_fits(path):
    raw = np.asarray(fits.getdata(path))
    data = _collapse_to_mono(raw)
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(f"Expected 2-D monochrome HDR FITS; got {data.shape}")
    data[~np.isfinite(data)] = np.nan
    return data


def _collapse_to_mono(data, rtol=1e-6):
    """Handles FITS cubes that carry a mono image replicated across a small
    leading/trailing axis (e.g. NAXIS3=3 written by some capture/processing
    tools even for genuinely monochrome data -- observed directly in a real
    DEB file: a (3, ny, nx) cube where all three planes were bit-identical).
    np.squeeze() alone does NOT collapse this, since the axis has length 3,
    not 1 -- verified this crashes load_fits()'s ndim==2 check on that file.

    This only collapses when the channels are numerically identical (or
    within rtol/atol of each other, to tolerate float round-trip noise). If
    they genuinely differ, this is real multi-channel data and combining it
    would require a deliberate choice (luminance weighting, per-channel
    analysis, etc.) that shouldn't be made silently -- so it's left as-is
    and the ndim check above will raise with the true shape, rather than
    quietly averaging or picking one channel and giving a wrong answer."""
    data = np.asarray(data)
    if data.ndim != 3:
        return data
    channel_axis = next((ax for ax, n in enumerate(data.shape) if n <= 4), None)
    if channel_axis is None:
        return data
    moved = np.moveaxis(data, channel_axis, 0)
    n_ch = moved.shape[0]
    if n_ch == 1:
        return moved[0]
    ref = moved[0]
    if all(np.allclose(moved[c], ref, rtol=rtol, atol=1e-9, equal_nan=True) for c in range(1, n_ch)):
        return ref
    return data


def largest_centered_contour(data, level, cx, cy):
    contours = measure.find_contours(data, level)
    best = None
    center = np.array([[cx, cy]])
    for c in contours:
        if len(c) < MIN_CONTOUR_POINTS:
            continue
        xy = np.column_stack((c[:, 1], c[:, 0]))
        if (np.any(xy[:,0] <= BOUNDARY_MARGIN_PX) or
            np.any(xy[:,0] >= data.shape[1]-1-BOUNDARY_MARGIN_PX) or
            np.any(xy[:,1] <= BOUNDARY_MARGIN_PX) or
            np.any(xy[:,1] >= data.shape[0]-1-BOUNDARY_MARGIN_PX)):
            continue
        if not MplPath(xy).contains_points(center)[0]:
            continue
        x, y = xy[:,0], xy[:,1]
        area = 0.5*abs(np.sum(x*np.roll(y,-1)-y*np.roll(x,-1)))
        if best is None or area > best[0]:
            best = (area, xy)
    return best


def fit_ellipse(contour_xy, cx, cy, rsun):
    try:
        model = measure.EllipseModel.from_estimate(contour_xy)
    except (ValueError, np.linalg.LinAlgError):
        return None
    xc, yc = model.center
    a, b = model.axis_lengths
    theta = model.theta
    a, b = float(abs(a)), float(abs(b))
    if b > a:
        a, b = b, a
        theta += np.pi/2
    if a <= 0 or b <= 0:
        return None
    axis_ratio = a/b
    center_offset = np.hypot(xc-cx, yc-cy)/rsun
    if axis_ratio > MAX_AXIS_RATIO or center_offset > MAX_CENTER_OFFSET_RSUN:
        return None
    rmean = np.sqrt(a*b)/rsun
    eps = 1.0 - b/a
    resid = model.residuals(contour_xy)
    rms_px = float(np.sqrt(np.mean(resid**2)))
    params: dict[str, Any] = {
        "xc_px": float(xc), "yc_px": float(yc),
        "a_px": a, "b_px": b,
        "r_mean_rsun": float(rmean),
        "epsilon": float(eps),
        "axis_ratio": float(axis_ratio),
        "theta_image_deg": float(np.degrees(theta)%180),
        "center_offset_rsun": float(center_offset),
        "fit_rms_px": rms_px,
        "fit_rms_rel": float(rms_px/(np.sqrt(a*b))),
    }
    return params, model


def bootstrap_uncertainty(contour_xy, cx, cy, rsun, samples, rng):
    if samples <= 0:
        return np.nan, np.nan
    n_points = min(len(contour_xy), 500)
    base = rng.choice(len(contour_xy), size=n_points, replace=False)
    eps_values, theta_values = [], []
    for _ in range(samples):
        indices = rng.choice(base, size=n_points, replace=True)
        fit = fit_ellipse(contour_xy[indices], cx, cy, rsun)
        if fit is not None:
            params, _ = fit
            eps_values.append(params["epsilon"])
            theta_values.append(params["theta_image_deg"])
    if len(eps_values) < 2:
        return np.nan, np.nan
    theta_radians = np.radians(theta_values) * 2.0
    resultant = np.hypot(np.mean(np.cos(theta_radians)), np.mean(np.sin(theta_radians)))
    theta_sd = np.degrees(np.sqrt(max(0.0, -2.0 * np.log(max(resultant, 1e-12)))) / 2.0)
    return float(np.std(eps_values, ddof=1)), float(theta_sd)


def main():
    args = parse_args()
    path = Path(args.fits_path).expanduser().resolve() if args.fits_path else DEFAULT_INPUT
    if not path.exists():
        raise FileNotFoundError(
            f"FITS input not found: {path}. Supply a FITS path as the first argument."
        )
    out = Path(args.output_dir).expanduser().resolve() if args.output_dir else path.parent/"ellipse_analysis"
    out.mkdir(parents=True, exist_ok=True)

    data = load_fits(path)
    ny, nx = data.shape
    supplied_geometry = [args.cx is not None, args.cy is not None, args.rsun is not None]
    if any(supplied_geometry) and not all(supplied_geometry):
        raise ValueError("Supply --cx, --cy, and --rsun together, or omit all three.")
    geometry_source = "explicit CLI geometry"
    if args.cx is not None and args.cy is not None and args.rsun is not None:
        cx, cy, rsun = args.cx, args.cy, args.rsun
    else:
        # Auto-detect via Hough limb finding for ANY input file, not just
        # the project's canonical default path. The narrower original
        # version fell back to raw frame-center whenever a different FITS
        # was supplied on the command line -- verified this is actively
        # wrong, not just imprecise: on a real DEB frame (stret.fits) the
        # true disk center was offset from frame-center by ~96px (~0.27
        # solar radii), comfortably past MAX_CENTER_OFFSET_RSUN=0.20, so
        # every single ellipse fit was rejected and the run produced zero
        # results with no indication that geometry (not the fitting logic)
        # was the actual problem. Hough detection, verified visually
        # against this same frame, traces the limb correctly.
        detected_cx, detected_cy, detected_rsun, detected_ok = find_lunar_limb(data)
        if detected_ok:
            cx, cy, rsun = detected_cx, detected_cy, detected_rsun
            geometry_source = "automatic Hough limb detection"
        elif path == DEFAULT_INPUT:
            cx, cy, rsun = DEFAULT_CX, DEFAULT_CY, DEFAULT_RSUN_PX
            geometry_source = "fallback project geometry (Hough detection failed)"
        else:
            cx, cy, rsun = nx / 2, ny / 2, DEFAULT_RSUN_PX
            geometry_source = "frame-center fallback (Hough detection failed -- UNVERIFIED, results likely unreliable)"
            print(f"[!] WARNING: automatic limb detection failed on this file and it isn't "
                  f"the project's default input, so there's no calibrated geometry to fall "
                  f"back on. Using raw frame-center with a placeholder radius -- treat any "
                  f"results from this run as unreliable until you supply --cx/--cy/--rsun "
                  f"measured directly from this specific image.")
    work = gaussian_filter(np.nan_to_num(data, nan=0), args.sigma) if args.sigma > 0 else data.copy()

    print(f"[+] {path.name}: {nx}x{ny}")
    print(f"[+] Solar center used: ({cx:.2f}, {cy:.2f}) px")
    print(f"[+] Solar radius used: {rsun:.2f} px")
    print(f"[+] Geometry source: {geometry_source}")

    Y, X = np.indices(work.shape)
    rr = np.hypot(X-cx, Y-cy)/rsun
    target_r = np.linspace(R_MIN, R_MAX, N_LEVELS)
    radial_r = np.arange(R_MIN, R_MAX+0.01, 0.01)
    radial_i = []
    for r0 in radial_r:
        vals = work[(rr >= r0-0.005)&(rr < r0+0.005)&np.isfinite(work)]
        radial_i.append(np.nanmedian(vals) if vals.size else np.nan)
    radial_i = np.asarray(radial_i)
    good = np.isfinite(radial_i) & (radial_i > 0)
    if good.sum() < 10:
        raise RuntimeError("Not enough positive radial-profile samples.")
    mono = np.minimum.accumulate(radial_i[good])
    rad_good = radial_r[good]
    if args.profile_mode == "monotonic":
        profile_values = mono
    else:
        profile_values = radial_i[good]
    levels = np.interp(target_r, rad_good, profile_values)

    records = []
    overlays = []
    for tr, level in zip(target_r, levels):
        found = largest_centered_contour(work, level, cx, cy)
        if found is None:
            continue
        area, contour = found
        fit = fit_ellipse(contour, cx, cy, rsun)
        if fit is None:
            continue
        params, model = fit
        if not (R_MIN <= params["r_mean_rsun"] <= R_MAX):
            continue
        params["target_r_rsun"] = float(tr)
        params["isophote_level"] = float(level)
        params["profile_mode"] = args.profile_mode
        params["epsilon_bootstrap_sd"], params["theta_bootstrap_sd"] = \
            bootstrap_uncertainty(contour, cx, cy, rsun, args.bootstrap, np.random.default_rng(42 + len(records)))
        params["quality_flag"] = (
            "good" if params["fit_rms_rel"] <= MAX_FIT_RMS_REL and
            params["center_offset_rsun"] <= MAX_QUALITY_CENTER_OFFSET_RSUN
            else "review"
        )
        params["contour"] = contour
        params["model"] = model
        records.append(params)
        overlays.append(params)

    if len(records) < 10:
        raise RuntimeError(f"Only {len(records)} valid ellipse fits; check center/radius/input.")

    n_requested = len(target_r)
    n_fitted = len(records)
    n_rejected = n_requested - n_fitted
    max_r_reached = max(r["r_mean_rsun"] for r in records)
    edge_limited = (max_r_reached < R_MAX - 0.02)  # requested range not fully reached
    if edge_limited:
        print(f"[!] Requested isophote range up to R={R_MAX} Rsun, but only reached "
              f"R={max_r_reached:.2f} Rsun ({n_rejected}/{n_requested} requested levels "
              f"produced no valid fit). This commonly means the corona extends past the "
              f"image edge at the outer radii -- check whether the frame crop, not the "
              f"physics, is limiting the usable range here.")

    rows = [{k:v for k,v in r.items() if k not in ("contour","model")} for r in records]
    df = pd.DataFrame(rows).sort_values("r_mean_rsun")
    stem = path.stem
    csv_path = out/f"{stem}_ellipse_measurements.csv"
    df.to_csv(csv_path, index=False)

    meta = {
        "input": str(path), "center_x_px": cx, "center_y_px": cy,
        "solar_radius_px": rsun, "r_min_rsun": R_MIN, "r_max_rsun": R_MAX,
        "n_levels_requested": n_requested, "n_levels_fitted": n_fitted,
        "max_r_reached_rsun": float(max_r_reached),
        "outer_range_edge_limited": bool(edge_limited),
        "smoothing_sigma_px": args.sigma,
        "geometry_source": geometry_source,
        "bootstrap_samples": args.bootstrap,
        "profile_mode": args.profile_mode,
        "quality_limits": {
            "max_fit_rms_rel": MAX_FIT_RMS_REL,
            "max_center_offset_rsun": MAX_QUALITY_CENTER_OFFSET_RSUN
        },
        "pa_status": "image_coordinates only; solar P.A. calibration not supplied -- "
                      "NOT directly comparable to solar-north-referenced P.A. values "
                      "(e.g. from the reference paper) without a separate plate-solve "
                      "or known camera-orientation calibration",
        "reference": "Penn et al. 2026, doi:10.3847/2041-8213/ae8797"
    }
    (out/f"{stem}_analysis_metadata.json").write_text(json.dumps(meta, indent=2))

    x = df.r_mean_rsun.to_numpy()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.4, 7.0), sharex=True, dpi=300)
    ax1.errorbar(x, df.epsilon, yerr=df.epsilon_bootstrap_sd,
                 fmt="o-", ms=3.2, lw=1.1, capsize=2)
    ax1.set_ylabel(r"Ellipse ellipticity $\epsilon=1-b/a$")
    ax1.grid(True, alpha=.25)
    ax2.errorbar(x, df.theta_image_deg, yerr=df.theta_bootstrap_sd,
                 fmt="o-", ms=3.2, lw=1.1, capsize=2)
    ax2.set_ylabel("Semi-major-axis angle (image coordinates, deg)")
    ax2.set_xlabel(r"Mean isophote height ($R_\odot$)")
    ax2.grid(True, alpha=.25)
    fig.suptitle("DEB 2026 Coronal Isophote Geometry — Preliminary", y=.98)
    fig.tight_layout()
    fig_path = out/f"{stem}_coronal_geometry_preliminary.png"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.6, 6.8), dpi=250)
    finite = np.isfinite(data)
    lo, hi = np.nanpercentile(data[finite], [5, 99.7])
    ax.imshow(np.clip(data, lo, hi), cmap="gray", origin="upper")
    step = max(1, len(overlays)//12)
    for r in overlays[::step]:
        c = r["contour"]
        ax.plot(c[:,0], c[:,1], lw=.6, alpha=.55)
        e = Ellipse((r["xc_px"], r["yc_px"]), 2*r["a_px"], 2*r["b_px"],
                    angle=r["theta_image_deg"], fill=False, lw=1.0, alpha=.9)
        ax.add_patch(e)
    ax.plot(cx, cy, "+", ms=11, mew=2)
    ax.set_title("Observed isophotes + fitted ellipses — Preliminary")
    ax.set_aspect("equal")
    ax.set_xlim(0, nx); ax.set_ylim(ny, 0)
    overlay_path = out/f"{stem}_isophote_ellipse_overlay.png"
    fig.savefig(overlay_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[+] {len(df)} valid ellipse fits")
    print(f"[+] CSV:     {csv_path}")
    print(f"[+] Figure:  {fig_path}")
    print(f"[+] Overlay: {overlay_path}")


if __name__ == "__main__":
    main()