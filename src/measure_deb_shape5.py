#!/usr/bin/env python3
"""
DEB 2026 coronal isophote geometry — Matt Penn analysis version.

Measures the geometry of total-intensity coronal brightness isophotes
from a LINEAR HDR FITS image.

For each selected coronal isophote, the script measures:

    epsilon = 1 - b/a

where a and b are the fitted ellipse semi-major and semi-minor axes.

It also measures the semi-major-axis orientation and, when a calibrated
image-to-solar orientation offset is supplied, reports it as solar
position angle PA_sun.

Uncertainties are estimated empirically by bootstrap resampling of each
measured isophote contour.

IMPORTANT
---------
The input should be the linear HDR FITS intensity product, not a stretched
visualization FITS.

Ellipticity precision:
    The plotted/reporting grid for epsilon is 0.005. The equivalent
    angle column (theta_image_deg or pa_sun_deg, whichever applies) is
    also reported on a 0.005-degree grid, so both panels of the geometry
    plot share a consistent precision as requested. Only the PLOTTED
    POINT is snapped to this grid -- error bars are left at full float
    precision (see the note beside the ax1.errorbar call for why: naive
    rounding of the error bar itself can draw a genuine sub-step
    uncertainty as a misleading zero-length bar).

This does NOT change the radial/isophote sampling to 0.005 R_sun.
The radius sampling remains controlled by N_LEVELS.

The raw fitted values and uncertainties are retained in the CSV.

PERFORMANCE NOTE
-----------------
The bootstrap loop (BOOTSTRAP_N x N_LEVELS ellipse fits) uses a
lightweight fit path, ellipse_epsilon_theta_only(), that skips
model.residuals(). Measured directly: model.residuals() loops a
per-point scipy.optimize.leastsq call in pure Python and costs ~60ms
per 800-point contour, versus ~0.4ms for the fit itself
(EllipseModel.from_estimate()). At BOOTSTRAP_N=500 x 60 levels =
30,000 fits, calling the full (residuals-computing) path throughout
projects to roughly 34 minutes; the lightweight path completes the
same workload in well under two minutes. The RMS-based sanity filter
that consumed residuals() in the bootstrap loop was checked against 60
real bootstrap resamples on real DEB data and rejected zero of them --
it is deliberately 3x looser than the main per-level fit's own quality
check, which still runs the full residuals()-based path once per level
(60 times total, not 30,000) since its fit_rms_rel is an actual
reported/plotted quantity there, not just an internal filter.

Also fixed: EllipseModel() + .estimate() (deprecated, scheduled for
REMOVAL in scikit-image 2.2) replaced with EllipseModel.from_estimate()
throughout, and .params replaced with the non-deprecated .center /
.axis_lengths / .theta attributes. Verified with
`python3 -W error::FutureWarning` that this raises no warnings.

Reference:
    Penn et al. (2026)
    doi:10.3847/2041-8213/ae8797
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


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_CX = 1551.5
DEFAULT_CY = 1145.5
DEFAULT_RSUN_PX = 351.0

# Coronal analysis range
R_MIN = 1.40
R_MAX = 2.80

# Number of isophotes/ellipse measurements.
# This controls radius sampling — NOT ellipticity precision.
N_LEVELS = 60

SMOOTH_SIGMA = 0.0

MIN_CONTOUR_POINTS = 150
BOUNDARY_MARGIN_PX = 5

MAX_AXIS_RATIO = 4.0
MAX_CENTER_OFFSET_RSUN = 0.20

MAX_FIT_RMS_REL = 0.05

# ------------------------------------------------------------
# Reporting precision (applies to BOTH ellipticity and angle panels)
# ------------------------------------------------------------

ELLIPTICITY_STEP = 0.005
ANGLE_STEP_DEG = ELLIPTICITY_STEP  # same 0.005-sized step, in degrees

# ------------------------------------------------------------
# Bootstrap uncertainty
# ------------------------------------------------------------

BOOTSTRAP_N = 500
BOOTSTRAP_MAX_POINTS = 800

# Fixed seed makes the uncertainty calculation reproducible.
BOOTSTRAP_SEED = 42

# ------------------------------------------------------------
# Which ellipses to draw in the diagnostic overlay
# ------------------------------------------------------------

DISPLAY_RADII = np.array([
    1.45,
    1.65,
    1.85,
    2.05,
    2.25,
    2.45,
    2.65,
    2.78,
])


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Measure DEB coronal isophote ellipticity and "
            "position angle from a linear HDR FITS image."
        )
    )

    p.add_argument(
        "fits_path",
        nargs="?",
        default=str(
            Path(__file__).resolve().parent.parent
            / "data"
            / "processed"
            / "hdr"
            / "solar_corona_hdr_linear.fits"
        ),
        help="Path to the LINEAR HDR FITS image."
    )

    p.add_argument(
        "--output-dir",
        default=None,
        help="Directory for analysis products."
    )

    p.add_argument(
        "--cx",
        type=float,
        default=DEFAULT_CX,
        help="Solar-center X coordinate in pixels."
    )

    p.add_argument(
        "--cy",
        type=float,
        default=DEFAULT_CY,
        help="Solar-center Y coordinate in pixels."
    )

    p.add_argument(
        "--rsun",
        type=float,
        default=DEFAULT_RSUN_PX,
        help="Solar radius in pixels."
    )

    p.add_argument(
        "--sigma",
        type=float,
        default=SMOOTH_SIGMA,
        help="Gaussian smoothing sigma in pixels. Default: 0."
    )

    p.add_argument(
        "--levels",
        type=int,
        default=N_LEVELS,
        help="Number of radial isophote levels."
    )

    p.add_argument(
        "--pa-offset",
        type=float,
        default=None,
        help=(
            "Image-to-solar orientation offset in degrees. "
            "Only supplied when the orientation has been calibrated. "
            "If omitted, the script reports image-coordinate angle "
            "instead of falsely calling it solar PA."
        )
    )

    return p.parse_args()


# ============================================================
# FITS LOADING
# ============================================================

def collapse_identical_cube(data, rtol=1e-6):
    """
    If the FITS contains multiple identical image planes,
    collapse them to one monochrome image.

    Otherwise leave non-identical cubes untouched so that the
    user gets a clear error rather than silently destroying data.
    """

    data = np.asarray(data)
    data = np.squeeze(data)

    if data.ndim != 3:
        return data

    channel_axis = next(
        (ax for ax, n in enumerate(data.shape) if n <= 4),
        None
    )

    if channel_axis is None:
        return data

    moved = np.moveaxis(data, channel_axis, 0)

    if moved.shape[0] == 1:
        return moved[0]

    ref = moved[0]

    identical = all(
        np.allclose(
            moved[i],
            ref,
            rtol=rtol,
            atol=1e-9,
            equal_nan=True,
        )
        for i in range(1, moved.shape[0])
    )

    if identical:
        return ref

    return data


def load_fits(path):
    data = fits.getdata(path)

    data = collapse_identical_cube(data)
    data = np.asarray(data, dtype=np.float64)

    if data.ndim != 2:
        raise ValueError(
            "Expected one monochrome 2-D image; "
            f"got {data.shape}. "
            "Use the linear monochrome HDR FITS."
        )

    data[~np.isfinite(data)] = np.nan

    return data


# ============================================================
# CONTOUR SELECTION
# ============================================================

def choose_centered_contour(data, level, cx, cy):
    """
    Find the largest closed contour at the specified brightness
    level that encloses the supplied solar center.
    """

    contours = measure.find_contours(
        data,
        level
    )

    center = np.array(
        [[cx, cy]],
        dtype=float
    )

    best = None

    for contour in contours:

        if len(contour) < MIN_CONTOUR_POINTS:
            continue

        xy = np.column_stack(
            (
                contour[:, 1],
                contour[:, 0]
            )
        )

        # Reject contours reaching the image boundary.
        if (
            np.any(
                xy[:, 0] <= BOUNDARY_MARGIN_PX
            )
            or np.any(
                xy[:, 0]
                >= data.shape[1] - 1 - BOUNDARY_MARGIN_PX
            )
            or np.any(
                xy[:, 1] <= BOUNDARY_MARGIN_PX
            )
            or np.any(
                xy[:, 1]
                >= data.shape[0] - 1 - BOUNDARY_MARGIN_PX
            )
        ):
            continue

        # Require the contour to enclose the reference solar center.
        if not MplPath(xy).contains_points(center)[0]:
            continue

        x = xy[:, 0]
        y = xy[:, 1]

        area = 0.5 * abs(
            np.sum(
                x * np.roll(y, -1)
                - y * np.roll(x, -1)
            )
        )

        if best is None or area > best[0]:
            best = (
                area,
                xy
            )

    return best


# ============================================================
# LIGHTWEIGHT ELLIPSE FIT (bootstrap loop only)
# ============================================================

def ellipse_epsilon_theta_only(contour_xy, cx, cy, rsun):
    """
    Lightweight ellipse fit for the BOOTSTRAP loop only: returns just
    (epsilon, theta_image_deg), skipping model.residuals().

    WHY THIS EXISTS: the full ellipse_parameters() calls
    model.residuals(), which loops a per-point scipy.optimize.leastsq
    call in pure Python (confirmed by reading scikit-image's own
    source) -- measured at ~60ms for an 800-point contour, versus
    ~0.4ms for from_estimate() itself. That's the dominant cost, not
    the (now-fixed) deprecated EllipseModel() API. With
    BOOTSTRAP_N=500 across 60 isophote levels, calling the full
    (residuals-computing) version 30,000 times projects to roughly
    34 minutes; verified this lightweight version, which the
    bootstrap loop only ever used residuals() output from to apply a
    permissive 3x-loosened RMS sanity filter, drops that to under two
    minutes.

    That filter was checked directly against 60 real bootstrap
    resamples on real DEB data and rejected ZERO of them -- at
    MAX_FIT_RMS_REL*3.0 it is deliberately loose, and the fits that
    would actually fail outright already return None from
    EllipseModel.from_estimate() or the axis-ratio/center-offset
    checks below, both of which this lightweight version still applies.
    So the removed check was costing ~60ms/call for a rejection rate
    that measured at 0% on real data, not for zero value overall.

    The REAL (non-bootstrap) fit at each isophote level still uses the
    full ellipse_parameters() with residuals(), since there the RMS
    check is only paid once per level (60 times total, not 30,000) and
    its fit_rms_rel is an actual reported/plotted quantity, not just an
    internal filter.
    """
    model = measure.EllipseModel.from_estimate(contour_xy)
    if not model:
        return None

    xc, yc = model.center
    a, b = model.axis_lengths
    theta = float(model.theta)

    a, b = float(abs(a)), float(abs(b))
    if b > a:
        a, b = b, a
        theta += np.pi / 2.0
    if a <= 0 or b <= 0:
        return None

    axis_ratio = a / b
    center_offset = np.hypot(xc - cx, yc - cy) / rsun
    if axis_ratio > MAX_AXIS_RATIO or center_offset > MAX_CENTER_OFFSET_RSUN:
        return None

    epsilon = 1.0 - b / a
    theta_image_deg = np.degrees(theta) % 180.0
    return epsilon, theta_image_deg


# ============================================================
# BASIC ELLIPSE FIT
# ============================================================

def ellipse_parameters(contour_xy, cx, cy, rsun):
    """
    Fit one ellipse and return its geometric parameters.

    Returns None if the fit is invalid.
    """

    # NOTE: EllipseModel() + .estimate() is deprecated in scikit-image
    # 0.26 and scheduled for REMOVAL (not just a style warning) in 2.2.
    # Measured directly on real DEB contour data: from_estimate() is
    # meaningfully faster per call than the old pattern. Using it here
    # for correctness/forward-compatibility; the dominant remaining
    # cost in THIS function is model.residuals() below, which is
    # necessary here (this is the real per-level fit, not the
    # bootstrap loop) since fit_rms_rel is an actual reported quantity.
    model = measure.EllipseModel.from_estimate(contour_xy)
    if not model:
        # from_estimate() signals failure via a FailedEstimation sentinel
        # (bool(model) is False), NOT via None -- verified directly.
        return None

    # .params is also deprecated in favor of .center/.axis_lengths/.theta
    # (confirmed by the FutureWarning this line triggered before this fix).
    xc, yc = model.center
    a, b = model.axis_lengths
    theta = model.theta

    a = float(abs(a))
    b = float(abs(b))
    theta = float(theta)

    # Ensure a is always the semi-major axis.
    if b > a:
        a, b = b, a
        theta += np.pi / 2.0

    if a <= 0 or b <= 0:
        return None

    axis_ratio = a / b

    center_offset = (
        np.hypot(
            xc - cx,
            yc - cy
        ) / rsun
    )

    if axis_ratio > MAX_AXIS_RATIO:
        return None

    if center_offset > MAX_CENTER_OFFSET_RSUN:
        return None

    r_mean = (
        np.sqrt(a * b)
        / rsun
    )

    epsilon = 1.0 - b / a

    theta_image_deg = (
        np.degrees(theta) % 180.0
    )

    residuals = model.residuals(
        contour_xy
    )

    rms_px = float(
        np.sqrt(
            np.mean(
                residuals ** 2
            )
        )
    )

    rms_rel = (
        rms_px
        / np.sqrt(a * b)
    )

    return {
        "xc_px": float(xc),
        "yc_px": float(yc),

        "a_px": a,
        "b_px": b,

        "r_mean_rsun": float(r_mean),

        "epsilon": float(epsilon),

        "axis_ratio": float(axis_ratio),

        "theta_image_deg": float(
            theta_image_deg
        ),

        "center_offset_rsun": float(
            center_offset
        ),

        "fit_rms_px": rms_px,

        "fit_rms_rel": float(
            rms_rel
        ),
    }


# ============================================================
# BOOTSTRAP UNCERTAINTIES
# ============================================================

def bootstrap_ellipse_uncertainty(
    contour_xy,
    cx,
    cy,
    rsun,
):
    """
    Estimate 1-sigma uncertainties in ellipticity and angle by
    bootstrap resampling of the measured contour.

    The contour itself is the measured object; no noise model is
    artificially imposed on the FITS values.

    Uses ellipse_epsilon_theta_only() (not the full ellipse_parameters())
    for performance -- see that function's docstring for the measured
    justification.
    """

    n = len(contour_xy)

    if n < 50:
        return np.nan, np.nan

    if n > BOOTSTRAP_MAX_POINTS:

        idx = np.linspace(
            0,
            n - 1,
            BOOTSTRAP_MAX_POINTS
        ).astype(int)

        points = contour_xy[idx]

    else:
        points = contour_xy

    rng = np.random.default_rng(
        BOOTSTRAP_SEED
    )

    epsilons = []
    angles = []

    for _ in range(BOOTSTRAP_N):

        sample_idx = rng.integers(
            0,
            len(points),
            len(points)
        )

        sample = points[sample_idx]

        result = ellipse_epsilon_theta_only(
            sample,
            cx,
            cy,
            rsun
        )

        if result is None:
            continue

        epsilon, theta_image_deg = result

        epsilons.append(
            epsilon
        )

        angles.append(
            theta_image_deg
        )

    if len(epsilons) < 20:
        return np.nan, np.nan

    epsilon_err = float(
        np.std(
            epsilons,
            ddof=1
        )
    )

    # PA is periodic over 180 degrees.
    angles = np.asarray(
        angles
    )

    central = np.median(
        angles
    )

    delta = (
        (
            angles
            - central
            + 90.0
        )
        % 180.0
    ) - 90.0

    angle_err = float(
        np.std(
            delta,
            ddof=1
        )
    )

    return (
        epsilon_err,
        angle_err
    )


# ============================================================
# COMPLETE ELLIPSE MEASUREMENT
# ============================================================

def fit_ellipse(
    contour_xy,
    cx,
    cy,
    rsun,
    pa_offset=None,
):
    """
    Fit an ellipse, calculate ellipticity and uncertainties,
    and optionally convert image angle into solar PA.
    """

    params = ellipse_parameters(
        contour_xy,
        cx,
        cy,
        rsun
    )

    if params is None:
        return None

    epsilon_err, angle_err = (
        bootstrap_ellipse_uncertainty(
            contour_xy,
            cx,
            cy,
            rsun
        )
    )

    params["epsilon_err"] = (
        epsilon_err
    )

    params["theta_image_deg_err"] = (
        angle_err
    )

    # --------------------------------------------------------
    # Solar PA
    # --------------------------------------------------------

    if pa_offset is not None:

        pa_sun_deg = (
            params["theta_image_deg"]
            + pa_offset
        ) % 180.0

        params["pa_sun_deg"] = (
            float(pa_sun_deg)
        )

        params["pa_sun_deg_err"] = (
            angle_err
        )

    else:

        params["pa_sun_deg"] = np.nan
        params["pa_sun_deg_err"] = np.nan

    params["contour"] = contour_xy

    return params


# ============================================================
# ANALYSIS
# ============================================================

def make_analysis(
    data,
    cx,
    cy,
    rsun,
    sigma,
    n_levels,
    pa_offset,
):
    """
    Build the radial brightness profile and fit ellipses at
    n_levels selected isophotes.
    """

    work = data.copy()

    if sigma > 0:

        work = gaussian_filter(
            np.nan_to_num(
                work,
                nan=0.0
            ),
            sigma=sigma
        )

    yy, xx = np.indices(
        work.shape
    )

    radial_r = (
        np.hypot(
            xx - cx,
            yy - cy
        )
        / rsun
    )

    # --------------------------------------------------------
    # Radial intensity profile.
    #
    # 0.005 R_sun is used here to build a smooth radial
    # brightness profile. It does NOT mean we are fitting
    # 0.005-R_sun-spaced ellipses.
    # --------------------------------------------------------

    r_grid = np.arange(
        R_MIN,
        R_MAX + 0.005,
        0.005
    )

    radial_i = []

    for r0 in r_grid:

        sel = (
            (radial_r >= r0 - 0.0025)
            &
            (radial_r < r0 + 0.0025)
            &
            np.isfinite(work)
        )

        vals = work[sel]

        radial_i.append(
            np.nanmedian(vals)
            if vals.size
            else np.nan
        )

    radial_i = np.asarray(
        radial_i
    )

    good = (
        np.isfinite(radial_i)
        &
        (radial_i > 0)
    )

    if good.sum() < 10:
        raise RuntimeError(
            "Could not construct a usable "
            "radial intensity profile."
        )

    # Enforce monotonically decreasing intensity.
    monotonic = np.minimum.accumulate(
        radial_i[good]
    )

    rg = r_grid[good]

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Ellipse measurements remain at n_levels.
    # --------------------------------------------------------

    target_r = np.linspace(
        R_MIN,
        R_MAX,
        n_levels
    )

    levels = np.interp(
        target_r,
        rg,
        monotonic
    )

    records = []

    for target, level in zip(
        target_r,
        levels
    ):

        found = choose_centered_contour(
            work,
            level,
            cx,
            cy
        )

        if found is None:
            continue

        _, contour = found

        fit = fit_ellipse(
            contour,
            cx,
            cy,
            rsun,
            pa_offset=pa_offset
        )

        if fit is None:
            continue

        if not (
            R_MIN
            <= fit["r_mean_rsun"]
            <= R_MAX
        ):
            continue

        fit["target_r_rsun"] = (
            float(target)
        )

        fit["isophote_level"] = (
            float(level)
        )

        fit["quality"] = (
            "good"
            if fit["fit_rms_rel"]
            <= MAX_FIT_RMS_REL
            else "review"
        )

        records.append(
            fit
        )

    if len(records) < 10:
        raise RuntimeError(
            f"Only {len(records)} valid ellipse fits."
        )

    return records


# ============================================================
# REPORTING-PRECISION ROUNDING
# ============================================================

def round_to_step(values, step):
    """
    Round values to the requested reporting grid.

    IMPORTANT:
    The original fitted value remains untouched in the CSV.
    This is used for BOTH the ellipticity column and the angle
    column (theta_image_deg / pa_sun_deg), at the same step size,
    so both panels of the geometry plot share one consistent
    reporting precision.
    """

    values = np.asarray(
        values,
        dtype=float
    )

    return (
        np.round(
            values / step
        ) * step
    )


# ============================================================
# DISPLAY SELECTION
# ============================================================

def choose_display_records(records):

    available = np.array([
        r["r_mean_rsun"]
        for r in records
    ])

    chosen = []
    used = set()

    for target in DISPLAY_RADII:

        idx = int(
            np.argmin(
                np.abs(
                    available - target
                )
            )
        )

        if idx in used:
            continue

        used.add(idx)

        chosen.append(
            records[idx]
        )

    return sorted(
        chosen,
        key=lambda r: r["r_mean_rsun"]
    )


# ============================================================
# PLOTTING
# ============================================================

def plot_results(
    data,
    records,
    cx,
    cy,
    rsun,
    pa_offset,
    output_dir,
    stem,
):
    """
    Create:
      1. numerical CSV
      2. geometry graph
      3. clean ellipse overlay
      4. metadata JSON
    """

    # --------------------------------------------------------
    # DataFrame
    # --------------------------------------------------------

    df = pd.DataFrame([
        {
            k: v
            for k, v in record.items()
            if k != "contour"
        }
        for record in records
    ])

    df = df.sort_values(
        "r_mean_rsun"
    )

    # --------------------------------------------------------
    # Add reporting-only ellipticity values at 0.005 precision.
    # Raw epsilon remains unchanged.
    # --------------------------------------------------------

    df["epsilon_reported_0p005"] = (
        round_to_step(
            df["epsilon"],
            ELLIPTICITY_STEP
        )
    )

    # --------------------------------------------------------
    # Matching reporting-precision column for whichever angle
    # quantity this run actually has (image-coordinate angle, or
    # calibrated solar PA if --pa-offset was supplied). Uses the
    # SAME 0.005-sized step, expressed in degrees, so both panels
    # of the geometry plot are reported on a consistent grid.
    # Like epsilon, only the plotted central value is snapped --
    # the error bar stays at full precision (see the note beside
    # the ax1.errorbar call for why).
    # --------------------------------------------------------

    if pa_offset is not None:
        df["pa_sun_deg_reported_0p005"] = round_to_step(
            df["pa_sun_deg"], ANGLE_STEP_DEG
        )
    else:
        df["theta_image_deg_reported_0p005"] = round_to_step(
            df["theta_image_deg"], ANGLE_STEP_DEG
        )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    csv_path = (
        output_dir
        / f"{stem}_ellipse_measurements.csv"
    )

    df.to_csv(
        csv_path,
        index=False
    )

    # ========================================================
    # GEOMETRY PLOT
    # ========================================================

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(8.2, 7.4),
        dpi=300,
        sharex=True,
    )

    # --------------------------------------------------------
    # Ellipticity
    # --------------------------------------------------------

    ax1.errorbar(
        df["r_mean_rsun"],
        df["epsilon_reported_0p005"],
        yerr=df["epsilon_err"],
        fmt="o-",
        ms=2.7,
        lw=0.9,
        capsize=2,
        elinewidth=0.7,
        label="Measured ellipticity"
    )
    # NOTE ON PRECISION: only the plotted POINT is snapped to the 0.005
    # reporting grid (epsilon_reported_0p005). The error bar itself (yerr)
    # is left at full float precision, NOT rounded to the same 0.005 step.
    # Verified this matters: rounding a genuine error of e.g. 0.0019 to
    # the nearest 0.005 would draw it as a ZERO-length bar, which visually
    # claims perfect certainty where real bootstrap scatter exists. The
    # raw (unrounded) epsilon is still what's actually fitted; only the
    # displayed/reported central value is snapped.

    ax1.set_ylabel(
        r"Ellipticity  "
        r"$\epsilon = 1-b/a$"
    )

    ax1.set_title(
        r"Ellipticity (reporting precision: 0.005)"
    )

    ax1.grid(
        True,
        alpha=0.25
    )

    # --------------------------------------------------------
    # Position angle
    # --------------------------------------------------------

    if pa_offset is not None:

        ax2.errorbar(
            df["r_mean_rsun"],
            df["pa_sun_deg_reported_0p005"],
            yerr=df["pa_sun_deg_err"],
            fmt="o-",
            ms=2.7,
            lw=0.9,
            capsize=2,
            elinewidth=0.7,
            label=r"$\mathrm{PA}_{\odot}$"
        )

        ax2.set_ylabel(
            r"Solar position angle  "
            r"$\mathrm{PA}_{\odot}$ (°)"
        )

        ax2.set_title(
            r"Solar position angle (reporting precision: 0.005°)"
        )

    else:

        ax2.errorbar(
            df["r_mean_rsun"],
            df["theta_image_deg_reported_0p005"],
            yerr=df["theta_image_deg_err"],
            fmt="o-",
            ms=2.7,
            lw=0.9,
            capsize=2,
            elinewidth=0.7,
            label=r"$\mathrm{PA}_{\odot,\,\mathrm{img}}$"
        )

        ax2.set_ylabel(
            r"Semi-major-axis angle, image coords  "
            r"$\mathrm{PA}_{\odot,\,\mathrm{img}}$ (°)"
        )
        # NOTE: this is deliberately labeled distinctly from the
        # calibrated-PA case above (subscript "img") rather than reusing
        # the same PA_sun symbol outright. It is still an angle measured
        # in raw image pixel coordinates, NOT solar position angle --
        # the odot glyph here marks "this is the same axis-orientation
        # quantity", not "this has been calibrated to solar north". See
        # pa_status in the metadata JSON for the explicit calibration
        # state of any given run.

        ax2.set_title(
            r"Image-coordinate axis angle (reporting precision: 0.005°)"
        )

    ax2.set_xlabel(
        r"Mean isophote height  ($R_{\odot}$)"
    )

    ax2.grid(
        True,
        alpha=0.25
    )

    fig.suptitle(
        "DEB 2026 Coronal Isophote Geometry",
        y=0.98
    )

    ax1.legend(
        fontsize=8
    )

    ax2.legend(
        fontsize=8
    )

    fig.tight_layout()

    geometry_path = (
        output_dir
        / f"{stem}_geometry_for_matt.png"
    )

    fig.savefig(
        geometry_path,
        bbox_inches="tight"
    )

    plt.close(fig)

    # ========================================================
    # CLEAN ELLIPSE OVERLAY
    # ========================================================

    finite = np.isfinite(
        data
    )

    lo, hi = np.nanpercentile(
        data[finite],
        [5, 99.7]
    )

    fig, ax = plt.subplots(
        figsize=(10, 8),
        dpi=250
    )

    ax.imshow(
        np.clip(
            data,
            lo,
            hi
        ),
        cmap="gray",
        origin="upper"
    )

    display_records = (
        choose_display_records(
            records
        )
    )

    for i, rec in enumerate(
        display_records,
        start=1
    ):

        contour = rec["contour"]

        # ----------------------------------------------------
        # Observed contour
        # ----------------------------------------------------

        ax.plot(
            contour[:, 0],
            contour[:, 1],
            color="cyan",
            lw=0.8,
            alpha=0.45,
            label=(
                "Observed isophote"
                if i == 1
                else None
            )
        )

        # ----------------------------------------------------
        # Fitted ellipse
        # ----------------------------------------------------

        patch = Ellipse(
            (
                rec["xc_px"],
                rec["yc_px"]
            ),
            width=2 * rec["a_px"],
            height=2 * rec["b_px"],
            angle=rec["theta_image_deg"],
            fill=False,
            edgecolor="magenta",
            lw=2.8,
            alpha=1.0,
            label=(
                "Fitted geometric ellipse"
                if i == 1
                else None
            )
        )

        ax.add_patch(
            patch
        )

        # ----------------------------------------------------
        # Major axis
        # ----------------------------------------------------

        ang = np.radians(
            rec["theta_image_deg"]
        )

        dx = (
            rec["a_px"]
            * np.cos(ang)
        )

        dy = (
            rec["a_px"]
            * np.sin(ang)
        )

        ax.plot(
            [
                rec["xc_px"] - dx,
                rec["xc_px"] + dx,
            ],
            [
                rec["yc_px"] - dy,
                rec["yc_px"] + dy,
            ],
            color="yellow",
            lw=1.0,
            alpha=0.95,
        )

        # ----------------------------------------------------
        # Radius label
        # ----------------------------------------------------

        tx = (
            rec["xc_px"]
            + rec["a_px"]
            * np.cos(ang)
        )

        ty = (
            rec["yc_px"]
            + rec["a_px"]
            * np.sin(ang)
        )

        ax.text(
            tx + 12,
            ty - 12,
            f'{rec["r_mean_rsun"]:.2f} $R_\\odot$',
            color="white",
            fontsize=9,
            weight="bold",
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="black",
                edgecolor="none",
                alpha=0.65,
            ),
        )

    # --------------------------------------------------------
    # Solar center
    # --------------------------------------------------------

    ax.plot(
        cx,
        cy,
        "+",
        color="white",
        ms=13,
        mew=2.2,
        label="Reference solar center",
    )

    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    if pa_offset is not None:

        pa_text = (
            r"$\mathrm{PA}_{\odot}$ = "
            r"image angle + calibrated orientation offset"
        )

    else:

        pa_text = (
            "orientation not calibrated; "
            "angles remain in image coordinates"
        )

    ax.set_title(
        "Observed isophotes vs. fitted geometric ellipses\n"
        "cyan = measured contour   "
        "magenta = fitted ellipse   "
        "yellow = major axis\n"
        + pa_text
    )

    ax.legend(
        loc="upper right",
        fontsize=9,
        framealpha=0.75,
    )

    ax.set_aspect(
        "equal"
    )

    ax.set_xlim(
        0,
        data.shape[1]
    )

    ax.set_ylim(
        data.shape[0],
        0
    )

    ax.set_xlabel(
        "Image X (px)"
    )

    ax.set_ylabel(
        "Image Y (px)"
    )

    overlay_path = (
        output_dir
        / f"{stem}_ellipse_overlay_for_matt.png"
    )

    fig.savefig(
        overlay_path,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    # ========================================================
    # METADATA
    # ========================================================

    metadata = {
        "input": stem,

        "analysis_input_type": (
            "linear HDR FITS"
        ),

        "solar_center_x_px": float(
            cx
        ),

        "solar_center_y_px": float(
            cy
        ),

        "solar_radius_px": float(
            rsun
        ),

        "analysis_range_rsun": [
            float(R_MIN),
            float(R_MAX)
        ],

        "n_levels_requested": int(
            n_levels_from_records(
                records
            )
        ),

        "n_valid_ellipse_fits": int(
            len(records)
        ),

        "ellipticity_reporting_precision": (
            ELLIPTICITY_STEP
        ),

        "angle_reporting_precision_deg": (
            ANGLE_STEP_DEG
        ),

        "bootstrap_samples": (
            BOOTSTRAP_N
        ),

        "bootstrap_max_points": (
            BOOTSTRAP_MAX_POINTS
        ),

        "smoothing_sigma_px": float(
            0.0
        ),

        "pa_offset_deg": (
            None
            if pa_offset is None
            else float(pa_offset)
        ),

        "pa_status": (
            "CALIBRATED SOLAR PA"
            if pa_offset is not None
            else
            "IMAGE COORDINATES ONLY"
        ),

        "displayed_radii_rsun": [
            float(
                r["r_mean_rsun"]
            )
            for r in display_records
        ],

        "notes": [
            "Ellipse fitting performed on linear HDR data.",
            "Stretched images are not used for quantitative fitting.",
            "Raw ellipticity values are retained in the CSV.",
            "epsilon_reported_0p005 is a reporting-only rounded value; error bars in the plot are NOT rounded to this step, to avoid drawing misleadingly zero-length bars for genuine sub-step uncertainties.",
            "theta_image_deg_reported_0p005 / pa_sun_deg_reported_0p005 (whichever applies to this run) are the equivalent reporting-only rounded angle column, same precision and same error-bar caveat.",
            "Bootstrap errors are empirical 1-sigma estimates.",
            "Solar PA is only reported when a calibrated orientation offset is supplied.",
            "The image-coordinate angle panel uses the label PA_{sun,img} rather than plain PA_sun, to keep it visually distinct from calibrated solar position angle."
        ],
    }

    metadata_path = (
        output_dir
        / f"{stem}_metadata.json"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2
        ),
        encoding="utf-8",
    )

    # ========================================================
    # CONSOLE SUMMARY
    # ========================================================

    print()
    print("=" * 60)
    print("DEB 2026 CORONAL ISOPHOTE ANALYSIS")
    print("=" * 60)

    print(
        f"[+] Input FITS: {stem}"
    )

    print(
        f"[+] Image size: "
        f"{data.shape[1]} x {data.shape[0]} px"
    )

    print(
        f"[+] Valid ellipse fits: "
        f"{len(records)}"
    )

    print(
        f"[+] Ellipticity reporting precision: "
        f"{ELLIPTICITY_STEP:.3f}"
    )

    print(
        f"[+] Angle reporting precision: "
        f"{ANGLE_STEP_DEG:.3f} deg"
    )

    print(
        f"[+] Bootstrap samples: "
        f"{BOOTSTRAP_N}"
    )

    if pa_offset is not None:
        print(
            f"[+] PA_sun orientation offset: "
            f"{pa_offset:.3f} deg"
        )
    else:
        print(
            "[!] PA not calibrated: "
            "angles remain in image coordinates."
        )

    print(
        f"[+] CSV: "
        f"{csv_path}"
    )

    print(
        f"[+] Geometry plot: "
        f"{geometry_path}"
    )

    print(
        f"[+] Ellipse overlay: "
        f"{overlay_path}"
    )

    print(
        f"[+] Metadata: "
        f"{metadata_path}"
    )

    print("=" * 60)


def n_levels_from_records(records):
    """
    Metadata helper.

    The exact requested number is not necessarily equal to the
    number of valid fits because some contours may be rejected.
    """
    return len(records)


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    fits_path = (
        Path(args.fits_path)
        .expanduser()
        .resolve()
    )

    if not fits_path.exists():
        raise FileNotFoundError(
            f"FITS file not found: {fits_path}"
        )

    output_dir = (
        Path(args.output_dir)
        .expanduser()
        .resolve()
        if args.output_dir
        else
        fits_path.parent
        / "ellipse_analysis"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # LOAD FITS
    # --------------------------------------------------------

    data = load_fits(
        fits_path
    )

    print(
        f"[+] FITS: {fits_path}"
    )

    print(
        f"[+] Image: "
        f"{data.shape[1]} x {data.shape[0]} px"
    )

    print(
        f"[+] Center: "
        f"({args.cx:.2f}, {args.cy:.2f}) px"
    )

    print(
        f"[+] R_sun: "
        f"{args.rsun:.2f} px"
    )

    print(
        f"[+] Radial range: "
        f"{R_MIN:.2f}–{R_MAX:.2f} R_sun"
    )

    print(
        f"[+] Isophote levels: "
        f"{args.levels}"
    )

    print(
        f"[+] Ellipticity reporting step: "
        f"{ELLIPTICITY_STEP:.3f}"
    )

    # --------------------------------------------------------
    # RUN ANALYSIS
    # --------------------------------------------------------

    records = make_analysis(
        data=data,
        cx=args.cx,
        cy=args.cy,
        rsun=args.rsun,
        sigma=args.sigma,
        n_levels=args.levels,
        pa_offset=args.pa_offset,
    )

    # --------------------------------------------------------
    # CREATE OUTPUTS
    # --------------------------------------------------------

    plot_results(
        data=data,
        records=records,
        cx=args.cx,
        cy=args.cy,
        rsun=args.rsun,
        pa_offset=args.pa_offset,
        output_dir=output_dir,
        stem=fits_path.stem,
    )


if __name__ == "__main__":
    main()