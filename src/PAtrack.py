#!/usr/bin/env python3
"""
Track the Moon's motion through the DEB 2026 totality FITS sequence
and use the measured motion to resolve the camera orientation.

INPUT SEQUENCE
--------------
D:\\AIMAR\\Astronomía\\Program_Projects\\solar-corona-hdr-pipeline\\
data\\raw\\lights\\totality\\totality_2026-08-12_*_400.0ms.fits

REFERENCE MOON POSITION
-----------------------
solar_corona_hdr_linear.fits provides:

    MOON-X = 1561.145...
    MOON-Y = 1136.413...
    MOON-R = 341.889...

The script assumes that this position belongs to the same image
geometry as the sequence and uses it as the initial tracking position.

METHOD
------
For each 400 ms frame:

1. Estimate the image gradient.
2. Search around the previous Moon center.
3. Score candidate circular limbs near the known Moon radius.
4. Select the strongest circular edge.
5. Record the Moon center.
6. Fit Moon position as a function of time.
7. Derive the observed direction of lunar motion.

This is intended to resolve the two possible camera-orientation
solutions obtained from the single Moon position.

It does NOT silently choose a PA offset. It prints the results
and saves diagnostic information so the orientation can be verified.

Outputs
-------
moon_tracking.csv
moon_tracking_diagnostic.png
moon_tracking_summary.txt
"""

from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from astropy.io import fits
from scipy.ndimage import (
    gaussian_filter,
    sobel,
    map_coordinates,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

TOTALITY_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "lights"
    / "totality"
)

HDR_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "hdr"
    / "solar_corona_hdr_linear.fits"
)


# ============================================================
# KNOWN STARTING VALUES FROM YOUR HDR
# ============================================================

INITIAL_MOON_X = 1561.145186359174
INITIAL_MOON_Y = 1136.4134784792975
MOON_RADIUS = 341.8894408341663

# Assumed solar center used in your ellipse analysis.
SOLAR_CX = 1551.5
SOLAR_CY = 1145.5


# ============================================================
# TRACKING PARAMETERS
# ============================================================

# Search radius around the previous Moon position.
SEARCH_RADIUS_PX = 25.0

# Candidate-center spacing.
SEARCH_STEP_PX = 1.0

# Number of angular samples around the lunar limb.
N_ANGLE = 360

# Search several radii around the known Moon radius.
RADIUS_HALF_WIDTH = 8.0
RADIUS_STEP = 1.0

# Mild smoothing before calculating gradients.
GRADIENT_SIGMA = 1.5

# Reject measurements whose score is too weak relative to the
# first successful measurement.
MIN_RELATIVE_SCORE = 0.25

# Use only good detections in the final motion fit.
MIN_TRACK_POINTS = 5

# For robustness, optionally skip the first/last few frames.
EDGE_FRAMES_TO_IGNORE = 0


# ============================================================
# UTILITIES
# ============================================================

def parse_timestamp(filename):
    """
    Extract HHMMSS from filenames such as:

        totality_2026-08-12_182938_400.0ms.fits

    Returns seconds since midnight.
    """

    match = re.search(
        r"_([0-9]{6})_[0-9.]+ms\.fits$",
        filename.name,
        re.IGNORECASE
    )

    if not match:
        return np.nan

    hhmmss = match.group(1)

    hour = int(hhmmss[0:2])
    minute = int(hhmmss[2:4])
    second = int(hhmmss[4:6])

    return (
        hour * 3600
        + minute * 60
        + second
    )


def normalize_image(data):
    """
    Robustly normalize an image for tracking.

    This does not modify the original FITS.
    """

    data = np.asarray(
        data,
        dtype=np.float64
    )

    finite = np.isfinite(data)

    if not finite.any():
        raise ValueError(
            "Image contains no finite pixels."
        )

    lo, hi = np.nanpercentile(
        data[finite],
        [1.0, 99.5]
    )

    if hi <= lo:
        raise ValueError(
            "Image has insufficient dynamic range."
        )

    image = (
        np.clip(data, lo, hi)
        - lo
    ) / (hi - lo)

    return image


def load_image(path):
    data = fits.getdata(path)

    data = np.asarray(
        data,
        dtype=np.float64
    )

    data = np.squeeze(data)

    # The totality 400-ms frames are expected to be monochrome.
    if data.ndim != 2:
        raise ValueError(
            f"{path.name}: expected 2-D image, "
            f"got {data.shape}"
        )

    return normalize_image(data)


def circular_samples(
    cx,
    cy,
    radius,
    n=N_ANGLE
):
    """
    Pixel coordinates sampled around a circle.

    Returns x, y arrays.
    """

    theta = np.linspace(
        0,
        2 * np.pi,
        n,
        endpoint=False
    )

    x = (
        cx
        + radius * np.cos(theta)
    )

    y = (
        cy
        + radius * np.sin(theta)
    )

    return x, y


def circular_edge_score(
    gradient,
    cx,
    cy,
    radius
):
    """
    Mean gradient magnitude along a circular limb.

    A real lunar edge should produce a strong response around
    most of the circumference.
    """

    x, y = circular_samples(
        cx,
        cy,
        radius
    )

    coords = np.vstack(
        [
            y,
            x
        ]
    )

    values = map_coordinates(
        gradient,
        coords,
        order=1,
        mode="nearest"
    )

    # Robust score: high mean while suppressing a few extreme
    # single-pixel features.
    return float(
        np.percentile(
            values,
            75
        )
    )


def track_moon_center(
    image,
    previous_x,
    previous_y,
    moon_radius
):
    """
    Search for the lunar limb near the previous position.
    """

    smoothed = gaussian_filter(
        image,
        GRADIENT_SIGMA
    )

    gx = sobel(
        smoothed,
        axis=1
    )

    gy = sobel(
        smoothed,
        axis=0
    )

    gradient = np.hypot(
        gx,
        gy
    )

    # Keep candidates inside the image.
    h, w = image.shape

    x_values = np.arange(
        previous_x - SEARCH_RADIUS_PX,
        previous_x + SEARCH_RADIUS_PX + SEARCH_STEP_PX,
        SEARCH_STEP_PX
    )

    y_values = np.arange(
        previous_y - SEARCH_RADIUS_PX,
        previous_y + SEARCH_RADIUS_PX + SEARCH_STEP_PX,
        SEARCH_STEP_PX
    )

    radii = np.arange(
        moon_radius - RADIUS_HALF_WIDTH,
        moon_radius + RADIUS_HALF_WIDTH + RADIUS_STEP,
        RADIUS_STEP
    )

    best_score = -np.inf
    best = None

    for cy in y_values:

        if (
            cy - moon_radius - 12 < 0
            or
            cy + moon_radius + 12 >= h
        ):
            continue

        for cx in x_values:

            if (
                cx - moon_radius - 12 < 0
                or
                cx + moon_radius + 12 >= w
            ):
                continue

            for radius in radii:

                score = circular_edge_score(
                    gradient,
                    cx,
                    cy,
                    radius
                )

                if score > best_score:

                    best_score = score

                    best = (
                        float(cx),
                        float(cy),
                        float(radius),
                        float(score)
                    )

    return best


# ============================================================
# MOTION FIT
# ============================================================

def fit_linear_motion(df):
    """
    Fit X(t) and Y(t) independently.

    Returns velocity in pixels / second and motion direction
    in image coordinates.
    """

    good = df["good"].to_numpy()

    if good.sum() < MIN_TRACK_POINTS:
        raise RuntimeError(
            "Not enough valid Moon detections "
            "for a motion fit."
        )

    t = df.loc[good, "time_s"].to_numpy()

    x = df.loc[good, "moon_x"].to_numpy()
    y = df.loc[good, "moon_y"].to_numpy()

    px = np.polyfit(
        t,
        x,
        1
    )

    py = np.polyfit(
        t,
        y,
        1
    )

    vx = float(
        px[0]
    )

    vy = float(
        py[0]
    )

    speed = float(
        np.hypot(
            vx,
            vy
        )
    )

    angle = (
        np.degrees(
            np.arctan2(
                vy,
                vx
            )
        )
        % 360.0
    )

    return {
        "vx_px_s": vx,
        "vy_px_s": vy,
        "speed_px_s": speed,
        "image_motion_angle_deg": float(angle),
        "x_fit": px,
        "y_fit": py,
    }


# ============================================================
# RUN
# ============================================================

def main():

    print("=" * 72)
    print("DEB 2026 MOON TRACKING / CAMERA ORIENTATION")
    print("=" * 72)

    print(
        f"Totality directory:\n{TOTALITY_DIR}"
    )

    if not TOTALITY_DIR.exists():
        raise FileNotFoundError(
            f"Directory not found:\n{TOTALITY_DIR}"
        )

    if not HDR_PATH.exists():
        raise FileNotFoundError(
            f"HDR file not found:\n{HDR_PATH}"
        )

    # --------------------------------------------------------
    # Find 400 ms totality frames
    # --------------------------------------------------------

    files = sorted(
        TOTALITY_DIR.glob(
            "totality_2026-08-12_*_400.0ms.fits"
        ),
        key=parse_timestamp
    )

    if not files:
        raise FileNotFoundError(
            "No 400 ms totality FITS files found."
        )

    print(
        f"\nFound {len(files)} 400 ms frames."
    )

    print(
        f"First: {files[0].name}"
    )

    print(
        f"Last : {files[-1].name}"
    )

    # --------------------------------------------------------
    # Check starting Moon information in HDR
    # --------------------------------------------------------

    with fits.open(
        HDR_PATH,
        memmap=True
    ) as hdul:

        hdr = hdul[0].header

    hdr_moon_x = float(
        hdr.get(
            "MOON-X",
            INITIAL_MOON_X
        )
    )

    hdr_moon_y = float(
        hdr.get(
            "MOON-Y",
            INITIAL_MOON_Y
        )
    )

    hdr_moon_r = float(
        hdr.get(
            "MOON-R",
            MOON_RADIUS
        )
    )

    print()
    print("Reference Moon position from HDR:")
    print(
        f"  X = {hdr_moon_x:.3f} px"
    )
    print(
        f"  Y = {hdr_moon_y:.3f} px"
    )
    print(
        f"  R = {hdr_moon_r:.3f} px"
    )

    # --------------------------------------------------------
    # Load first frame
    # --------------------------------------------------------

    print()
    print(
        f"Loading first frame: "
        f"{files[0].name}"
    )

    first_image = load_image(
        files[0]
    )

    h, w = first_image.shape

    print(
        f"Image size: {w} x {h}"
    )

    # The HDR Moon position is our initial guide.
    previous_x = hdr_moon_x
    previous_y = hdr_moon_y

    rows = []

    # --------------------------------------------------------
    # Track each frame
    # --------------------------------------------------------

    for i, path in enumerate(files):

        timestamp = parse_timestamp(
            path
        )

        if not np.isfinite(timestamp):
            continue

        try:

            image = load_image(
                path
            )

            result = track_moon_center(
                image,
                previous_x,
                previous_y,
                hdr_moon_r
            )

            if result is None:

                print(
                    f"[{i+1:3d}/{len(files)}] "
                    f"{path.name}: FAILED"
                )

                rows.append({
                    "filename": path.name,
                    "time_s": timestamp,
                    "moon_x": np.nan,
                    "moon_y": np.nan,
                    "moon_radius": np.nan,
                    "score": np.nan,
                    "good": False,
                })

                continue

            x, y, radius, score = result

            # First-frame score establishes the scale.
            if not rows:

                reference_score = score

            else:

                reference_score = max(
                    reference_score,
                    score
                )

            good = (
                score
                >= reference_score
                * MIN_RELATIVE_SCORE
            )

            rows.append({
                "filename": path.name,
                "time_s": timestamp,
                "moon_x": x,
                "moon_y": y,
                "moon_radius": radius,
                "score": score,
                "good": good,
            })

            if good:

                previous_x = x
                previous_y = y

                print(
                    f"[{i+1:3d}/{len(files)}] "
                    f"{path.name}  "
                    f"Moon=({x:.2f}, {y:.2f})  "
                    f"R={radius:.2f}  "
                    f"score={score:.5g}"
                )

            else:

                print(
                    f"[{i+1:3d}/{len(files)}] "
                    f"{path.name}: WEAK "
                    f"Moon=({x:.2f}, {y:.2f})"
                )

        except Exception as exc:

            print(
                f"[{i+1:3d}/{len(files)}] "
                f"{path.name}: ERROR: {exc}"
            )

            rows.append({
                "filename": path.name,
                "time_s": timestamp,
                "moon_x": np.nan,
                "moon_y": np.nan,
                "moon_radius": np.nan,
                "score": np.nan,
                "good": False,
            })

    # --------------------------------------------------------
    # DataFrame
    # --------------------------------------------------------

    df = pd.DataFrame(rows)

    if len(df) == 0:
        raise RuntimeError(
            "No frames were processed."
        )

    df["elapsed_s"] = (
        df["time_s"]
        - df["time_s"].min()
    )

    # --------------------------------------------------------
    # Save raw tracking results
    # --------------------------------------------------------

    csv_path = (
        TOTALITY_DIR
        / "moon_tracking.csv"
    )

    df.to_csv(
        csv_path,
        index=False
    )

    print()
    print(
        f"[+] Tracking CSV: {csv_path}"
    )

    # --------------------------------------------------------
    # Fit motion
    # --------------------------------------------------------

    motion = fit_linear_motion(
        df
    )

    vx = motion["vx_px_s"]
    vy = motion["vy_px_s"]
    speed = motion["speed_px_s"]
    angle = motion["image_motion_angle_deg"]

    print()
    print("=" * 72)
    print("MEASURED MOON MOTION")
    print("=" * 72)

    print(
        f"Valid detections : "
        f"{int(df['good'].sum())} / {len(df)}"
    )

    print(
        f"Vx = {vx:+.8f} px/s"
    )

    print(
        f"Vy = {vy:+.8f} px/s"
    )

    print(
        f"Speed = {speed:.8f} px/s"
    )

    print(
        f"Image motion angle = "
        f"{angle:.6f}°"
    )

    # --------------------------------------------------------
    # Compare first / last detected positions
    # --------------------------------------------------------

    good_df = df[
        df["good"]
    ].copy()

    first = good_df.iloc[0]
    last = good_df.iloc[-1]

    dx = (
        last["moon_x"]
        - first["moon_x"]
    )

    dy = (
        last["moon_y"]
        - first["moon_y"]
    )

    print()
    print(
        "Observed first → last displacement:"
    )

    print(
        f"ΔX = {dx:+.6f} px"
    )

    print(
        f"ΔY = {dy:+.6f} px"
    )

    # --------------------------------------------------------
    # Retrieve previous candidate orientation results
    # --------------------------------------------------------
    #
    # These came from your single-frame Moon/Sun geometry:
    #
    #   candidate A = +75.140866°
    #   candidate B = 348.557495°
    #
    # We print them here so the tracking result can be compared
    # against the eventual ephemeris calculation.
    # --------------------------------------------------------

    candidate_a = 75.140866
    candidate_b = 348.557495

    print()
    print("=" * 72)
    print("PREVIOUS SINGLE-FRAME ORIENTATION CANDIDATES")
    print("=" * 72)

    print(
        f"Candidate A = "
        f"{candidate_a:.6f}°"
    )

    print(
        f"Candidate B = "
        f"{candidate_b:.6f}°"
    )

    print()
    print(
        "The measured Moon motion is now available to determine "
        "which handedness is correct."
    )

    # --------------------------------------------------------
    # Diagnostic plots
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(10, 7),
        dpi=250
    )

    good = df["good"]

    ax.plot(
        df.loc[good, "elapsed_s"],
        df.loc[good, "moon_x"],
        "o-",
        ms=3,
        lw=1,
        label="Moon X"
    )

    ax.plot(
        df.loc[good, "elapsed_s"],
        df.loc[good, "moon_y"],
        "o-",
        ms=3,
        lw=1,
        label="Moon Y"
    )

    ax.set_xlabel(
        "Elapsed time (s)"
    )

    ax.set_ylabel(
        "Moon centre (px)"
    )

    ax.set_title(
        "DEB 2026 Moon tracking — image coordinates"
    )

    ax.grid(
        True,
        alpha=0.25
    )

    ax.legend()

    tracking_plot = (
        TOTALITY_DIR
        / "moon_tracking_xy.png"
    )

    fig.savefig(
        tracking_plot,
        bbox_inches="tight"
    )

    plt.close(fig)

    # --------------------------------------------------------
    # X-Y trajectory
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(8, 8),
        dpi=250
    )

    ax.plot(
        df.loc[good, "moon_x"],
        df.loc[good, "moon_y"],
        "o-",
        ms=3,
        lw=1
    )

    ax.scatter(
        [first["moon_x"]],
        [first["moon_y"]],
        s=70,
        marker="o",
        label="First tracked frame"
    )

    ax.scatter(
        [last["moon_x"]],
        [last["moon_y"]],
        s=70,
        marker="x",
        label="Last tracked frame"
    )

    ax.set_xlabel(
        "Image X (px)"
    )

    ax.set_ylabel(
        "Image Y (px)"
    )

    ax.set_title(
        "DEB 2026 measured lunar motion"
    )

    ax.invert_yaxis()

    ax.set_aspect(
        "equal"
    )

    ax.grid(
        True,
        alpha=0.25
    )

    ax.legend()

    trajectory_plot = (
        TOTALITY_DIR
        / "moon_tracking_trajectory.png"
    )

    fig.savefig(
        trajectory_plot,
        bbox_inches="tight"
    )

    plt.close(fig)

    # --------------------------------------------------------
    # Summary text
    # --------------------------------------------------------

    summary_path = (
        TOTALITY_DIR
        / "moon_tracking_summary.txt"
    )

    summary = f"""
DEB 2026 MOON TRACKING
======================

Sequence:
{files[0].name}
through
{files[-1].name}

Number of frames:
{len(files)}

Valid tracked frames:
{int(df['good'].sum())}

Reference Moon centre:
X = {hdr_moon_x:.6f} px
Y = {hdr_moon_y:.6f} px

Reference Moon radius:
R = {hdr_moon_r:.6f} px

Measured lunar motion:

Vx = {vx:+.10f} px/s
Vy = {vy:+.10f} px/s
Speed = {speed:.10f} px/s

Image-coordinate motion angle:
{angle:.8f} deg

First → last displacement:

dX = {dx:+.8f} px
dY = {dy:+.8f} px

Previous single-frame camera-orientation candidates:

A = {candidate_a:.8f} deg
B = {candidate_b:.8f} deg

IMPORTANT:
The motion measurement resolves the image handedness only when
compared against an independently calculated lunar-motion vector.
Do NOT put candidate A or B directly into the ellipse script yet.
"""

    summary_path.write_text(
        summary.strip()
        + "\n",
        encoding="utf-8"
    )

    print()
    print("=" * 72)
    print("OUTPUTS")
    print("=" * 72)

    print(
        f"CSV:\n{csv_path}"
    )

    print(
        f"XY diagnostic:\n{tracking_plot}"
    )

    print(
        f"Trajectory:\n{trajectory_plot}"
    )

    print(
        f"Summary:\n{summary_path}"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()