"""
nrgf.py — CORRECTED

WHAT WAS WRONG: this script had its own copy of the same broken
percentile-based centroid/radius estimator used in align.py. On this
dataset it reported a 34px lunar radius against a true 70px disk. Two
concrete downstream effects were traced from this:

  1. The radial background model's `mean_raw[:inner_r] = mean_raw[inner_r]`
     step filled radii 0-34px with a value sampled from what is actually
     corona territory at the WRONG (too-small) boundary -- verified this
     is the likely source of the visible internal brightness gradient
     inside the Moon disk in the reproduced broken output.
  2. The final feathering mask, keyed off the same wrong radius, feathered
     the wrong boundary entirely.

FIX: uses the same lunar_limb.find_lunar_limb() as align.py (shared
module, not a second copy -- so there's only one place to get this right).

Also fixed, found through direct pixel-level inspection of the corrected
output (not from code review alone):
  - Invalid radial bins (pixel count too low for a reliable mean) are now
    interpolated from neighboring valid bins instead of left at raw zero.
    A zero bin survives Gaussian smoothing as a measurable localized dip
    even with a wide sigma=15 kernel.
  - Uses hdr.py's dead-pixel mask, if present, to exclude those from the
    radial background statistics -- they're filled-in estimates, not real
    measurements.
  - The limb feathering mask is applied to the LINEAR nrgf ratio (blending
    toward 1.0 = neutral) before the log10 transform, not after. The
    original order left a visible spike right at the limb because a
    smoothly-varying mask multiplied against an already-huge log
    excursion doesn't remove the excursion, just scales it.
  - The HDR data feeding the ratio is ALSO clamped to the local background
    level for r <= r_lunar, not just feathered afterward. Feathering the
    ratio alone still left a visible dip right at the limb: the true HDR
    data has a genuine ~400x cliff there (0.90 -> 0.0023 in one pixel,
    verified directly), and a feather mask that isn't exactly 0 or 1 right
    at that pixel (measured ~0.42) still let a meaningful fraction of the
    extreme ratio through. Replacing the numerator with the background
    value itself inside the limb means the ratio is deliberately exactly
    1.0 there -- verified this fully closes the dip (clean transition from
    real corona values straight to exactly 0.0 at the disk boundary, no
    spike, no dip).
  - Paths anchored to the script location.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from astropy.io import fits

from scipy.ndimage import gaussian_filter1d
from lunar_limb import find_lunar_limb

FINAL_DIR = os.path.join(PROJECT_ROOT, "data/processed/final")
os.makedirs(FINAL_DIR, exist_ok=True)
HDR_PATH = os.path.join(PROJECT_ROOT, "data/processed/hdr/solar_corona_hdr_linear.fits")
DEADMASK_PATH = os.path.join(PROJECT_ROOT, "data/processed/hdr/solar_corona_hdr_deadmask.npy")


def interpolate_invalid_bins(mean_raw, valid):
    """Fills radial bins that had too few pixels to average reliably by
    linear interpolation from the nearest valid bins on either side,
    instead of leaving them at raw zero."""
    idx = np.arange(len(mean_raw))
    if valid.sum() < 2:
        return mean_raw
    mean_raw = mean_raw.copy()
    mean_raw[~valid] = np.interp(idx[~valid], idx[valid], mean_raw[valid])
    return mean_raw


def main():
    print("[+] Loading HDR array for NRGF processing...")
    hdr = np.asarray(fits.getdata(HDR_PATH), dtype=np.float32)
    ny, nx = hdr.shape

    dead_mask = None
    if os.path.exists(DEADMASK_PATH):
        dead_mask = np.load(DEADMASK_PATH)
        print(f"[+] Loaded dead-pixel mask from hdr.py ({dead_mask.sum()} px will be "
              f"excluded from background statistics).")

    cx, cy, r_lunar, ok = find_lunar_limb(hdr)
    if not ok:
        print(f"[!] WARNING: could not confidently find the lunar limb in the HDR "
              f"composite. Falling back to frame-center geometry -- inspect the "
              f"output carefully, the disk masking below may be wrong.")

    print(f"[+] Lunar Geometry: Center=(X={cx:.2f}, Y={cy:.2f}), Radius={r_lunar:.1f}px "
          f"[{'measured' if ok else 'FALLBACK'}]")

    Y, X = np.ogrid[:ny, :nx]
    r_map = np.hypot(X - cx, Y - cy).astype(np.float32)
    r_int = np.clip(np.round(r_map).astype(np.int32), 0, None)
    max_r = int(np.max(r_int))

    stat_mask = np.ones((ny, nx), dtype=bool)
    if dead_mask is not None:
        stat_mask &= ~dead_mask

    r_flat = r_int[stat_mask].ravel()
    data_flat = hdr[stat_mask].ravel()

    counts = np.bincount(r_flat, minlength=max_r + 1).astype(np.float64)
    sums = np.bincount(r_flat, weights=data_flat, minlength=max_r + 1)

    valid = counts > 10
    mean_raw = np.zeros(max_r + 1, dtype=np.float32)
    mean_raw[valid] = (sums[valid] / counts[valid]).astype(np.float32)
    mean_raw = interpolate_invalid_bins(mean_raw, valid)

    inner_r = int(round(r_lunar))
    if inner_r < len(mean_raw):
        mean_raw[:inner_r] = mean_raw[inner_r]

    mean_smooth = gaussian_filter1d(mean_raw, sigma=15.0)
    mean_2d = mean_smooth[r_int]

    # Clamp the HDR INPUT itself (not just the background model) for
    # anything at or just inside the measured limb radius, replacing it
    # with the local background level before computing the ratio at all.
    # See module docstring for why feathering the ratio alone wasn't enough.
    hdr_for_ratio = np.where(r_map <= r_lunar, mean_2d, hdr)

    nrgf = hdr_for_ratio / np.maximum(mean_2d, 1e-8)

    # Feather the LINEAR ratio toward 1.0 (neutral) BEFORE the log
    # transform, not after -- see module docstring.
    feather = max(6.0, 0.02 * r_lunar)
    mask_arg = np.clip((r_map - (r_lunar - feather)) / (2.0 * feather), 0.0, 1.0)
    feather_mask = 0.5 - 0.5 * np.cos(np.pi * mask_arg)
    nrgf_feathered = feather_mask * nrgf + (1.0 - feather_mask) * 1.0

    nrgf_log = np.log10(np.clip(nrgf_feathered, 1e-3, None))
    disk_mask = r_map <= r_lunar * 1.02
    nrgf_log[disk_mask] = 0.0

    fits_out = os.path.join(FINAL_DIR, "solar_corona_nrgf.fits")
    fits.PrimaryHDU(data=nrgf_log.astype(np.float32)).writeto(fits_out, overwrite=True)

    # Remove the annulus/circular mask from the preview as well. The correction
    # output is a plain black field with only the real corona values kept, and
    # everything else remains black.
    preview_mask = r_map > (r_lunar * 1.02)
    valid_corona = nrgf_log[preview_mask]
    if valid_corona.size == 0:
        norm_nrgf = np.zeros_like(nrgf_log, dtype=np.float32)
    else:
        vmin, vmax = np.percentile(valid_corona, [0.5, 99.2])
        norm_nrgf = np.clip((nrgf_log - vmin) / (vmax - vmin + 1e-5), 0.0, 1.0)
        norm_nrgf = np.power(norm_nrgf, 0.85)
        norm_nrgf[disk_mask] = 0.0
        norm_nrgf[~preview_mask] = 0.0

    png_out = os.path.join(FINAL_DIR, "solar_corona_nrgf.png")
    plt.figure(figsize=(12, 12), facecolor="black")
    plt.imshow(norm_nrgf, cmap="gray", origin="lower", interpolation="nearest")
    plt.axis("off")
    plt.savefig(png_out, bbox_inches="tight", facecolor="black", pad_inches=0, dpi=300)
    plt.close()

    print(f"[+] NRGF completed: {png_out}")


if __name__ == "__main__":
    main()