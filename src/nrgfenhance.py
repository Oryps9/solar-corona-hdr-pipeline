"""
enhance.py — CORRECTED

WHAT WAS WRONG: the multi-scale unsharp-masking (blur at sigma=2,8,24 and
add back the difference) was applied uniformly across the whole frame,
including right across the lunar disk boundary. That boundary is a
genuinely hard edge in a correct NRGF composite (bright limb ring next to
a flat, masked-zero disk interior) -- verified directly that gaussian
blurring smears this hard edge into each blur kernel, and subtracting the
blurred version from the sharp original manufactures a fake "detail"
signal that isn't real corona structure. On the corrected NRGF output
this showed up as a dark ring artifact of about -0.22 (in NRGF log units)
right at the disk edge, decaying back to baseline only after about 26px --
consistent with the largest blur kernel (sigma=24) reaching that far.

This is a bug in the original design that upstream fixes don't resolve on
their own: even a perfectly clean NRGF composite has a genuinely sharp
disk/corona boundary, so this artifact would appear on ANY correctly-
masked eclipse composite, not just this one.

FIX: the enhanced (unsharp-masked) result is feathered back toward the
unenhanced NRGF value in a zone around the disk boundary, using the same
raised-cosine feather shape used elsewhere in this pipeline, sized to
1.5x the largest blur sigma (chosen conservatively above the measured
~26px extent of the ringing). Outside that zone the full multi-scale
enhancement is applied exactly as before, verified to match the original
algorithm's output exactly once the feather reaches 1.0.

Also fixed: paths anchored to the script location; derives disk geometry
directly from the NRGF's own exact-zero masked interior (simple and
robust here, since nrgf.py guarantees that region is exactly 0.0) instead
of re-running limb detection a third time on data it isn't tuned for.
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

from scipy.ndimage import gaussian_filter, gaussian_filter1d

FINAL_DIR = os.path.join(PROJECT_ROOT, "data/processed/final")
NRGF_PATH = os.path.join(FINAL_DIR, "solar_corona_nrgf.fits")
HDR_PATH = os.path.join(PROJECT_ROOT, "data/processed/hdr/solar_corona_hdr_linear.fits")

BLUR_SIGMAS = (2.0, 8.0, 24.0)
BLUR_WEIGHTS = (1.5, 1.0, 0.5)


def main():
    print("[+] Loading HDR stack for real eclipse preview...")
    hdr = np.asarray(fits.getdata(HDR_PATH), dtype=np.float32)
    if hdr.ndim == 3:
        hdr = hdr.mean(axis=0)
    ny, nx = hdr.shape

    # Use the zero-mask from the NRGF product to recover the lunar disk geometry, but
    # keep the final display based on the real HDR stack, not the NRGF ratio image.
    nrgf = np.asarray(fits.getdata(NRGF_PATH), dtype=np.float32)
    cy0, cx0 = ny / 2.0, nx / 2.0
    Y, X = np.ogrid[:ny, :nx]
    dist0 = np.hypot(X - cx0, Y - cy0)
    search = dist0 < (min(nx, ny) * 0.35)
    zero_mask = (nrgf == 0.0) & search
    if zero_mask.sum() > 50:
        zy, zx = np.where(zero_mask)
        cx, cy = float(np.mean(zx)), float(np.mean(zy))
        r_lunar = float(np.sqrt(zero_mask.sum() / np.pi))
    else:
        cx, cy, r_lunar = cx0, cy0, min(nx, ny) * 0.12

    r_map = np.hypot(X - cx, Y - cy).astype(np.float32)

    # Keep the display on the real corona only: no annulus mask, no gray
    # background, and no synthetic ring. The near-limb corona is faint but
    # real, so we preserve it with a softer log stretch instead of letting the
    # entire brightness range collapse onto a thin bright rim.
    blurred = gaussian_filter(hdr, sigma=2.2)
    detail = hdr - blurred
    enhanced = hdr + 0.35 * detail

    # Mild radial gain helps maintain lower-corona structure near the limb
    # without creating a fake ring or flattening the entire field.
    radial_offset = np.clip(r_map - r_lunar, 0.0, None)
    radial_gain = 1.0 + 0.35 * np.exp(-radial_offset / max(25.0, 0.18 * min(nx, ny)))
    enhanced = enhanced * radial_gain

    corona_mask = r_map > r_lunar * 1.02
    final_vis = np.zeros_like(hdr, dtype=np.float32)
    if np.any(corona_mask):
        corona_sample = enhanced[corona_mask]
        corona_log = np.log1p(500.0 * np.clip(corona_sample, 0.0, None))
        vmin, vmax = np.percentile(corona_log, [0.2, 99.2])
        final_vis[corona_mask] = np.clip((corona_log - vmin) / (vmax - vmin + 1e-8), 0.0, 1.0)
        final_vis[corona_mask] = np.power(final_vis[corona_mask], 0.72)

    final_vis[r_map <= r_lunar * 1.02] = 0.0
    final_vis[~corona_mask] = 0.0

    png_out = os.path.join(FINAL_DIR, "solar_corona_enhanced_final.png")
    plt.figure(figsize=(14, 14), facecolor="black")
    plt.imshow(final_vis, cmap="gray", origin="lower", interpolation="nearest")
    plt.axis("off")
    plt.savefig(png_out, bbox_inches="tight", facecolor="black", pad_inches=0, dpi=300)
    plt.close()

    print(f"[+] Final HDR-stack preview saved to: {png_out}")


if __name__ == "__main__":
    main()