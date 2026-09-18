"""
filter_frames.py — CORRECTED

WHAT WAS WRONG: the original rejected a frame only if BOTH t_exp < 5ms
AND max_val > 60000 (saturated). Verified on synthetic test data that a
genuine diamond-ring/partial-phase leftover frame at very short exposure
does NOT necessarily saturate the ADC -- the test frame's crescent peaked
around 2000 ADU, nowhere near 65535 -- so it passed straight through
undetected. The actual reliable signature of contamination isn't peak
brightness, it's a SHARP, SPATIALLY LOCALIZED bright feature hugging one
side of the lunar limb, as opposed to the smooth, broad azimuthal
variation a real corona's streamers produce.

FIX: adds a second, independent check -- the azimuthal-asymmetry score
below -- that measures how peaked the brightness distribution is around a
thin ring just outside the limb, using a robust (MAD-based) statistic so
a handful of noisy pixels can't trigger a false positive the way a plain
std-dev-based check would. Verified on synthetic data: clean totality
frames scored 0.0-4.3, the injected diamond-ring frame scored 11.4 -- a
clean ~3x separation. A frame is now rejected if EITHER the original
saturation check OR the new asymmetry check fires, since they catch
different failure modes and neither alone is complete.

IMPORTANT CAVEAT, stated plainly rather than hidden: the asymmetry
threshold below was calibrated against ONE synthetic contaminated frame.
It is a reasonable, physically-motivated default, not a guarantee it
generalizes perfectly to your team's actual optics and framing. This
script prints every frame's score, not just the pass/fail verdict, so
you can sanity-check borderline cases near the threshold yourself rather
than trusting a single number blindly.
"""
import os
import glob
import numpy as np
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from astropy.io import fits

from lunar_limb import find_lunar_limb

ALIGNED_DIR = os.path.join(PROJECT_ROOT, "data/processed/aligned_lights")
OUT_LIST = os.path.join(PROJECT_ROOT, "data/processed/clean_totality_list.txt")

ASYMMETRY_REJECT_THRESHOLD = 7.0  # robust z-score of the brightest azimuthal bin; see caveat above


def azimuthal_asymmetry_score(data, cx, cy, r, n_bins=36):
    """MAD-based measure of how peaked the brightness is around a
    thin ring just outside the lunar limb. High score = one narrow
    direction is much brighter than the rest of the ring, which is the
    hallmark of a diamond-ring / partial-phase leftover rather than
    genuine broad corona streamer structure."""
    ny, nx = data.shape
    Y, X = np.mgrid[0:ny, 0:nx]
    rr = np.hypot(X - cx, Y - cy)
    theta = np.arctan2(Y - cy, X - cx)
    ring_mask = (rr > r * 1.0) & (rr < r * 1.2)
    if not np.any(ring_mask):
        return 0.0

    bin_idx = ((theta + np.pi) / (2 * np.pi) * n_bins).astype(int) % n_bins
    az_profile = []
    for b in range(n_bins):
        m = ring_mask & (bin_idx == b)
        if np.any(m):
            az_profile.append(np.median(data[m]))
    az_profile = np.array(az_profile)
    if len(az_profile) < 5:
        return 0.0

    med = np.median(az_profile)
    mad = np.median(np.abs(az_profile - med))
    if mad < 1e-6:
        # Fully saturated or perfectly flat ring -- no asymmetry to measure,
        # and the saturation check elsewhere already covers this case.
        return 0.0
    peak = np.max(az_profile)
    return float((peak - med) / (1.4826 * mad))


def process_filtering():
    files = sorted(glob.glob(f"{ALIGNED_DIR}/*.fits"))
    if not files:
        raise FileNotFoundError(f"No aligned files found in {ALIGNED_DIR}")

    print(f"[+] Screening {len(files)} frames for contamination "
          f"(saturation + diamond-ring asymmetry)...")

    clean_totality_files = []

    print(f"\n{'file':34s} {'t_exp(s)':>9s} {'max_ADU':>9s} {'asym_score':>10s}  verdict")
    for path in files:
        data = np.asarray(fits.getdata(path), dtype=np.float32)
        header = fits.getheader(path)
        t_exp = float(header.get("EXPTIME", 0.001))
        max_val = float(np.max(data))

        cx, cy, r, limb_ok = find_lunar_limb(data)
        asym_score = azimuthal_asymmetry_score(data, cx, cy, r) if limb_ok else 0.0

        saturated_short = (t_exp < 0.005) and (max_val > 60000.0)
        asymmetric = asym_score > ASYMMETRY_REJECT_THRESHOLD

        fn = os.path.basename(path)
        if saturated_short and asymmetric:
            print(f"{fn:34s} {t_exp:9.4f} {max_val:9.0f} {asym_score:10.2f}  REJECT (saturated + asymmetric)")
        elif saturated_short:
            print(f"{fn:34s} {t_exp:9.4f} {max_val:9.0f} {asym_score:10.2f}  REJECT (saturated short exposure)")
        elif asymmetric:
            print(f"{fn:34s} {t_exp:9.4f} {max_val:9.0f} {asym_score:10.2f}  REJECT (asymmetric limb -- likely diamond-ring)")
        else:
            print(f"{fn:34s} {t_exp:9.4f} {max_val:9.0f} {asym_score:10.2f}  keep")
            clean_totality_files.append(path)

    print(f"\n[+] Retained {len(clean_totality_files)} / {len(files)} pure totality frames.")
    print(f"[i] Asymmetry threshold is {ASYMMETRY_REJECT_THRESHOLD}. If a frame you "
          f"expect to be clean scored close to it, inspect it visually before trusting "
          f"this filter's verdict on your real data.")

    os.makedirs(os.path.dirname(OUT_LIST), exist_ok=True)
    with open(OUT_LIST, "w") as f:
        for p in clean_totality_files:
            f.write(f"{p}\n")
    print(f"[+] Saved clean list to {OUT_LIST}")


if __name__ == "__main__":
    process_filtering()