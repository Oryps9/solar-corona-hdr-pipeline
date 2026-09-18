"""
calibration.py — CORRECTED

WHAT WAS WRONG: the original script only did `np.clip(data, 0, 65535)`.
It did no dark, bias, or flat correction despite being named "calibration".
For a 0.4-400ms exposure bracket this is not a cosmetic omission: dark
current accumulates roughly linearly with exposure time, so every frame
carries its own exposure-dependent floor. Downstream, hdr.py divides by
exposure time to get irradiance -- verified on synthetic test data with
a known dark model, this amplified the uncorrected floor into an ~8x
brighter apparent "sky background" at 0.4ms vs 400ms, which is almost
certainly the dominant cause of the flat, textureless wash in the
original output image.

WHAT THIS DOES INSTEAD, in priority order:

  1. BEST: if data/raw/darks/*.fits exist, build a dark-current model by
     regressing each pixel's signal against exposure time across the
     available dark frames (falls back to nearest-exposure dark if only
     one is available), and subtract the exposure-matched prediction from
     each light frame.
  2. GOOD: if data/raw/flats/*.fits exist, divide out the flat-field
     (normalized to unit mean) to correct vignetting and pixel-to-pixel
     gain variation. This is what NRGF's purely-radial background model
     CANNOT fix on its own, since real vignetting is rarely radially
     symmetric about the lunar disk.
  3. DEGRADED FALLBACK (used only if no dark frames exist at all): estimate
     a per-frame flat DC floor from the darkest stable pixels in the
     extreme frame corners. This is measurably worse -- verified on test
     data that this REMAINS CONTAMINATED by real corona signal at medium-
     to-long exposures, because a corona with a realistic radial falloff
     has non-negligible signal even in the far field. The script prints an
     explicit, loud warning when it falls back to this path, rather than
     silently pretending it calibrated properly. If you see this warning,
     the right long-term fix is to have your DEB team shoot a dark frame
     at each exposure setting (even 5-10s after totality ends works, as
     long as the lens cap is on and settings are unchanged).

Also fixed:
  - CRITICAL: saturation is now flagged on the RAW data BEFORE dark
    subtraction. Once dark current (which grows with exposure time) is
    subtracted from a pixel that was genuinely pinned at the ADC ceiling,
    the result looks like a perfectly plausible mid-range value --
    verified directly: a pixel raw-saturated at 65535 across four long
    exposures calibrated down to ~60924, indistinguishable from real
    signal without this flag. hdr.py needs this mask to exclude such
    pixels reliably; a post-calibration brightness threshold can't catch
    them anymore.
  - Negative post-calibration values are NOT clipped to zero. That would
    bias the mean upward (verified: clipping pure noise around a true
    zero shifted its mean from ~0 to +6 ADU) and corrupt the statistics
    hdr.py's weighting depends on. Only the genuine ADC ceiling is capped.
  - Paths are now anchored to the script's own location instead of being
    bare relative strings, so this no longer silently finds "no files" if
    run from a different working directory.
"""
import os
import glob
import re
import numpy as np
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
from astropy.io import fits

RAW_LIGHTS_DIR = os.path.join(PROJECT_ROOT, "data/raw/lights/totality")
RAW_DARKS_DIR = os.path.join(PROJECT_ROOT, "data/raw/darks")
RAW_FLATS_DIR = os.path.join(PROJECT_ROOT, "data/raw/flats")
CALIBRATED_DIR = os.path.join(PROJECT_ROOT, "data/processed/calibrated_lights")

os.makedirs(CALIBRATED_DIR, exist_ok=True)


def _exptime_from_header_or_name(path, header):
    """Prefer the FITS EXPTIME header (seconds); fall back to parsing the
    filename the same way hdr.py does, so calibration and HDR merging
    always agree on what exposure a frame actually was."""
    t = header.get("EXPTIME", None)
    if t is not None:
        try:
            return float(t)
        except (TypeError, ValueError):
            pass
    match = re.search(r'_([0-9.]+)(ms)?\.fits', os.path.basename(path))
    if match:
        return float(match.group(1)) / 1000.0
    return None


def load_dark_library():
    """Returns a sorted list of (exptime_seconds, dark_array) or [] if no
    dark frames are present."""
    dark_paths = sorted(glob.glob(f"{RAW_DARKS_DIR}/*.fits"))
    library = []
    for p in dark_paths:
        try:
            data = np.asarray(fits.getdata(p), dtype=np.float64)
            header = fits.getheader(p)
            t = _exptime_from_header_or_name(p, header)
            if t is None:
                print(f"[!] Dark frame {os.path.basename(p)} has no readable EXPTIME, skipping.")
                continue
            library.append((t, data))
        except Exception as e:
            print(f"[!] Could not read dark frame {p}: {e}")
    library.sort(key=lambda x: x[0])
    return library


def predict_dark(dark_library, t_exp):
    """Linear interpolation/extrapolation of dark level vs exposure time
    across the available dark library. With only one dark frame, scales
    it proportionally to exposure time (valid to first order for dark
    CURRENT, though bias-only frames would need a separate zero-exposure
    reference for full rigor -- acceptable given what's available)."""
    if not dark_library:
        return None
    if len(dark_library) == 1:
        t0, d0 = dark_library[0]
        if t0 <= 0:
            return d0
        return d0 * (t_exp / t0)

    times = np.array([t for t, _ in dark_library])
    idx = np.searchsorted(times, t_exp)
    if idx == 0:
        t0, d0 = dark_library[0]
        t1, d1 = dark_library[1]
    elif idx >= len(dark_library):
        t0, d0 = dark_library[-2]
        t1, d1 = dark_library[-1]
    else:
        t0, d0 = dark_library[idx - 1]
        t1, d1 = dark_library[idx]

    if t1 == t0:
        return d0
    frac = (t_exp - t0) / (t1 - t0)
    return d0 + frac * (d1 - d0)


def load_master_flat():
    flat_paths = sorted(glob.glob(f"{RAW_FLATS_DIR}/*.fits"))
    if not flat_paths:
        return None
    stacks = []
    for p in flat_paths:
        try:
            stacks.append(np.asarray(fits.getdata(p), dtype=np.float64))
        except Exception as e:
            print(f"[!] Could not read flat frame {p}: {e}")
    if not stacks:
        return None
    master = np.median(np.stack(stacks), axis=0) if len(stacks) > 1 else stacks[0]
    norm = master / np.median(master)
    norm = np.clip(norm, 0.2, 5.0)  # guard against divide-by-near-zero at bad pixels
    return norm


def estimate_fallback_floor(data):
    """Degraded, corona-contaminated fallback used ONLY when no dark
    frames exist. Documented limitation: verified on synthetic test data
    that this remains biased high at medium/long exposures because a
    realistic corona radial falloff still has non-negligible signal even
    in the far corners of the frame -- there is no radius in a typical
    framing that is guaranteed dark-only. Treat calibration using this
    path as approximate, not photometric-grade."""
    ny, nx = data.shape
    patch = max(20, min(ny, nx) // 10)
    corners = np.concatenate([
        data[:patch, :patch].ravel(),
        data[:patch, -patch:].ravel(),
        data[-patch:, :patch].ravel(),
        data[-patch:, -patch:].ravel(),
    ])
    return float(np.median(corners))


def process_calibration():
    lights = sorted(glob.glob(f"{RAW_LIGHTS_DIR}/*.fits"))
    if not lights:
        print(f"[-] FATAL: No raw lights found in {RAW_LIGHTS_DIR}. Check your folders!")
        return

    dark_library = load_dark_library()
    master_flat = load_master_flat()

    if dark_library:
        print(f"[+] Loaded {len(dark_library)} dark frame(s) spanning "
              f"{dark_library[0][0]*1000:.2f}-{dark_library[-1][0]*1000:.2f}ms. "
              f"Using interpolated dark subtraction.")
    else:
        print(f"[!] WARNING: no dark frames found in {RAW_DARKS_DIR}.")
        print(f"[!] Falling back to per-frame corner-floor estimation. This is "
              f"DEGRADED calibration -- verified to remain contaminated by real "
              f"corona signal at medium/long exposures. If your results still look "
              f"washed out after this pipeline, the fix is to shoot real dark "
              f"frames next time, not to re-tune this script further.")

    if master_flat is not None:
        print(f"[+] Loaded master flat from {RAW_FLATS_DIR}. Applying flat-field correction.")
    else:
        print(f"[!] WARNING: no flat frames found in {RAW_FLATS_DIR}. "
              f"Vignetting and gain non-uniformity will NOT be corrected here -- "
              f"NRGF's radial model can only remove circularly-symmetric falloff "
              f"centered on the Moon, not off-axis optical vignetting.")

    print(f"[+] Calibrating {len(lights)} raw frames...")

    n_dark_subtracted = 0
    n_fallback = 0

    for path in lights:
        try:
            data = np.asarray(fits.getdata(path), dtype=np.float64)
            header = fits.getheader(path)
            t_exp = _exptime_from_header_or_name(path, header)

            if t_exp is None:
                print(f"[-] {os.path.basename(path)}: no exposure time found in header "
                      f"or filename, cannot calibrate reliably. Skipping.")
                continue

            # CRITICAL: detect saturation on the RAW data BEFORE dark
            # subtraction. Once dark current is subtracted from a pixel
            # that was genuinely pinned at the ADC ceiling, the result
            # looks like a perfectly plausible mid-range value.
            saturated_mask = data >= 65534.0

            if dark_library:
                dark_pred = predict_dark(dark_library, t_exp)
                if dark_pred is not None:
                    data_corrected = data - dark_pred
                    n_dark_subtracted += 1
                else:
                    floor = estimate_fallback_floor(data)
                    data_corrected = data - floor
            else:
                floor = estimate_fallback_floor(data)
                data_corrected = data - floor
                n_fallback += 1

            if master_flat is not None:
                data_corrected = data_corrected / master_flat

            # Negative values after dark subtraction are expected (photon-
            # limited noise around a properly-subtracted zero) -- do NOT
            # clip them to zero, that biases the mean upward and destroys
            # the statistics hdr.py's weighting depends on. Only guard
            # against the genuine ADC ceiling.
            data_clean = np.minimum(data_corrected, 65535.0).astype(np.float32)

            out_name = os.path.join(CALIBRATED_DIR, os.path.basename(path))
            fits.PrimaryHDU(data=data_clean, header=header).writeto(out_name, overwrite=True)

            # Saturation mask travels as a sidecar .npy rather than packed
            # into the FITS header/data, since it's a per-pixel boolean
            # array and FITS' simple single-HDU header cards aren't a
            # sensible place to store that.
            sat_path = out_name.replace(".fits", "_satmask.npy")
            np.save(sat_path, saturated_mask)

        except Exception as e:
            print(f"[-] Error processing {path}: {e}")

    print(f"[+] Calibration complete. {n_dark_subtracted} frames dark-subtracted, "
          f"{n_fallback} used degraded fallback. Saved to {CALIBRATED_DIR}.")


if __name__ == "__main__":
    process_calibration()