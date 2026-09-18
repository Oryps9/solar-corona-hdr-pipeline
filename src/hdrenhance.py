import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ALIGNED_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "aligned_lights")
HDR_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "hdr")
os.makedirs(HDR_DIR, exist_ok=True)

OUT_FITS = os.path.join(HDR_DIR, "solar_corona_hdr_linear.fits")
OUT_PNG = os.path.join(HDR_DIR, "solar_corona_hdr_preview.png")

# Sensor calibration bounds (16-bit ADU)
SATURATION_LIMIT = 56000.0
NOISE_FLOOR = 150.0
Z_MID = 30000.0

def extract_exposure_ms(filename):
    match = re.search(r'_([0-9.]+)(ms)?\.fits', filename)
    if match:
        return float(match.group(1))
    return 1.0

def estimate_sky_background(data, corner_fraction=0.08):
    h, w = data.shape
    ch, cw = int(h * corner_fraction), int(w * corner_fraction)
    corners = np.concatenate([
        data[:ch, :cw].ravel(),
        data[:ch, -cw:].ravel(),
        data[-ch:, :cw].ravel(),
        data[-ch:, -cw:].ravel()
    ])
    med = np.median(corners)
    std = np.std(corners)
    valid_corners = corners[np.abs(corners - med) < 2.5 * (std + 1e-5)]
    return float(np.median(valid_corners))

def build_hdr():
    files = sorted(glob.glob(f"{ALIGNED_DIR}/*.fits"))
    if not files:
        print(f"[-] FATAL: No aligned FITS files found in {ALIGNED_DIR}")
        return

    print(f"[+] Building calibrated irradiance HDR from {len(files)} frames...")
    
    ref_hdr = fits.getheader(files[0])
    ny, nx = fits.getdata(files[0]).shape

    hdr_sum = np.zeros((ny, nx), dtype=np.float64)
    weight_sum = np.zeros((ny, nx), dtype=np.float64)

    for path in files:
        data = np.asarray(fits.getdata(path), dtype=np.float64)
        exp_time = extract_exposure_ms(os.path.basename(path))

        # 1. Sky Background Subtraction
        sky_bg = estimate_sky_background(data)
        data_clean = np.maximum(data - sky_bg, 0.0)

        # 2. Tent Weighting (Peak confidence at mid-scale)
        weight = 1.0 - np.abs((data - Z_MID) / Z_MID)
        weight = np.clip(weight, 0.0, 1.0)

        # 3. Hard Linearity & Noise Masking
        weight[data >= SATURATION_LIMIT] = 0.0
        weight[data <= NOISE_FLOOR] = 0.0

        # 4. Irradiance Accumulation
        irradiance = data_clean / exp_time
        hdr_sum += weight * irradiance
        weight_sum += weight

    # Avoid division by zero in fully dark or saturated areas
    mask_valid = weight_sum > 0.0
    hdr_final = np.zeros((ny, nx), dtype=np.float64)
    hdr_final[mask_valid] = hdr_sum[mask_valid] / weight_sum[mask_valid]

    # Save 32-bit linear scientific FITS
    fits.PrimaryHDU(data=hdr_final.astype(np.float32), header=ref_hdr).writeto(OUT_FITS, overwrite=True)
    print(f"[+] Linear HDR saved: {OUT_FITS}")

    # Generate log-scaled visual verification PNG
    preview = np.log10(np.clip(hdr_final, np.percentile(hdr_final[mask_valid], 5), np.percentile(hdr_final[mask_valid], 99.8)) + 1.0)
    preview = (preview - np.min(preview)) / (np.max(preview) - np.min(preview) + 1e-8)
    plt.imsave(OUT_PNG, preview, cmap='inferno')
    print(f"[+] Visual preview saved: {OUT_PNG}")

if __name__ == "__main__":
    build_hdr()