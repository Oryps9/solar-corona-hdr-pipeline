import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage import measure
from astropy.io import fits

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 1. Resolve Target FITS File Path
if len(sys.argv) > 1:
    FITS_PATH = os.path.abspath(sys.argv[1])
else:
    # Default fallbacks
    for candidate in [
        "data/processed/hdr/umbra_hdr.fits",
        "data/processed/hdr/solar_corona_hdr_linear.fits",
        "data/processed/hdr/DEB_HDR_stretched_mono.fits"
    ]:
        p = os.path.join(PROJECT_ROOT, candidate)
        if os.path.exists(p):
            FITS_PATH = p
            break
    else:
        print("[-] FATAL: No valid HDR FITS file found in data/processed/hdr/")
        sys.exit(1)

OUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
OUT_PNG = os.path.join(OUT_DIR, "deb_coronal_shape_clean.png")
OVERLAY_PNG = os.path.join(OUT_DIR, "deb_isophotes_overlay.png")
os.makedirs(OUT_DIR, exist_ok=True)

print(f"[+] Loading HDR data from: {FITS_PATH}")
raw_data = fits.getdata(FITS_PATH)
data = np.asarray(raw_data, dtype=np.float64)

# 2. Collapse Multi-Channel / 3D Arrays to 2D
if data.ndim == 3:
    if data.shape[0] == 3:
        data = 0.299 * data[0] + 0.587 * data[1] + 0.114 * data[2]
    elif data.shape[2] == 3:
        data = 0.299 * data[:, :, 0] + 0.587 * data[:, :, 1] + 0.114 * data[:, :, 2]
    elif data.shape[0] == 1:
        data = data[0]
    elif data.shape[2] == 1:
        data = data[:, :, 0]

ny, nx = data.shape

# Apply mild Gaussian smoothing to filter high-frequency sensor noise without distorting large-scale geometry
data_smooth = gaussian_filter(data, sigma=1.2)

# 3. Auto-Detect Solar/Lunar Center and Radius in Pixels
Y, X = np.ogrid[:ny, :nx]
dist_from_mid = np.hypot(X - nx / 2.0, Y - ny / 2.0)
central_roi = dist_from_mid < (min(nx, ny) * 0.35)

valid_core = data[central_roi & (data > 0) & np.isfinite(data)]
if len(valid_core) > 0:
    dark_thresh = np.percentile(valid_core, 15.0)
    moon_mask = (data <= dark_thresh) & central_roi
    ly, lx = np.where(moon_mask)
    if len(ly) > 100:
        cx_sun = float(np.mean(lx))
        cy_sun = float(np.mean(ly))
        r_sun_px = float(np.sqrt(len(ly) / np.pi))
    else:
        cx_sun, cy_sun = nx / 2.0, ny / 2.0
        r_sun_px = min(nx, ny) * 0.165
else:
    cx_sun, cy_sun = nx / 2.0, ny / 2.0
    r_sun_px = min(nx, ny) * 0.165

print(f"[+] Solar Center: (X={cx_sun:.1f}, Y={cy_sun:.1f}) | Calibrated R_sun: {r_sun_px:.1f} px")

# 4. Logarithmic Sampling of Coronal Surface Brightness
valid_pixels = data_smooth[(data_smooth > 0) & np.isfinite(data_smooth)]
i_min = float(np.percentile(valid_pixels, 35.0))
i_max = float(np.percentile(valid_pixels, 99.8))
levels = np.geomspace(max(i_min, 1e-7), max(i_max, 1e-6), num=120)

records = []
margin = 15
overlay_pts = []

for lvl in levels:
    contours = measure.find_contours(data_smooth, lvl)
    if not contours:
        continue

    for c in contours:
        if len(c) < 100:
            continue

        pts = np.fliplr(c).astype(np.float32)

        # Reject boundary-clipping contours
        if (np.any(pts[:, 0] <= margin) or np.any(pts[:, 0] >= nx - margin) or
            np.any(pts[:, 1] <= margin) or np.any(pts[:, 1] >= ny - margin)):
            continue

        try:
            ellipse = cv2.fitEllipseDirect(pts)
        except cv2.error:
            try:
                ellipse = cv2.fitEllipse(pts)
            except cv2.error:
                continue

        (xc, yc), (d1, d2), angle = ellipse
        a = max(d1, d2) / 2.0
        b = min(d1, d2) / 2.0

        if b <= 0 or a <= 0:
            continue

        # Concentricity constraint (reject asymmetric artifacts/glare)
        center_drift = np.hypot(xc - cx_sun, yc - cy_sun)
        if center_drift > 0.35 * r_sun_px:
            continue

        # Position Angle of the semi-major axis relative to vertical
        pa = angle if d2 >= d1 else (angle + 90.0) % 180.0
        r_mean = np.sqrt(a * b) / r_sun_px
        ludendorff_E = (a / b) - 1.0
        ellipticity = 1.0 - (b / a)

        # Physical coronal evaluation interval (matching DEB paper regime)
        if 1.48 <= r_mean <= 2.85 and ludendorff_E < 0.35 and ellipticity < 0.25:
            records.append({
                "r": r_mean,
                "E": ludendorff_E,
                "eps": ellipticity,
                "pa": pa
            })
            overlay_pts.append(pts)

print(f"[+] Total valid physical isophotes accepted: {len(records)}")

if not records:
    print("[-] FATAL: No valid physical isophotes found in 1.48 - 2.85 R_sun interval.")
    sys.exit(1)

records = sorted(records, key=lambda k: k["r"])
r_arr = np.array([k["r"] for k in records])
E_arr = np.array([k["E"] for k in records])
eps_arr = np.array([k["eps"] for k in records])
pa_arr = np.array([k["pa"] for k in records])

# 5. Diagnostic Contour Overlay
plt.figure(figsize=(10, 10), facecolor="black")
v_low, v_high = np.percentile(valid_pixels, [5.0, 99.5])
stretched_vis = np.log10(np.clip(data, v_low, v_high) - v_low + 1.0)
plt.imshow(stretched_vis, cmap="inferno", origin="lower")

for pts in overlay_pts:
    plt.plot(pts[:, 0], pts[:, 1], color="cyan", lw=0.6, alpha=0.6)

plt.plot(cx_sun, cy_sun, "+", color="white", ms=12, mew=2, label="Solar Center")
plt.axis("off")
plt.savefig(OVERLAY_PNG, bbox_inches="tight", facecolor="black", dpi=200)
plt.close()
print(f"[+] Diagnostic overlay saved: {OVERLAY_PNG}")

# 6. Generate 3-Panel Publication Figure
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7, 10), sharex=True, dpi=300)
fig.suptitle("DEB Coronal Shape (2026)", fontsize=13, fontweight="bold", y=0.92)

# Panel 1: Ludendorff Flattening (E)
ax1.plot(r_arr, E_arr, "o-", color="#1f77b4", lw=1.2, ms=3, label="Measured $E$")
ax1.set_ylabel(r"Ludendorff Flattening ($E$)", fontsize=10)
ax1.grid(True, linestyle="--", alpha=0.5)

mask_fit = (r_arr >= 1.48) & (r_arr <= 1.72)
if np.sum(mask_fit) > 3:
    poly = np.polyfit(r_arr[mask_fit], E_arr[mask_fit], deg=1)
    ax1.plot(r_arr[mask_fit], np.polyval(poly, r_arr[mask_fit]), "--", color="darkred", lw=2, label=r"Linear Fit ($R \leq R_{\max}$)")
ax1.legend(loc="upper right", fontsize=8)

# Panel 2: Ellipticity (epsilon)
ax2.plot(r_arr, eps_arr, "s-", color="#2ca02c", lw=1.2, ms=3)
ax2.set_ylabel(r"Ellipticity ($\epsilon = 1 - b/a$)", fontsize=10)
ax2.grid(True, linestyle="--", alpha=0.5)

# Panel 3: Position Angle (PA)
ax3.plot(r_arr, pa_arr, "^-", color="#d62728", lw=1.2, ms=3)
ax3.set_xlabel(r"Mean Isophote Height ($R_\odot$)", fontsize=10)
ax3.set_ylabel(r"Semi-major Axis Position Angle ($\mathrm{PA}_\odot$)", fontsize=10)
ax3.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig(OUT_PNG, bbox_inches="tight")
plt.close()
print(f"[+] Publication profile successfully saved to: {OUT_PNG}")