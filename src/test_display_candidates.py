import os
import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter
from lunar_limb import find_lunar_limb

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
HDR_PATH = os.path.join(PROJECT_ROOT, 'data/processed/hdr/solar_corona_hdr_linear.fits')
NRGF_PATH = os.path.join(PROJECT_ROOT, 'data/processed/final/solar_corona_nrgf.fits')

hdr = np.asarray(fits.getdata(HDR_PATH), dtype=np.float32)
if hdr.ndim == 3:
    hdr = hdr.mean(axis=0)

nrgf = np.asarray(fits.getdata(NRGF_PATH), dtype=np.float32)
ny, nx = hdr.shape
Y, X = np.ogrid[:ny, :nx]
cy0, cx0 = ny / 2.0, nx / 2.0

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
corona_mask = r_map > r_lunar * 1.02


def eval_method(name, arr):
    arr = arr.copy()
    arr[~corona_mask] = 0.0
    arr[r_map <= r_lunar * 1.02] = 0.0
    nz = arr[arr > 0]
    print(name, 'nonzero=', nz.size, 'min=', float(nz.min()), 'p1=', float(np.percentile(nz, 1)), 'p10=', float(np.percentile(nz, 10)), 'p50=', float(np.percentile(nz, 50)), 'p90=', float(np.percentile(nz, 90)), 'p99=', float(np.percentile(nz, 99)), 'max=', float(nz.max()))


# method 1: current
blurred = gaussian_filter(hdr, sigma=2.2)
detail = hdr - blurred
enhanced = hdr + 0.42 * detail
current = np.zeros_like(hdr, dtype=np.float32)
current[corona_mask] = enhanced[corona_mask]
qmin, qmax = np.percentile(current[corona_mask], [1.0, 99.5])
current[corona_mask] = np.clip((current[corona_mask] - qmin) / (qmax - qmin + 1e-8), 0.0, 1.0)
current = np.power(current, 0.9)
current[r_map <= r_lunar * 1.02] = 0.0
current[~corona_mask] = 0.0
eval_method('current', current)

# method 2: log stretch
log_img = np.zeros_like(hdr, dtype=np.float32)
log_img[corona_mask] = np.log1p(500.0 * enhanced[corona_mask])
qlo, qhi = np.percentile(log_img[corona_mask], [0.5, 99.0])
log_display = np.zeros_like(hdr, dtype=np.float32)
log_display[corona_mask] = np.clip((log_img[corona_mask] - qlo) / (qhi - qlo + 1e-8), 0.0, 1.0)
log_display = np.power(log_display, 0.75)
log_display[r_map <= r_lunar * 1.02] = 0.0
log_display[~corona_mask] = 0.0
eval_method('log_display', log_display)

# method 3: radial background method
rad = np.clip(r_map - r_lunar, 0.0, None)
radial_gain = 1.0 + 0.7 * np.exp(-rad / (0.25 * min(nx, ny)))
radial_img = hdr * radial_gain
radial_img[~corona_mask] = 0.0
radial_img[r_map <= r_lunar * 1.02] = 0.0
p_lo, p_hi = np.percentile(radial_img[corona_mask], [0.5, 99.2])
radial_display = np.zeros_like(hdr, dtype=np.float32)
radial_display[corona_mask] = np.clip((radial_img[corona_mask] - p_lo) / (p_hi - p_lo + 1e-8), 0.0, 1.0)
radial_display = np.power(radial_display, 0.8)
radial_display[r_map <= r_lunar * 1.02] = 0.0
radial_display[~corona_mask] = 0.0
eval_method('radial_display', radial_display)

# method 4: more aggressive near-limb retention but still with full field
inner_mask = r_map > r_lunar * 0.98
soft = np.zeros_like(hdr, dtype=np.float32)
soft[inner_mask] = np.log1p(800.0 * enhanced[inner_mask])
q0, q1 = np.percentile(soft[inner_mask], [0.1, 99.2])
soft[inner_mask] = np.clip((soft[inner_mask] - q0) / (q1 - q0 + 1e-8), 0.0, 1.0)
soft = np.power(soft, 0.7)
soft[r_map <= r_lunar * 0.98] = 0.0
soft[~inner_mask] = 0.0
eval_method('soft_inner', soft)
