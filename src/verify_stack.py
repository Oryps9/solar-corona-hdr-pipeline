import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from matplotlib.colors import LogNorm

def verify_pipeline():
    print("[+] Initializing Scientific Stack Test...")
    
    # 1. Generate synthetic 2D array simulating a raw coronal matrix
    size = 512
    x, y = np.meshgrid(np.linspace(-2, 2, size), np.linspace(-2, 2, size))
    r = np.sqrt(x**2 + y**2) + 0.01
    
    # Inverse square-like radial falloff + simulated sensor noise
    synthetic_corona = (1.0 / r**2) + np.random.normal(0, 0.05, (size, size))
    synthetic_corona = np.clip(synthetic_corona, 0.01, None)

    # 2. Construct FITS Header with target observation metadata
    hdr = fits.Header()
    hdr['TELESCOP'] = 'Askar FMA180 Pro'
    hdr['CAMERA']   = 'Player One Neptune-M'
    hdr['EXPTIME']  = 0.5  # Seconds
    hdr['GAIN']     = 100
    hdr['OBSERVER'] = 'Aimar'
    
    primary_hdu = fits.PrimaryHDU(data=synthetic_corona, header=hdr)
    hdul = fits.HDUList([primary_hdu])
    
    os.makedirs('data/processed', exist_ok=True)
    test_filepath = 'data/processed/test_synthetic_corona.fits'
    hdul.writeto(test_filepath, overwrite=True)
    print(f"[+] Successfully generated FITS array: {test_filepath}")

    # 3. Parse header & render log-scale intensity plot
    with fits.open(test_filepath) as hdul:  # type: ignore
        data = np.asarray(hdul[0].data)  # type: ignore
        header = hdul[0].header  # type: ignore

        plt.figure(figsize=(7, 6))
        plt.imshow(data, cmap='inferno', norm=LogNorm())
        plt.colorbar(label='Digital Counts (DN)')
        plt.title(f"Stack Verification | Exp: {header['EXPTIME']}s | {header['CAMERA']}")

        plot_path = 'data/processed/stack_verification_plot.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[+] Rendered verification plot: {plot_path}")

if __name__ == "__main__":
    verify_pipeline()