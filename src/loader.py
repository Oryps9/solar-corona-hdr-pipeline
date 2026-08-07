import os
import glob
import numpy as np
from astropy.io import fits
import pandas as pd

class FITSLoader:
    """
    Ingestion and header metadata parsing engine for raw solar eclipse frames.
    """
    def __init__(self, raw_data_dir: str):
        self.raw_data_dir = raw_data_dir
        self.file_manifest = []

    def scan_directory(self) -> pd.DataFrame:
        """
        Scans raw data directory, parses FITS headers, and returns a metadata DataFrame.
        """
        search_path = os.path.join(self.raw_data_dir, "**", "*.fits")
        fits_files = glob.glob(search_path, recursive=True)
        
        if not fits_files:
            print(f"[!] Warning: No .FITS files found in {self.raw_data_dir}")
            return pd.DataFrame()

        records = []
        for filepath in fits_files:
            try:
                with fits.open(filepath) as hdul:
                    header = hdul[0].header
                    data = hdul[0].data
                    
                    record = {
                        "filepath": filepath,
                        "filename": os.path.basename(filepath),
                        "exptime": header.get("EXPTIME", np.nan),
                        "gain": header.get("GAIN", np.nan),
                        "camera": header.get("CAMERA", "Unknown"),
                        "telescope": header.get("TELESCOP", "Unknown"),
                        "dimensions": data.shape if data is not None else None,
                        "mean_dn": np.mean(data) if data is not None else np.nan,
                        "max_dn": np.max(data) if data is not None else np.nan
                    }
                    records.append(record)
            except Exception as e:
                print(f"[!] Error reading {filepath}: {e}")

        self.file_manifest = pd.DataFrame(records)
        print(f"[+] Successfully indexed {len(records)} raw FITS files.")
        return self.file_manifest

if __name__ == "__main__":
    import os
    
    # Target raw data directory; fall back to processed test array if raw is empty
    raw_dir = "data/raw"
    has_raw_files = os.path.exists(raw_dir) and any(f.endswith('.fits') for f in os.listdir(raw_dir))
    target_dir = raw_dir if has_raw_files else "data/processed"
    
    print(f"[+] Testing FITSLoader against directory: {target_dir}")
    loader = FITSLoader(target_dir)
    manifest = loader.scan_directory()
    print("\n--- EXTRACTED METADATA MANIFEST ---")
    print(manifest.to_string())