import os
import glob
import cv2
import numpy as np
from astropy.io import fits

# Anchor paths to script location, not working directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
aligned_dir = os.path.join(project_root, "data", "processed", "aligned_lights")
files = sorted(glob.glob(os.path.join(aligned_dir, "*.fits")))

if not files:
    print("No aligned files found!")
    exit()

print("[+] Running Blink Test. The Moon should remain DEAD CENTER.")
print("[+] Press and hold any key to fast-forward. Press 'q' to quit.")

for f in files:
    # FIX: Use np.asarray to safely load data
    data = np.asarray(fits.getdata(f), dtype=np.float32)
    
    mean = np.mean(data)
    std = np.std(data)
    vmin, vmax = max(0, mean - 0.5 * std), mean + 3 * std
    norm = np.clip((data - vmin) / (vmax - vmin + 1e-5), 0, 1)
    
    display_img = (norm * 255).astype(np.uint8)
    
    cv2.putText(display_img, f.split('\\')[-1].split('/')[-1], (30, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Blink Test", display_img)
    
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()