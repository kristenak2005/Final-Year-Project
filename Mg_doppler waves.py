import asdf
import matplotlib.pyplot as plt
import numpy as np

# --- Load Doppler map only ---
filename = "IRIS_fitting_Si_IV_1403_20160520_131758.asdf"
#filename = "IRIS_fitting_C_II_1334_20160520_131758.asdf"
#filename = "IRIS_fitting_C_II_1335_20160520_131758.asdf"
#filename = "IRIS_fitting_Si_IV_1394_20160520_131758.asdf"

with asdf.open(filename) as af:
    mgii_k_vdopp = af.tree["mgii_k_vdopp"]

# --- Basic parameters ---
nx, ny = mgii_k_vdopp.data.shape[1], mgii_k_vdopp.data.shape[0]

# X-axis: raster time (force 0 → 10000 s)
t_array = np.linspace(0, 10000, nx)

# Y-axis: solar slit positions (arcsec)
slit_pos = mgii_k_vdopp.meta['crval2'] + mgii_k_vdopp.meta['cdelt2'] * (np.arange(ny) - mgii_k_vdopp.meta['crpix2'])

# --- Pick a line near Solar Y = 89 arcsec ---
desired_y = -89
y_index = np.argmin(np.abs(slit_pos - desired_y))
actual_y = slit_pos[y_index]

# Extract the Doppler values, ignoring NaNs
dopp_line = np.nan_to_num(mgii_k_vdopp.data[y_index, :], nan=np.nanmean(mgii_k_vdopp.data))

# --- Plot the oscillation ---
plt.figure(figsize=(9, 4))
plt.plot(t_array, dopp_line, color='royalblue', lw=1.4)
plt.title(f"IRIS Si IV 1403 Å  Doppler Oscillation at Solar Y ≈ {actual_y:.1f} arcsec")
plt.xlabel("Time from raster start (s)")
plt.ylabel("Velocity (km/s)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
