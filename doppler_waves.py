import asdf
import matplotlib.pyplot as plt
import numpy as np

# --- Load Doppler map only ---
filename = "IRIS_fitting_Si_IV_1403_20160520_131758.asdf"
with asdf.open(filename) as af:
    dopp_map = af.tree["dopp_map"]

# --- Basic parameters ---
nx, ny = dopp_map.data.shape[1], dopp_map.data.shape[0]

# X-axis: raster time (force 0 → 10000 s)
t_array = np.linspace(0, 10000, nx)

# Y-axis: solar slit positions (arcsec)
slit_pos = dopp_map.meta['crval2'] + dopp_map.meta['cdelt2'] * (np.arange(ny) - dopp_map.meta['crpix2'])

# --- Pick a line near Solar Y = 89 arcsec ---
desired_y = -89
y_index = np.argmin(np.abs(slit_pos - desired_y))
actual_y = slit_pos[y_index]

# Extract the Doppler values, ignoring NaNs
dopp_line = np.nan_to_num(dopp_map.data[y_index, :], nan=np.nanmean(dopp_map.data))

# --- Plot the oscillation ---
plt.figure(figsize=(9, 4))
plt.plot(t_array, dopp_line, color='royalblue', lw=1.4)
plt.title(f"IRIS Si IV 1403 Å  Doppler Oscillation at Solar Y ≈ {actual_y:.1f} arcsec")
plt.xlabel("Time from raster start (s)")
plt.ylabel("Velocity (km/s)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
