import asdf
import numpy as np
import matplotlib.pyplot as plt
from WaLSAtools import WaLSAtools

filename = "IRIS_fitting_Si_IV_1403_20160520_131758.asdf"
with asdf.open(filename) as af:
    dopp_map = af.tree["dopp_map"]
    meta = dopp_map.meta

nx, ny = dopp_map.data.shape[1], dopp_map.data.shape[0]
cadence = meta.get('STEPT_AV', 1.0)  # seconds per step
t_array = np.linspace(0, nx * cadence, nx)

slit_pos = meta['crval2'] + meta['cdelt2'] * (np.arange(ny) - meta['crpix2'])
target_y = -89
y_index = np.argmin(np.abs(slit_pos - target_y))
v_time = dopp_map.data[y_index, :]
time = t_array
mask = np.isfinite(v_time)
v_time = v_time[mask]
time = time[mask]
v_detrended = v_time - np.mean(v_time)

fft_power, fft_freqs, fft_significance, _ = WaLSAtools(
    signal=v_detrended,
    time=time,
    method='fft',
    siglevel=0.95,
    apod=0.1
)

fft_power_norm = 100 * fft_power / np.max(fft_power)
plt.figure(figsize=(10,5))
plt.plot(fft_freqs, fft_power_norm, label='FFT Power (%)')
plt.xlabel("Frequency (Hz)")
plt.ylabel("Normalized Power (%)")
plt.title(f"FFT Analysis — IRIS Doppler Oscillations at Y = {target_y} arcsec")
plt.grid(True)
plt.legend()
plt.show()
