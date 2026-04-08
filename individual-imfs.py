import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import asdf

from WaLSAtools import WaLSAtools  

filename = "IRIS_fitting_Si_IV_1403_20160520_131758.asdf"
FALLBACK_CADENCE_S = 16.2
target_y = -90  
map_key = "dopp_map"   

with asdf.open(filename) as af:
    data_map = af.tree[map_key]
    meta = dict(data_map.meta)

data = np.asarray(data_map.data)
ny, nx = data.shape

cadence = float(meta.get("STEPT_AV", FALLBACK_CADENCE_S))
time = np.arange(nx) * cadence

slit_pos = float(meta["crval2"]) + float(meta["cdelt2"]) * (
    np.arange(ny) - float(meta["crpix2"])
)
y_index = int(np.argmin(np.abs(slit_pos - target_y)))
actual_y = slit_pos[y_index]

series = data[y_index, :].astype(float)
mask = np.isfinite(series) & np.isfinite(time)
series = series[mask]
time = time[mask]
signal = series - np.nanmean(series)

(
    HHT_power_spectrum,
    HHT_significance_level,
    HHT_freq_bins,
    psd_spectra_fft,
    confidence_levels_fft,
    imfs,
    IMF_significance_levels,
    instantaneous_frequencies,
) = WaLSAtools(
    signal=signal,
    time=time,
    method="emd",
    siglevel=0.95,
)

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.grid": False,
})
plt.rc("axes", linewidth=0.8)
plt.rc("lines", linewidth=1.0)

tmin = float(np.nanmin(time))
tmax = float(np.nanmax(time))

n_panels = len(imfs) + 1  
fig, axes = plt.subplots(
    n_panels, 1,
    figsize=(6.5, 1.1 * n_panels),
    sharex=True,
    constrained_layout=True
)

if n_panels == 1:
    axes = [axes]

axes[0].plot(time, signal, color="black")
signal_label = "Doppler velocity" if map_key == "dopp_map" else "Asymmetry"
axes[0].set_ylabel("Signal")
#axes[0].set_title(
 #   f"{signal_label} signal analysed with EMD\n"
  #  f"Si IV 1403 at Y = {actual_y:.1f}\""
#)
axes[0].yaxis.set_label_coords(-0.08, 0.5)
axes[0].xaxis.set_minor_locator(AutoMinorLocator(5))
axes[0].yaxis.set_minor_locator(AutoMinorLocator(3))
axes[0].tick_params(axis="both", which="major", direction="in", length=6, width=1.0)
axes[0].tick_params(axis="both", which="minor", direction="in", length=3, width=1.0)
axes[0].set_xlim(tmin, tmax)

#IMF's
for i, imf in enumerate(imfs):
    ax = axes[i + 1]
    ax.plot(time, imf, color="black")
    ax.set_ylabel(f"IMF {i+1}")
    ax.yaxis.set_label_coords(-0.08, 0.5)
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(3))
    ax.tick_params(axis="both", which="major", direction="in", length=6, width=1.0)
    ax.tick_params(axis="both", which="minor", direction="in", length=3, width=1.0)
    ax.set_xlim(tmin, tmax)

axes[-1].set_xlabel("Time (s)")

plt.savefig("images/emd_imfs_signal_y-90.png", bbox_inches="tight")
plt.show()
