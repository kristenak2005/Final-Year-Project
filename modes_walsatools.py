import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import asdf

from WaLSAtools import WaLSAtools  # type: ignore

# files
# filename = "IRIS_fitting_Si_IV_1394_20160520_131758.asdf"
#Primary wave diagnostic
filename = "IRIS_fitting_Si_IV_1403_20160520_131758.asdf"
# filename = "IRIS_fitting_smooth_MgII_20160520_131758.asdf"
# filename = "IRIS_fitting_C_II_1335_20160520_131758.asdf"
#filename = "IRIS_fitting_C_II_1334_20160520_131758.asdf"

FALLBACK_CADENCE_S = 16.2  # from OBS summary (Step Cad)
target_y = -90  # arcsec

with asdf.open(filename) as af:
    dopp_map = af.tree["asym_map"]
    meta = dict(dopp_map.meta)

data = np.asarray(dopp_map.data)
ny, nx = data.shape

cadence = float(meta.get("STEPT_AV", FALLBACK_CADENCE_S))
time = np.arange(nx) * cadence

slit_pos = float(meta["crval2"]) + float(meta["cdelt2"]) * (np.arange(ny) - float(meta["crpix2"]))
y_index = int(np.argmin(np.abs(slit_pos - target_y)))

v_time = data[y_index, :].astype(float)
mask = np.isfinite(v_time) & np.isfinite(time)
v_time = v_time[mask]
time = time[mask]

signal = v_time - np.nanmean(v_time)

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
    "font.size": 12,
    "axes.titlesize": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.grid": False,
})
plt.rc("axes", linewidth=1.0)
plt.rc("lines", linewidth=1.3)

colors = plt.cm.tab10(np.linspace(0, 1, len(imfs)))

tmin = float(np.nanmin(time))
tmax = float(np.nanmax(time))

fig, axes = plt.subplots(len(imfs), 1, figsize=(8.0, 1.6 * len(imfs)), sharex=True, constrained_layout=True)

if len(imfs) == 1:
    axes = [axes]

for i, imf in enumerate(imfs):
    ax = axes[i]
    ax.plot(time, imf, color=colors[i])
    ax.set_ylabel(f"IMF {i+1}")
    ax.yaxis.set_label_coords(-0.08, 0.5)
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(3))
    ax.tick_params(axis="both", which="major", direction="in", length=6, width=1.0)
    ax.tick_params(axis="both", which="minor", direction="in", length=3, width=1.0)
    ax.set_xlim(tmin, tmax)

axes[-1].set_xlabel("Time (s)")

plt.show()
