import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import matplotlib.gridspec as gridspec
import asdf

from WaLSAtools import WaLSAtools  # type: ignore

#files
#filename = "IRIS_fitting_Si_IV_1394_20160520_131758.asdf"
#filename = "IRIS_fitting_Si_IV_1403_20160520_131758.asdf"
#filename = "IRIS_fitting_smooth_MgII_20160520_131758.asdf"
#filename = "IRIS_fitting_C_II_1335_20160520_131758.asdf"
filename = "IRIS_fitting_C_II_1334_20160520_131758.asdf"
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

dt = float(np.median(np.diff(time)))
fs = 1.0 / dt
f_nyq = fs / 2.0

signal = v_time - np.nanmean(v_time)

print("---- Sampling diagnostics ----")
print(f"nx = {nx}")
print(f"Cadence used (s):     {cadence:.3f}")
print(f"Median dt used (s):   {dt:.3f}")
print(f"Sampling rate (Hz):   {fs:.5f}")
print(f"Nyquist freq (Hz):    {f_nyq:.5f}")
print(f"Min resolvable P (s): {1/f_nyq:.1f}")

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
    "legend.fontsize": 10,
    "figure.titlesize": 12.5,
    "axes.grid": False,
})

significance_threshold = 0.05
plt.rc("axes", linewidth=1.0)
plt.rc("lines", linewidth=1.3)

colors = plt.cm.tab10(np.linspace(0, 1, len(imfs)))

tmin = float(np.nanmin(time))
tmax = float(np.nanmax(time))


f_3min = 1 / 180.0  # 0.00556 Hz
f_5min = 1 / 300.0  # 0.00333 Hz


fig = plt.figure(figsize=(8.0, 9.2), constrained_layout=True)
gs = gridspec.GridSpec(len(imfs) + 2, 2, height_ratios=[1] * len(imfs) + [0.3, 2], figure=fig)

# IMFs + instantaneous frequencies
for i, (imf, freq) in enumerate(zip(imfs, instantaneous_frequencies)):
    ax_imf = fig.add_subplot(gs[i, 0])
    ax_if = fig.add_subplot(gs[i, 1])

    # Grey background if IMF insignificant
    if IMF_significance_levels[i] > significance_threshold:
        ax_imf.set_facecolor("lightgray")
        ax_if.set_facecolor("lightgray")

    # IMF plot
    ax_imf.plot(time, imf, color=colors[i])
    ax_imf.set_ylabel(f"IMF {i+1}")
    ax_imf.yaxis.set_label_coords(-0.16, 0.5)
    ax_imf.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax_imf.yaxis.set_minor_locator(AutoMinorLocator(3))
    ax_imf.tick_params(axis="both", which="major", direction="in", length=6, width=1.0)
    ax_imf.tick_params(axis="both", which="minor", direction="in", length=3, width=1.0)
    ax_imf.set_xlim(tmin, tmax)
    if i < len(imfs) - 1:
        ax_imf.set_xticklabels([])
    if i == 0:
        ax_imf.set_title("(a) IMFs")
    if i == len(imfs) - 1:
        ax_imf.set_xlabel("Time (s)")

    # Instantaneous frequency plot
    freq = np.asarray(freq, dtype=float)
    # Ensure same length as time
    if freq.size < time.size:
        freq = np.pad(freq, (0, time.size - freq.size), mode="edge")
    elif freq.size > time.size:
        freq = freq[: time.size]

    ax_if.plot(time, freq, color=colors[i])
    ax_if.set_ylabel(f"IF {i+1} (Hz)")
    ax_if.yaxis.set_label_coords(-0.1, 0.5)
    ax_if.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax_if.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax_if.tick_params(axis="both", which="major", direction="in", length=6, width=1.0)
    ax_if.tick_params(axis="both", which="minor", direction="in", length=3, width=1.0)
    ax_if.set_xlim(tmin, tmax)
    if i < len(imfs) - 1:
        ax_if.set_xticklabels([])
    if i == 0:
        ax_if.set_title("(b) Instantaneous Frequencies")
    if i == len(imfs) - 1:
        ax_if.set_xlabel("Time (s)")


# (c) HHT marginal spectrum
ax_hht = fig.add_subplot(gs[-1, 0])

HHT_freq_bins = np.asarray(HHT_freq_bins, dtype=float)
HHT_power_spectrum = np.asarray(HHT_power_spectrum, dtype=float)
HHT_significance_level = np.asarray(HHT_significance_level, dtype=float)

hmask = HHT_freq_bins <= f_nyq

ax_hht.plot(HHT_freq_bins[hmask], HHT_power_spectrum[hmask], color="black")
ax_hht.plot(HHT_freq_bins[hmask], HHT_significance_level[hmask], linestyle="--", color="green")

ax_hht.set_title("(c) HHT Marginal Spectrum")
ax_hht.set_xlabel("Frequency (Hz)")
ax_hht.set_ylabel("Power")
ax_hht.xaxis.set_minor_locator(AutoMinorLocator(5))
ax_hht.yaxis.set_minor_locator(AutoMinorLocator(5))
ax_hht.tick_params(axis="both", which="major", direction="out", length=6, width=1.0)
ax_hht.tick_params(axis="both", which="minor", direction="out", length=3, width=1.0)
ax_hht.set_xlim(0, f_nyq)
ax_hht.set_ylim(bottom=0)

# Physical guide lines
ax_hht.axvline(f_3min, color="gray", linestyle=":", label="3 min")
ax_hht.axvline(f_5min, color="gray", linestyle="--", label="5 min")

ax_fft = fig.add_subplot(gs[-1, 1])

for i, (xf, psd) in enumerate(psd_spectra_fft):
    xf = np.asarray(xf, dtype=float)
    psd = np.asarray(psd, dtype=float)

    m = xf <= f_nyq
    ax_fft.plot(xf[m], psd[m], label=f"IMF {i+1}", color=colors[i])

    conf = np.asarray(confidence_levels_fft[i], dtype=float)
    if conf.shape != psd.shape:
        conf = None
    if conf is not None:
        ax_fft.plot(xf[m], conf[m], linestyle="--", color=colors[i], alpha=0.8)

ax_fft.set_title("(d) FFT Spectra of IMFs")
ax_fft.set_xlabel("Frequency (Hz)")
ax_fft.set_ylabel("Power")
ax_fft.xaxis.set_minor_locator(AutoMinorLocator(5))
ax_fft.yaxis.set_minor_locator(AutoMinorLocator(5))
ax_fft.tick_params(axis="both", which="major", direction="out", length=6, width=1.0)
ax_fft.tick_params(axis="both", which="minor", direction="out", length=3, width=1.0)
ax_fft.set_xlim(0, f_nyq)
ax_fft.set_ylim(bottom=0)

ax_fft.axvline(f_3min, color="gray", linestyle=":", label="3 min (5.56 mHz)")
ax_fft.axvline(f_5min, color="gray", linestyle="--", label="5 min (3.33 mHz)")

ax_fft.legend(loc="upper right", fontsize=8)

plt.show()
