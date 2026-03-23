import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import matplotlib.gridspec as gridspec
import asdf

from WaLSAtools import WaLSAtools

filename = "IRIS_fitting_C_II_1334_20160520_131758.asdf"
FALLBACK_CADENCE_S = 16.2
target_y = -90
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
dt = float(np.median(np.diff(time)))
fs = 1.0 / dt
f_nyq = fs / 2.0

print("---- Sampling diagnostics ----")
print(f"Cadence used (s):     {cadence:.3f}")
print(f"Median dt used (s):   {dt:.3f}")
print(f"Sampling rate (Hz):   {fs:.5f}")
print(f"Nyquist freq (Hz):    {f_nyq:.5f}")

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
})

colors = plt.cm.tab10(np.linspace(0, 1, len(imfs)))
significance_threshold = 0.05

tmin = float(np.nanmin(time))
tmax = float(np.nanmax(time))
fig = plt.figure(figsize=(8, 9), constrained_layout=True)

gs = gridspec.GridSpec(
    len(imfs) + 2,
    1,
    height_ratios=[1]*len(imfs) + [2,2],
    figure=fig
)

for i, imf in enumerate(imfs):

    ax = fig.add_subplot(gs[i,0])

    if IMF_significance_levels[i] > significance_threshold:
        ax.set_facecolor("lightgray")

    ax.plot(time, imf, color=colors[i])

    ax.set_ylabel(f"IMF {i+1}")
    ax.set_xlim(tmin, tmax)

    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(3))
    if i < len(imfs)-1:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel("Time (s)")

    if i == 0:
        ax.set_title("(a) Empirical Mode Decomposition")

ax_hht = fig.add_subplot(gs[-2,0])

HHT_freq_bins = np.asarray(HHT_freq_bins)
HHT_power_spectrum = np.asarray(HHT_power_spectrum)
HHT_significance_level = np.asarray(HHT_significance_level)

mask = HHT_freq_bins <= f_nyq

ax_hht.plot(HHT_freq_bins[mask], HHT_power_spectrum[mask], color="black")
ax_hht.plot(HHT_freq_bins[mask], HHT_significance_level[mask], linestyle="--", color="green")

ax_hht.set_title("(b) HHT Marginal Spectrum")
ax_hht.set_xlabel("Frequency (Hz)")
ax_hht.set_ylabel("Power")

ax_hht.set_xlim(0, f_nyq)
ax_hht.set_ylim(bottom=0)

ax_hht.xaxis.set_minor_locator(AutoMinorLocator(5))
ax_hht.yaxis.set_minor_locator(AutoMinorLocator(5))

ax_top_hht = ax_hht.twiny()
ax_top_hht.set_xlim(ax_hht.get_xlim())

freq_ticks = ax_hht.get_xticks()
freq_ticks = freq_ticks[(freq_ticks > 0) & (freq_ticks <= f_nyq)]

period_labels = [f"{1/(f*60):.1f}" for f in freq_ticks]

ax_top_hht.set_xticks(freq_ticks)
ax_top_hht.set_xticklabels(period_labels)

ax_top_hht.set_xlabel("Period (minutes)")

ax_fft = fig.add_subplot(gs[-1,0])

for i,(xf,psd) in enumerate(psd_spectra_fft):

    xf = np.asarray(xf)
    psd = np.asarray(psd)

    mask = xf <= f_nyq

    ax_fft.plot(xf[mask], psd[mask], label=f"IMF {i+1}", color=colors[i])

    conf = np.asarray(confidence_levels_fft[i])

    if conf.shape == psd.shape:
        ax_fft.plot(xf[mask], conf[mask], linestyle="--", color=colors[i], alpha=0.7)


ax_fft.set_title("(c) FFT Spectra of IMFs")
ax_fft.set_xlabel("Frequency (Hz)")
ax_fft.set_ylabel("Power")

ax_fft.set_xlim(0, f_nyq)
ax_fft.set_ylim(bottom=0)

ax_fft.xaxis.set_minor_locator(AutoMinorLocator(5))
ax_fft.yaxis.set_minor_locator(AutoMinorLocator(5))

ax_top_fft = ax_fft.twiny()
ax_top_fft.set_xlim(ax_fft.get_xlim())

freq_ticks = ax_fft.get_xticks()
freq_ticks = freq_ticks[(freq_ticks > 0) & (freq_ticks <= f_nyq)]

period_labels = [f"{1/(f*60):.1f}" for f in freq_ticks]

ax_top_fft.set_xticks(freq_ticks)
ax_top_fft.set_xticklabels(period_labels)

ax_top_fft.set_xlabel("Period (minutes)")

ax_fft.legend(loc="upper right", fontsize=8)


plt.show()

