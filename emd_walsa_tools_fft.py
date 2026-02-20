import asdf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import matplotlib.gridspec as gridspec

from WaLSAtools import WaLSAtools  # type: ignore
from scipy.fft import rfft, rfftfreq

# -------------------------------------------------
# Load IRIS Doppler map
# -------------------------------------------------
filename = "IRIS_fitting_Si_IV_1394_20160520_131758.asdf"

FALLBACK_CADENCE_S = 16.2  # from OBS summary (Step Cad)

with asdf.open(filename) as af:
    dopp_map = af.tree["dopp_map"]
    meta = dict(dopp_map.meta)

# -------------------------------------------------
# Time axis (robust cadence)
# -------------------------------------------------
ny, nx = dopp_map.data.shape
cadence = float(meta.get("STEPT_AV", FALLBACK_CADENCE_S))
time = np.arange(nx) * cadence

# -------------------------------------------------
# Slit position (arcsec)
# -------------------------------------------------
slit_pos = float(meta["crval2"]) + float(meta["cdelt2"]) * (np.arange(ny) - float(meta["crpix2"]))
target_y = -89.0  # arcsec
y_index = int(np.argmin(np.abs(slit_pos - target_y)))

# -------------------------------------------------
# Doppler time series
# -------------------------------------------------
v_time = np.array(dopp_map.data[y_index, :], dtype=float)

mask = np.isfinite(v_time)
v_time = v_time[mask]
time = time[mask]

# If there are gaps, dt may not be constant; use median spacing for FFT freq axis
dt = float(np.median(np.diff(time)))
fs = 1.0 / dt
f_nyq = fs / 2.0

# Detrend mean (you can also detrend linearly if you want)
signal = v_time - np.nanmean(v_time)

# Optional: apply a window to reduce spectral leakage
window = np.hanning(len(signal))
signal_win = signal * window

# -------------------------------------------------
# EMD using WaLSAtools
# -------------------------------------------------
(
    _,
    _,
    _,
    _,
    _,
    imfs,
    IMF_significance_levels,
    _
) = WaLSAtools(
    signal=signal,
    time=time,
    method="emd",
    siglevel=0.95
)

# -------------------------------------------------
# FFT/PSD of each IMF (only up to Nyquist)
# -------------------------------------------------
n = len(time)
freq = rfftfreq(n, d=dt)  # Hz, from 0..Nyquist

psd_spectra = []
for imf in imfs:
    imf = np.array(imf, dtype=float)
    imf = imf - np.nanmean(imf)
    imf_win = imf * np.hanning(len(imf))
    spec = rfft(imf_win)
    psd = (np.abs(spec) ** 2) / n  # simple power
    psd_spectra.append(psd)

# -------------------------------------------------
# Plot settings
# -------------------------------------------------
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})
plt.rc("axes", linewidth=1.0)
plt.rc("lines", linewidth=1.3)

significance_threshold = 0.05
colors = plt.cm.tab10(np.linspace(0, 1, len(imfs)))

# -------------------------------------------------
# Plot IMFs + FFT (stacked)
# -------------------------------------------------
fig = plt.figure(figsize=(8., 10), constrained_layout=True)
gs = gridspec.GridSpec(len(imfs) + 1, 1, height_ratios=[1]*len(imfs) + [2], figure=fig)

for i, imf in enumerate(imfs):
    ax = fig.add_subplot(gs[i, 0])

    if IMF_significance_levels[i] > significance_threshold:
        ax.set_facecolor("lightgray")

    ax.plot(time, imf, color=colors[i])
    ax.set_ylabel(f"IMF {i+1}")
    ax.yaxis.set_label_coords(-0.12, 0.5)

    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(3))
    ax.tick_params(axis="both", which="major", direction="in", length=6)
    ax.tick_params(axis="both", which="minor", direction="in", length=3)

    if i < len(imfs) - 1:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel("Time (s)")

    if i == 0:
        ax.set_title("(a) Empirical Mode Decomposition (IMFs)")

# FFT panel
ax_fft = fig.add_subplot(gs[-1, 0])

for i, psd in enumerate(psd_spectra):
    ax_fft.plot(freq, psd, label=f"IMF {i+1}", color=colors[i])

# --- IMPORTANT: frequency axis limited to Nyquist ---
ax_fft.set_xlim(0, f_nyq)

ax_fft.set_title("(d) FFT Spectra of IMFs")
ax_fft.set_xlabel("Frequency (Hz)")
ax_fft.set_ylabel("Power")
ax_fft.xaxis.set_minor_locator(AutoMinorLocator(5))
ax_fft.yaxis.set_minor_locator(AutoMinorLocator(5))
ax_fft.tick_params(axis="both", which="major", direction="out", length=6)
ax_fft.tick_params(axis="both", which="minor", direction="out", length=3)
ax_fft.set_ylim(bottom=0)

# Helpful reference lines (solar standard)
# 3-min period ~180 s => 1/180 = 0.00556 Hz
ax_fft.axvline(1/180, color="gray", linestyle=":", label="3 min (5.56 mHz)")
# 5-min period ~300 s => 1/300 = 0.00333 Hz
ax_fft.axvline(1/300, color="gray", linestyle="--", label="5 min (3.33 mHz)")

ax_fft.legend(loc="upper right", fontsize=9)

print("---- Sampling diagnostics ----")
print(f"Cadence used (s):     {cadence:.3f}")
print(f"Median dt used (s):   {dt:.3f}")
print(f"Sampling rate (Hz):   {fs:.4f}")
print(f"Nyquist freq (Hz):    {f_nyq:.4f}")
print(f"Min resolvable P (s): {1/f_nyq:.1f}")

plt.show()


