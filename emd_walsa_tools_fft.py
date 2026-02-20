import asdf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import matplotlib.gridspec as gridspec

from WaLSAtools import WaLSAtools, WaLSA_save_pdf  # type: ignore

# -------------------------------------------------
# Load IRIS Doppler map
# -------------------------------------------------
#filename = "IRIS_fitting_Si_IV_1403_20160520_131758.asdf"
#filename = "IRIS_fitting_C_II_1334_20160520_131758.asdf"
#filename = "IRIS_fitting_C_II_1335_20160520_131758.asdf"
filename = "IRIS_fitting_Si_IV_1394_20160520_131758.asdf"
#filename = "IRIS_fitting_Si_IV_1403_20160520_131758.asdf"
#filename = "IRIS_fitting_smooth_MgII_20160520_131758.asdf"

with asdf.open(filename) as af:
    dopp_map = af.tree["dopp_map"]
    meta = dopp_map.meta

# -------------------------------------------------
# Time axis
# -------------------------------------------------
ny, nx = dopp_map.data.shape
cadence = meta.get("STEPT_AV", 1.0)  # seconds
time = np.arange(nx) * cadence

# -------------------------------------------------
# Slit position (arcsec)
# -------------------------------------------------
slit_pos = meta["crval2"] + meta["cdelt2"] * (np.arange(ny) - meta["crpix2"])

target_y = -89.0  # arcsec
y_index = np.argmin(np.abs(slit_pos - target_y))

# -------------------------------------------------
# Doppler time series
# -------------------------------------------------
v_time = dopp_map.data[y_index, :]

mask = np.isfinite(v_time)
v_time = v_time[mask]
time = time[mask]

signal = v_time - np.mean(v_time)

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
# FFT of each IMF
# -------------------------------------------------
from scipy.fft import rfft, rfftfreq
from scipy.signal import periodogram

psd_spectra_fft = []
confidence_levels_fft = []
dt = time[1] - time[0]
n = len(time)

for imf in imfs:
    xf = rfftfreq(n, dt)
    psd = np.abs(rfft(imf))**2 / n
    psd_spectra_fft.append((xf, psd))
    confidence_levels_fft.append(np.full_like(xf, np.percentile(psd, 95)))

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
# Plot IMFs and FFT (panels a and d)
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

# Panel (d): FFT Spectra of IMFs
ax_fft = fig.add_subplot(gs[-1, 0])
for i, (xf, psd) in enumerate(psd_spectra_fft):
    ax_fft.plot(xf, psd, label=f"IMF {i+1}", color=colors[i])
    ax_fft.plot(xf, confidence_levels_fft[i], linestyle="--", color=colors[i])

ax_fft.set_title("(d) FFT Spectra of IMFs")
ax_fft.set_xlabel("Frequency (Hz)")
ax_fft.set_ylabel("Power")
ax_fft.xaxis.set_minor_locator(AutoMinorLocator(5))
ax_fft.yaxis.set_minor_locator(AutoMinorLocator(5))
ax_fft.tick_params(axis="both", which="major", direction="out", length=6)
ax_fft.tick_params(axis="both", which="minor", direction="out", length=3)
ax_fft.set_xlim(0, 0.5)
#ax_fft.set_yslim(0, 100)
ax_fft.set_ylim(bottom=0)

# Optional: add predefined vertical frequency lines
pre_defined_freq = [2, 5, 10, 12, 15, 18, 25, 33]
for freq in pre_defined_freq:
    ax_fft.axvline(x=freq, color='gray', linestyle=':')

# -------------------------------------------------
# Save figure
# -------------------------------------------------
#pdf_path = "Figures/IRIS_SiIV_IMFs_and_FFT.pdf"
#WaLSA_save_pdf(fig, pdf_path, color_mode="CMYK", dpi=300, bbox_inches="tight", pad_inches=0 )

plt.show()

