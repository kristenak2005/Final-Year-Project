import asdf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap
from WaLSAtools import WaLSAtools

# --------------------------------------------------------------------------
# Load IRIS Doppler velocity map
# --------------------------------------------------------------------------
filename = "IRIS_fitting_Si_IV_1394_20160520_131758.asdf"
with asdf.open(filename) as af:
    dopp_map = af.tree["dopp_map"]
    meta = dopp_map.meta

nx, ny = dopp_map.data.shape[1], dopp_map.data.shape[0]
cadence = meta.get("STEPT_AV", 1.0)  # seconds per raster step
sampling_rate = 1.0 / cadence

# Build time and slit coordinates
time = np.linspace(0, nx * cadence, nx)
slit_pos = meta["crval2"] + meta["cdelt2"] * (np.arange(ny) - meta["crpix2"])

# Choose solar Y position
desired_y = -89
y_index = np.argmin(np.abs(slit_pos - desired_y))
actual_y = slit_pos[y_index]

# Extract Doppler velocity line
signal = np.nan_to_num(dopp_map.data[y_index, :], nan=np.nanmean(dopp_map.data))

# --------------------------------------------------------------------------
# Run Morlet wavelet analysis
# --------------------------------------------------------------------------
(
    wavelet_power_morlet,
    wavelet_periods_morlet,
    wavelet_significance_morlet,
    coi_morlet,
    global_power_morlet,
    global_conf_morlet,
    rgws_morlet_power,
) = WaLSAtools(
    signal=signal,
    time=time,
    method="wavelet",
    siglevel=0.95,
    apod=0.1,
    mother="morlet",
    GWS=True,
    RGWS=True,
)

# --------------------------------------------------------------------------
# Create figure
# --------------------------------------------------------------------------
fig, ax_inset_l = plt.subplots(figsize=(10, 6))

# --------------------------------------------------------------------------
# Colormap (use your IDL table if present)
# --------------------------------------------------------------------------
try:
    rgb_values = np.loadtxt("Color_Tables/idl_colormap_20_modified.txt") / 255.0
    idl_colormap_20 = ListedColormap(rgb_values)
except Exception:
    idl_colormap_20 = plt.cm.inferno

cmap = plt.get_cmap(idl_colormap_20)
colorbar_label = "Power (%) | Morlet Wavelet"
ylabel = "Period (s)"
xlabel = "Time (s)"
pre_defined_freq = [2, 5, 10, 12, 15, 18, 25, 33]

# --------------------------------------------------------------------------
# Process wavelet output
# --------------------------------------------------------------------------
power = np.copy(wavelet_power_morlet)
power[power < 0] = 0
power = 100 * power / np.nanmax(power)
t = time
periods = wavelet_periods_morlet
coi = coi_morlet
sig_slevel = wavelet_significance_morlet
dt = cadence

# Remove unused space beyond COI
max_period = np.max(coi)
cutoff_index = np.argmax(periods > max_period)
if cutoff_index > 0 and cutoff_index <= len(periods):
    power = power[:cutoff_index, :]
    periods = periods[:cutoff_index]
    sig_slevel = sig_slevel[:cutoff_index, :]

# --------------------------------------------------------------------------
# Plot wavelet power spectrum
# --------------------------------------------------------------------------
levels = np.linspace(0, 100, 100)
CS = ax_inset_l.contourf(t, periods, power, levels=levels, cmap=cmap, extend="neither")

# 95% significance contour
ax_inset_l.contour(t, periods, sig_slevel, levels=[1], colors="k", linewidths=[0.6])

# Cone of influence
ax_inset_l.plot(t, coi, "-k", lw=1.15)
ax_inset_l.fill(
    np.concatenate([t, t[-1:] + dt, t[-1:] + dt, t[:1] - dt, t[:1] - dt]),
    np.concatenate([coi, [1e-9], [np.max(periods)], [np.max(periods)], [1e-9]]),
    color="none",
    edgecolor="k",
    alpha=1,
    hatch="xx",
)

# --------------------------------------------------------------------------
# Axis formatting
# --------------------------------------------------------------------------
ax_inset_l.set_yscale("log", base=10)
ax_inset_l.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
ax_inset_l.invert_yaxis()
ax_inset_l.set_xlim([t.min(), t.max()])
ax_inset_l.set_xlabel(xlabel)
ax_inset_l.set_ylabel(ylabel)
ax_inset_l.tick_params(axis="both", which="both", direction="out", length=8, width=1.5, top=True, right=True)
ax_inset_l.set_title(f"IRIS Si IV 1394 Å Morlet Wavelet Power at Y ≈ {actual_y:.1f} arcsec")

# --------------------------------------------------------------------------
# Secondary Y-axis (frequency)
# --------------------------------------------------------------------------
ax_freq = ax_inset_l.twinx()
min_frequency = 1 / np.max(periods)
max_frequency = 1 / np.min(periods)
ax_freq.set_yscale("log", base=10)
ax_freq.set_ylim([max_frequency, min_frequency])
ax_freq.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
ax_freq.invert_yaxis()
ax_freq.set_ylabel("Frequency (Hz)")
ax_freq.tick_params(axis="both", which="major", length=8, width=1.5)

# --------------------------------------------------------------------------
# Colorbar
# --------------------------------------------------------------------------
divider = make_axes_locatable(ax_inset_l)
cax = inset_axes(ax_inset_l, width="100%", height="5%", loc="upper center", borderpad=-1.4)
cbar = plt.colorbar(CS, cax=cax, orientation="horizontal")
cbar.set_label(colorbar_label, labelpad=8)
cbar.ax.tick_params(direction="out", top=True, labeltop=True, bottom=False, labelbottom=False)
cbar.ax.xaxis.set_label_position("top")
cbar.set_ticks([0, 20, 40, 60, 80, 100])
cbar.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))
cbar.ax.xaxis.set_minor_locator(AutoMinorLocator(4))

# --------------------------------------------------------------------------
# Predefined frequency lines
# --------------------------------------------------------------------------
for freqin in pre_defined_freq:
    ax_inset_l.axhline(y=1 / freqin, color="#32CD32", linewidth=0.7)

# --------------------------------------------------------------------------
# Show plot only (no saving)
# --------------------------------------------------------------------------
plt.tight_layout()
plt.show()
