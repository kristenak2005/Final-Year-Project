import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import asdf
from WaLSAtools import WaLSAtools  #


plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 11,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 11,
    'figure.titlesize': 12.5,
    'axes.grid': False,
})
plt.rc('axes', linewidth=1.0)
plt.rc('lines', linewidth=1.5)

filename = "IRIS_fitting_Si_IV_1394_20160520_131758.asdf"
# filename = "IRIS_fitting_Si_IV_1403_20160520_131758.asdf"
#filename = "IRIS_fitting_C_II_1334_20160520_131758.asdf"
# filename = "IRIS_fitting_C_II_1335_20160520_131758.asdf"
# filename = "IRIS_fitting_smooth_MgII_20160520_131758.asdf"

with asdf.open(filename) as af:
    dopp_map = af.tree["dopp_map"]
    meta = dopp_map.meta

data = np.asarray(dopp_map.data)

ny, nx = data.shape
cadence = meta.get("STEPT_AV", 1.0)
time = np.arange(nx) * cadence

slit_pos = meta["crval2"] + meta["cdelt2"] * (np.arange(ny) - meta["crpix2"])
target_y = -89.0
y_index = int(np.argmin(np.abs(slit_pos - target_y)))

v_time = data[y_index, :].astype(float)
mask = np.isfinite(v_time)
v_time = v_time[mask]
time = time[mask]

signal = v_time - np.mean(v_time)

# EMD
(
    _,
    _,
    _,
    _,
    _,
    imfs,
    _,
    _,
) = WaLSAtools(
    signal=signal,
    time=time,
    method="emd",
    siglevel=0.95,
)

# Add IMFs 3, 4 and 6 (Python index: 2, 3, 4)

indices = [2, 3, 4]
indices = [i for i in indices if i < len(imfs)]  # safety check

if len(indices) == 0:
    raise ValueError("No valid IMF indices found. Check how many IMFs were returned.")

combined_imf = np.sum([imfs[i] for i in indices], axis=0)

#plotting IMF
fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)

ax.plot(time, combined_imf, color="black")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Velocity")
ax.set_title("Reconstructed Signal (IMF 3 + 4 + 5)")

ax.set_xlim(time.min(), time.max())
ax.set_ylim(-10, 10)
ax.xaxis.set_minor_locator(AutoMinorLocator(5))
ax.yaxis.set_minor_locator(AutoMinorLocator(5))

ax.tick_params(axis="both", which="major", direction="in", length=6, width=1.0)
ax.tick_params(axis="both", which="minor", direction="in", length=3, width=1.0)

plt.show()
