import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import asdf
from WaLSAtools import WaLSAtools  # type: ignore

# Plot style
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 11,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 12.5,
    'axes.grid': False,
})
plt.rc('axes', linewidth=1.0)
plt.rc('lines', linewidth=1.4)

# User settings
cadence = 16.2
target_y = -900
imf_indices = [1]

# Files and map names in orde (Mg > C > Si)
datasets = [
    {
        "filename": "IRIS_fitting_smooth_MgII_20160520_131758.asdf",
        "map_key": "mgii_h_vdopp",
        "label": "(a) Mg II h",
    },
    {
        "filename": "IRIS_fitting_smooth_MgII_20160520_131758.asdf",
        "map_key": "mgii_k_vdopp",
        "label": "(b) Mg II k",
    },
    {
        "filename": "IRIS_fitting_C_II_1334_20160520_131758.asdf",
        "map_key": "dopp_map",
        "label": "(c) C II 1334",
    },
    {
        "filename": "IRIS_fitting_C_II_1335_20160520_131758.asdf",
        "map_key": "dopp_map",
        "label": "(d) C II 1335",
    },
    {
        "filename": "IRIS_fitting_Si_IV_1394_20160520_131758.asdf",
        "map_key": "dopp_map",
        "label": "(e) Si IV 1394",
    },
    {
        "filename": "IRIS_fitting_Si_IV_1403_20160520_131758.asdf",
        "map_key": "dopp_map",
        "label": "(f) Si IV 1403",
    },
]

def get_reconstructed_signal(filename, map_key, target_y, cadence, imf_indices):
    with asdf.open(filename) as af:
        mp = af.tree[map_key]
        meta = dict(mp.meta)
        data = np.asarray(mp.data, dtype=float)

    ny, nx = data.shape
    time = np.arange(nx, dtype=float) * cadence / 60.0

    slit_pos = float(meta["crval2"]) + float(meta["cdelt2"]) * (
        np.arange(ny) - float(meta["crpix2"])
    )
    y_index = int(np.argmin(np.abs(slit_pos - target_y)))

  
    v_time = data[y_index, :]
    mask = np.isfinite(v_time) & np.isfinite(time)
    v_time = v_time[mask]
    time = time[mask]

    signal = v_time - np.mean(v_time)

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
        time=time * 60.0,
        method="emd",
        siglevel=0.95,
    )

    valid_indices = [i for i in imf_indices if i < len(imfs)]
    if len(valid_indices) == 0:
        raise ValueError(f"No valid IMF indices found for {filename}")

    combined_imf = np.sum([imfs[i] for i in valid_indices], axis=0)

    return time, combined_imf, slit_pos[y_index]

fig, axes = plt.subplots(6, 1, figsize=(6, 12), constrained_layout=True)

for ax, ds in zip(axes, datasets):
    time, combined_imf, actual_y = get_reconstructed_signal(
        ds["filename"], ds["map_key"], target_y, cadence, imf_indices
    )

    ax.plot(time, combined_imf, color="black", linewidth=1.4)

    ax.set_title(ds["label"])
    ax.set_xlim(time.min(), time.max())
    ax.set_ylim(-10, 10)


    v_time = data[y_index, :]
    mask = np.isfinite(v_time) & np.isfinite(time)
    v_time = v_time[mask]
    time = time[mask]

    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))

    ax.tick_params(axis="both", which="major", direction="in", length=6, width=1.0)
    ax.tick_params(axis="both", which="minor", direction="in", length=3, width=1.0)

    ax.text(
        0.02, 0.92,
        f"Y = {actual_y:.1f}\"",
        transform=ax.transAxes,
        fontsize=9,
        ha="left",
        va="top"
    )

# axis labels
axes[-1].set_xlabel("Time (min)")

for ax in axes:
    ax.set_ylabel("Velocity (km s$^{-1}$)")

plt.show()
