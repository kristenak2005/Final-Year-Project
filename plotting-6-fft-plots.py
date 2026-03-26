import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import asdf

from WaLSAtools import WaLSAtools

# -------------------------
# Files and ASDF keys
# -------------------------
datasets = [
    ("IRIS_fitting_smooth_MgII_20160520_131758.asdf", "mgii_h_vdopp", "Mg II h"),
    ("IRIS_fitting_smooth_MgII_20160520_131758.asdf", "mgii_k_vdopp", "Mg II k"),
    ("IRIS_fitting_C_II_1334_20160520_131758.asdf", "asym_map", "C II 1334"),
    ("IRIS_fitting_C_II_1335_20160520_131758.asdf", "asym_map", "C II 1335"),
    ("IRIS_fitting_Si_IV_1394_20160520_131758.asdf", "dopp_map", "Si IV 1394"),
    ("IRIS_fitting_Si_IV_1403_20160520_131758.asdf", "dopp_map", "Si IV 1403"),
]

FALLBACK_CADENCE_S = 16.2
target_y = -95.0

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

fig, axes = plt.subplots(len(datasets), 1, figsize=(8, 14), sharex=False, constrained_layout=True)

for ax, (filename, map_key, label) in zip(axes, datasets):
    # -------------------------
    # Load data
    # -------------------------
    with asdf.open(filename) as af:
        data_map = af.tree[map_key]
        meta = dict(data_map.meta)
        data = np.asarray(data_map.data)

    ny, nx = data.shape
    cadence = 16.2
    time = np.arange(nx) * cadence

    slit_pos = float(meta["crval2"]) + float(meta["cdelt2"]) * (
        np.arange(ny) - float(meta["crpix2"])
    )
    y_index = int(np.argmin(np.abs(slit_pos - target_y)))

    v_time = data[y_index, :].astype(float)
    mask = np.isfinite(v_time) & np.isfinite(time)
    v_time = v_time[mask]
    time = time[mask]

    signal = v_time - np.mean(v_time)

    # -------------------------
    # Sampling
    # -------------------------
    dt = float(np.median(np.diff(time)))
    fs = 1.0 / dt
    f_nyq = fs / 2.0

    # -------------------------
    # Run WaLSAtools
    # -------------------------
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

    colors = plt.cm.tab10(np.linspace(0, 1, len(imfs)))

    # -------------------------
    # Plot FFT for this line
    # -------------------------
    for i, (xf, psd) in enumerate(psd_spectra_fft):
        xf = np.asarray(xf, dtype=float)
        psd = np.asarray(psd, dtype=float)

        mask_fft = np.isfinite(xf) & np.isfinite(psd) & (xf >= 0) & (xf <= f_nyq)
        xf_plot = xf[mask_fft]
        psd_plot = psd[mask_fft]
        ax.plot(xf_plot, psd_plot, color=colors[i], label=f"IMF {i+1}")

        conf = np.asarray(confidence_levels_fft[i], dtype=float)
        if conf.shape == psd.shape:
            conf_plot = conf[mask_fft]
            ax.plot(xf_plot, conf_plot, linestyle="--", color=colors[i], alpha=0.7)

    ax.set_ylabel("Power")
    ax.set_xlim(0, f_nyq)
    ax.set_ylim(bottom=0)

    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.tick_params(axis="both", which="major", direction="out", length=6, width=1.0)
    ax.tick_params(axis="both", which="minor", direction="out", length=3, width=1.0)

    ax.text(0.02, 0.88, label, transform=ax.transAxes)

    # Bottom axis label on every panel
    ax.set_xlabel("Frequency (Hz)")

    # -------------------------
    # Top axis: Period (minutes)
    # -------------------------
    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())

    freq_ticks = ax.get_xticks()
    freq_ticks = freq_ticks[(freq_ticks > 0) & (freq_ticks <= f_nyq)]

    period_labels = [f"{1/(f*60):.1f}" for f in freq_ticks]

    ax_top.set_xticks(freq_ticks)
    ax_top.set_xticklabels(period_labels)
    ax_top.set_xlabel("Period (minutes)")

    ax_top.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax_top.tick_params(axis="x", which="major", direction="out", length=6, width=1.0)
    ax_top.tick_params(axis="x", which="minor", direction="out", length=3, width=1.0)

    # Optional: legend only on first panel
    if label == "Mg II h":
        ax.legend(loc="upper right", fontsize=8)

plt.show()
