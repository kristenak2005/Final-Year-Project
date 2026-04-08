import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import asdf

from WaLSAtools import WaLSAtools

datasets = [
    ("IRIS_fitting_smooth_MgII_20160520_131758.asdf", "mgii_h_vdopp", "Mg II h"),
    ("IRIS_fitting_smooth_MgII_20160520_131758.asdf", "mgii_k_vdopp", "Mg II k"),
    ("IRIS_fitting_C_II_1334_20160520_131758.asdf", "dopp_map", "C II 1334"),
    ("IRIS_fitting_C_II_1335_20160520_131758.asdf", "dopp_map", "C II 1335"),
    ("IRIS_fitting_Si_IV_1394_20160520_131758.asdf", "dopp_map", "Si IV 1394"),
    ("IRIS_fitting_Si_IV_1403_20160520_131758.asdf", "dopp_map", "Si IV 1403"),
]

target_ys = [-90.0, -75.0]
FALLBACK_CADENCE_S = 16.2


plt.rcParams.update({
    "font.size": 11,        # base size
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
})


fig, axes = plt.subplots(
    len(datasets), 2,
    figsize=(7, 10.5),
    sharex=False,
    sharey=False,
    constrained_layout=True
)

for row, (filename, map_key, label) in enumerate(datasets):
    for col, target_y in enumerate(target_ys):
        ax = axes[row, col]

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

        v_time = data[y_index, :].astype(float)
        mask = np.isfinite(v_time) & np.isfinite(time)
        v_time = v_time[mask]
        time = time[mask]

        signal = v_time - np.nanmean(v_time)

        dt = float(np.median(np.diff(time)))
        fs = 1.0 / dt
        f_nyq = fs / 2.0

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

        HHT_freq_bins = np.asarray(HHT_freq_bins, dtype=float)
        HHT_power_spectrum = np.asarray(HHT_power_spectrum, dtype=float)
        HHT_significance_level = np.asarray(HHT_significance_level, dtype=float)

        mask_hht = (
            np.isfinite(HHT_freq_bins)
            & np.isfinite(HHT_power_spectrum)
            & np.isfinite(HHT_significance_level)
            & (HHT_freq_bins >= 0)
            & (HHT_freq_bins <= f_nyq)
        )
        freq_plot = HHT_freq_bins[mask_hht]
        power_plot = HHT_power_spectrum[mask_hht]
        sig_plot = HHT_significance_level[mask_hht]

        ax.plot(freq_plot, power_plot, color="black", label="HHT")
        ax.plot(freq_plot, sig_plot, linestyle="--", color="green", alpha=0.8)

        # -------------------------
        # Red vertical lines (3 & 5 min)
        # -------------------------
        periods_min = [5, 3]
        period_freqs = [1.0 / (p * 60.0) for p in periods_min]

        for pf in period_freqs:
            if pf <= f_nyq:
                 ax.axvline(pf, color="red", linestyle="--", linewidth=1.2, alpha=0.9)

        ax.set_xlim(0, f_nyq)
        ax.set_ylim(bottom=0)

        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.tick_params(axis="both", which="major", direction="out", length=6, width=1.0)
        ax.tick_params(axis="both", which="minor", direction="out", length=3, width=1.0)

        if col == 0:
            ax.set_ylabel("Power")

        if row == len(datasets) - 1:
            ax.set_xlabel("Frequency (Hz)")
        else:
            ax.tick_params(labelbottom=False)

        if col == 0:
            ax.text(0.02, 0.88, label, transform=ax.transAxes)

        if row == 0:
            ax.set_title(f'Y = {actual_y:.1f}"')

        if row == 0:
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

        if row == 0 and col == 0:
            ax.legend(loc="upper right", fontsize=8)
plt.savefig("images/hht--90--75.png", bbox_inches="tight")
plt.show()
