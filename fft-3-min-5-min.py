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

plt.rcParams.update({
    "font.size": 11,       
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
})

fig, axes = plt.subplots(
    len(datasets), 2,
    figsize=(7.0, 10.5),  
    sharex=False,
    sharey=False,
    constrained_layout=True
)
# Minutes
periods_min = [5, 3]
period_freqs = [1.0 / (p * 60.0) for p in periods_min]   # Hz

for row, (filename, map_key, label) in enumerate(datasets):
    for col, target_y in enumerate(target_ys):
        ax = axes[row, col]

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
        actual_y = slit_pos[y_index]

        v_time = data[y_index, :].astype(float)
        mask = np.isfinite(v_time) & np.isfinite(time)
        v_time = v_time[mask]
        time_clean = time[mask]

        signal = v_time - np.mean(v_time)

        dt = float(np.median(np.diff(time_clean)))
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
            time=time_clean,
            method="emd",
            siglevel=0.95,
        )

       
        # IMF 2 + IMF 3 in time domain
        imf23_signal = np.asarray(imfs[1], dtype=float) + np.asarray(imfs[2], dtype=float)
        imf23_signal = imf23_signal - np.mean(imf23_signal)

        # FFT of IMF2+IMF3
        n = len(imf23_signal)
        xf = np.fft.rfftfreq(n, d=dt)
        fft_vals = np.fft.rfft(imf23_signal)
        psd = (np.abs(fft_vals) ** 2) / n

        mask_fft = np.isfinite(xf) & np.isfinite(psd) & (xf >= 0) & (xf <= f_nyq)
        xf_plot = xf[mask_fft]
        psd_plot = psd[mask_fft]

        ax.plot(xf_plot, psd_plot, color="black", linewidth=1.0, label="IMF 2 + IMF 3")

        # Red vertical lines at 5, 3, 2 minutes
        for pf in period_freqs:
            if pf <= f_nyq:
                ax.axvline(pf, color="red", linestyle="--", linewidth=1.0, alpha=0.9)

        ax.set_xlim(0, f_nyq)
        ax.set_ylim(bottom=0)

        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.tick_params(axis="both", which="major", direction="out", length=5, width=0.9)
        ax.tick_params(axis="both", which="minor", direction="out", length=2.5, width=0.8)

        if col == 0:
            ax.set_ylabel("Power")

        if row == len(datasets) - 1:
            ax.set_xlabel("Frequency (Hz)")
        else:
            ax.set_xlabel("")
            ax.tick_params(labelbottom=False)

        if col == 0:
            ax.text(
                0.02, 0.88, label,
                transform=ax.transAxes,
                fontsize=10,
                ha="left",
                va="top"
            )

        if row == 0:
            ax.set_title(f'Y = {actual_y:.1f}"')

        if row == 0:
            ax_top = ax.twiny()
            ax_top.set_xlim(ax.get_xlim())

            top_ticks = []
            top_labels = []

            for pmin, pf in zip(periods_min, period_freqs):
                if pf <= f_nyq:
                    top_ticks.append(pf)
                    top_labels.append(f"{pmin}")

            ax_top.set_xticks(top_ticks)
            ax_top.set_xticklabels(top_labels)
            ax_top.set_xlabel("Period (min)")
            ax_top.tick_params(axis="x", which="major", direction="out", length=5, width=0.9)

        if row == 0 and col == 0:
            ax.legend(loc="upper right", frameon=False)

plt.savefig("images/fft_imf23_90_75.png", bbox_inches="tight")
plt.show()
