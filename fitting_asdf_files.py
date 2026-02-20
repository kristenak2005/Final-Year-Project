#!/usr/bin/env python3
"""
Plot an IRIS sit-and-stare Doppler velocity time–distance map (time vs Solar-Y)
from an ASDF "dopp_map" product, with a physically correct time axis.

Your observation summary:
- Sit-and-stare (Steps: 696 x 0")
- Step cadence ~16.2 s (use this if metadata is missing)
- SJI 1400 cadence ~20 s (not used here)
"""

import os
import asdf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# -----------------------------
# User settings
# -----------------------------
filename = "IRIS_fitting_Si_IV_1403_20160520_131758.asdf"

# If metadata cadence is missing/incorrect, use the known cadence from OBS summary:
FALLBACK_CADENCE_S = 16.2  # seconds (Step Cad from your table)

# Choose either a specific Solar-Y to mark, or set to None to mark the mid-slit
target_y_arcsec = -84.0  # from your pointing x,y: 40", -84"
# target_y_arcsec = None

# Color scale for Doppler velocity (km/s)
vmin, vmax = -10, 10

# Output
save_dir = "/mnt/scratch/data/spruksk2/IRIS/python_output/IRIS_output/20160520_131758/images/"
save_name = "IRIS_Si_IV_1403_Doppler_Map.png"


# -----------------------------
# Helper: get cadence robustly
# -----------------------------
def infer_cadence_seconds(meta: dict, nx: int) -> float:
    """
    Try common IRIS metadata keys, otherwise fall back to the known cadence.
    """
    # Common places cadence may live in custom products
    candidate_keys = [
        "STEPT_AV",     # you used this earlier
        "step_cadence", # sometimes used in custom pipelines
        "cadence",
        "dt",
        "CDELT1",       # not usually time for spectra, but just in case
    ]

    for k in candidate_keys:
        val = meta.get(k, None)
        if val is None:
            continue
        try:
            val = float(val)
            # sanity: cadence should be positive and not huge
            if 0.1 <= val <= 300.0:
                return val
        except Exception:
            pass

    # If nothing usable found, fall back
    return float(FALLBACK_CADENCE_S)


# -----------------------------
# Load Doppler map
# -----------------------------
with asdf.open(filename) as af:
    dopp_map = af.tree["dopp_map"]

data = np.array(dopp_map.data)  # shape: (ny, nx)
meta = dict(dopp_map.meta) if hasattr(dopp_map, "meta") else {}

ny, nx = data.shape


# -----------------------------
# Build coordinates
# -----------------------------
# Time axis (seconds) — physically correct
cadence_s = infer_cadence_seconds(meta, nx)
t_array = np.arange(nx) * cadence_s

# Solar-Y axis (arcsec) from WCS-like metadata
# slit_pos = crval2 + cdelt2*(pixel_index - crpix2)
# (Your earlier code assumes these exist; we'll guard with a clear error.)
required = ["crval2", "cdelt2", "crpix2"]
missing = [k for k in required if k not in meta]
if missing:
    raise KeyError(
        f"Missing required WCS keys in dopp_map.meta: {missing}\n"
        f"Available meta keys include: {sorted(meta.keys())[:40]} ..."
    )

crval2 = float(meta["crval2"])
cdelt2 = float(meta["cdelt2"])
crpix2 = float(meta["crpix2"])

slit_pos = crval2 + cdelt2 * (np.arange(ny) - crpix2)

# Choose Y to mark
if target_y_arcsec is None:
    target_y_arcsec = float(np.median(slit_pos))

y_index = int(np.argmin(np.abs(slit_pos - target_y_arcsec)))
marked_y = float(slit_pos[y_index])

print("---- Derived axes ----")
print(f"Data shape (ny, nx): {ny}, {nx}")
print(f"Cadence used:        {cadence_s:.3f} s")
print(f"Duration:            {t_array.max():.1f} s  (~{t_array.max()/60:.1f} min)")
print(f"Solar-Y range:       {slit_pos.min():.2f}″ to {slit_pos.max():.2f}″")
print(f"Marked Solar-Y:      requested {target_y_arcsec:.2f}″ -> nearest pixel {marked_y:.2f}″ (index {y_index})")


# -----------------------------
# Plot Doppler map
# -----------------------------
fig, ax = plt.subplots(figsize=(11, 5.5))

norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

im = ax.imshow(
    data,
    cmap="coolwarm",
    norm=norm,
    extent=[t_array.min(), t_array.max(), slit_pos.min(), slit_pos.max()],
    origin="lower",
    aspect="auto",
    interpolation="nearest",
)

# Mark the chosen Y
ax.axhline(marked_y, color="lime", lw=1.6, linestyle="--", label=f"Y ≈ {marked_y:.1f}″")

# Labels
ax.set_xlabel("Time from start (s)")
ax.set_ylabel("Solar Y (arcsec)")
ax.set_title("IRIS Si IV 1403 Å — Doppler Velocity Time–Distance Map")

cbar = plt.colorbar(im, ax=ax, label=r"$v_{\mathrm{Dopp}}$ (km/s)", shrink=0.85)

ax.legend(loc="upper right")
plt.tight_layout()

# -----------------------------
# Save
# -----------------------------
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, save_name)
plt.savefig(save_path, dpi=300)
print(f"✅ Saved Doppler map to: {save_path}")

plt.show()
