import os
import asdf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})


filename = "IRIS_fitting_Si_IV_1403_20160520_131758.asdf"
#filename = "IRIS_fitting_C_II_1334_20160520_131758.asdf"
FALLBACK_CADENCE_S = 16.2 
target_y_arcsec = -90 
# target_y_arcsec = None
vmin, vmax = -10, 10

save_dir = "/mnt/scratch/data/spruksk2/IRIS/python_output/IRIS_output/20160520_131758/images/"
save_name = "IRIS_Si_IV_1403_Doppler_Map.png"

def infer_cadence_seconds(meta: dict, nx: int) -> float:
    candidate_keys = [
        "STEPT_AV",    
        "step_cadence", 
        "cadence",
        "dt",
        "CDELT1",       
    ]

    for k in candidate_keys:
        val = meta.get(k, None)
        if val is None:
            continue
        try:
            val = float(val)
            if 0.1 <= val <= 300.0:
                return val
        except Exception:
            pass

    return float(FALLBACK_CADENCE_S)

with asdf.open(filename) as af:
    dopp_map = af.tree["dopp_map"]

data = np.array(dopp_map.data) 
meta = dict(dopp_map.meta) if hasattr(dopp_map, "meta") else {}

ny, nx = data.shape

cadence_s = infer_cadence_seconds(meta, nx)
t_array = np.arange(nx) * cadence_s
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

fig, ax = plt.subplots(figsize=(10, 6))

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
# Umbra
ax.axhline(marked_y, color="black", lw=2.5, linestyle="-", label=f"Umbra ≈ {marked_y:.1f}″")

# Penumbra
qs_y_arcsec = -75.0
qs_index = int(np.argmin(np.abs(slit_pos - qs_y_arcsec)))
qs_marked_y = float(slit_pos[qs_index])

ax.axhline(qs_marked_y, color="darkgreen", lw=2.5, linestyle="-", label=f"Penumbra ≈ {qs_marked_y:.1f}″")


ax.set_xlabel("Time from start (s)")
ax.set_ylabel("Solar Y (arcsec)")
#Si IV 1403 Å  Doppler Velocity TimeDistance Map")

cbar = plt.colorbar(im, ax=ax, label="Doppler Velocity (km/s)", shrink=0.85)

ax.legend(loc="upper right")
plt.tight_layout()

os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, save_name)
plt.savefig(save_path)
print(f" Saved Doppler map to: {save_path}")

plt.show()


