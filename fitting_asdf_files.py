import asdf
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import os

# --- Load Doppler map ---
filename = "IRIS_fitting_Si_IV_1403_20160520_131758.asdf"
with asdf.open(filename) as af:
    dopp_map = af.tree["dopp_map"]

# --- Build coordinates ---
nx, ny = dopp_map.data.shape[1], dopp_map.data.shape[0]
t_array = np.linspace(0, 10000, nx)
meta = dopp_map.meta
slit_pos = meta["crval2"] + meta["cdelt2"] * (np.arange(ny) - meta["crpix2"])

# --- Plot Doppler map ---
fig, ax = plt.subplots(figsize=(10, 5))
norm = colors.Normalize(vmin=-10, vmax=10)
im = ax.imshow(
    dopp_map.data,
    cmap="coolwarm",
    norm=norm,
    extent=[t_array.min(), t_array.max(), slit_pos.min(), slit_pos.max()],
    origin="lower",
    aspect="auto"
)

# --- Mark oscillatory region ---
osc_y = 90  # arcsec
ax.axhline(osc_y, color="lime", lw=1.3, linestyle="--", label=f"Y ≈ {osc_y}″")

# --- Formatting ---
ax.set_xlabel("Time from raster start (s)")
ax.set_ylabel("Solar Y (arcsec)")
plt.colorbar(im, ax=ax, label=r"$v_{\mathrm{Dopp}}$ (km/s)", shrink=0.6)
plt.title("IRIS Si IV 1403 Å — Doppler Velocity Map")
ax.legend(loc="upper right")
plt.tight_layout()

# --- Save plot ---
save_dir = "/mnt/scratch/data/spruksk2/IRIS/python_output/IRIS_output/20160520_131758/images/"
os.makedirs(save_dir, exist_ok=True)  # create the folder if it doesn't exist
save_path = os.path.join(save_dir, "IRIS_Si_IV_1403_Doppler_Map.png")
plt.savefig(save_path, dpi=300)
print(f"✅ Saved Doppler map to: {save_path}")

plt.show()
