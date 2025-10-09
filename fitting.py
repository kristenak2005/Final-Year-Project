import asdf
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.gridspec import GridSpec
import numpy as np

# --- File to open ---
filename = "IRIS_fitting_Si_IV_1403_20160520_131758.asdf"

# --- Open ASDF file and extract maps ---
with asdf.open(filename) as af:
    int_map = af.tree["int_map"]
    dopp_map = af.tree["dopp_map"]
    width_map = af.tree["width_map"]
    vnt_map = af.tree["vnt_map"]
    asym_map = af.tree["asym_map"]

# --- Raster time and slit position ---
cadence = int_map.meta.get('STEPT_AV', 1.0)  # seconds per step
nx, ny = int_map.data.shape[1], int_map.data.shape[0]
#t_array = np.arange(nx) * cadence
t_array = np.linspace(0, 10000, nx)
slit_pos = int_map.meta['crval2'] + int_map.meta['cdelt2'] * (np.arange(ny) - int_map.meta['crpix2'])

# --- Create figure ---
fig = plt.figure(constrained_layout=True, figsize=(10, 12))
plt.rcParams['font.size'] = 10
gs = GridSpec(nrows=5, ncols=1, hspace=0.05, wspace=0.05)
gs.update(left=0.08, right=0.92, bottom=0.05, top=0.95)

# --- 1. Intensity ---
ax1 = fig.add_subplot(gs[0, 0])
upr_bnd = np.nanpercentile(int_map.data, 99)
im1 = ax1.imshow(int_map.data, cmap='Reds_r', norm=colors.Normalize(vmin=0, vmax=upr_bnd),
                 extent=[t_array.min(), t_array.max(), slit_pos.min(), slit_pos.max()],
                 origin='lower', aspect='auto')
ax1.set_ylabel("Solar Y (arcsec)")
ax1.set_xticklabels([])
plt.colorbar(im1, ax=ax1, label='a) Intensity', shrink=0.6)

# --- 2. Asymmetry ---
ax2 = fig.add_subplot(gs[1, 0])
im2 = ax2.imshow(asym_map.data, cmap='seismic', norm=colors.Normalize(vmin=-1, vmax=1),
                 extent=[t_array.min(), t_array.max(), slit_pos.min(), slit_pos.max()],
                 origin='lower', aspect='auto')
ax2.set_ylabel(" ")
ax2.set_xticklabels([])
plt.colorbar(im2, ax=ax2, label='b) RB Asymmetry', shrink=0.6)

# --- 3. Doppler ---
ax3 = fig.add_subplot(gs[2, 0])
im3 = ax3.imshow(dopp_map.data, cmap='coolwarm', norm=colors.Normalize(vmin=-10, vmax=10),
                 extent=[t_array.min(), t_array.max(), slit_pos.min(), slit_pos.max()],
                 origin='lower', aspect='auto')
ax3.set_ylabel("Solar Y (arcsec)")
ax3.set_xticklabels([])
plt.colorbar(im3, ax=ax3, label='c) Velocity (km/s)', shrink=0.6)

# --- 4. Width ---
ax4 = fig.add_subplot(gs[3, 0])
im4 = ax4.imshow(width_map.data, cmap='cubehelix', norm=colors.Normalize(vmin=0, vmax=0.1),
                 extent=[t_array.min(), t_array.max(), slit_pos.min(), slit_pos.max()],
                 origin='lower', aspect='auto')
ax4.set_ylabel(" ")
ax4.set_xticklabels([])
plt.colorbar(im4, ax=ax4, label='d) Width (Å)', shrink=0.6)

# --- 5. Nonthermal velocity ---
ax5 = fig.add_subplot(gs[4, 0])
im5 = ax5.imshow(vnt_map.data, cmap='inferno', norm=colors.Normalize(vmin=0, vmax=30),
                 extent=[t_array.min(), t_array.max(), slit_pos.min(), slit_pos.max()],
                 origin='lower', aspect='auto')
ax5.set_ylabel(" ")
ax5.set_xlabel("Time from raster start (s)")
plt.colorbar(im5, ax=ax5, label='e) v_nt (km/s)', shrink=0.6)

# --- Overall title ---
plt.suptitle(f"IRIS Si IV 1403Å — {int_map.meta.get('date-obs', 'Unknown date')}")
plt.show()
