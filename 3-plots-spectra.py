import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

filename = "iris_l2_20160520_131758_3620110603_raster_t000_r00000.fits"

# Penumbra
target_y_values = [-65.0]

# Spectra lines
HDU_MG   = 8
HDU_CII  = 1
HDU_SIIV = 5

find_brightest_x_at_each_y = True
x_index_manual = 45

siiv_line_center = 1402.8
siiv_half_width = 0.20

mg_lines   = [2796.35, 2803.53]
cii_lines  = [1334.53, 1335.71]
siiv_lines = [1402.8]

def build_axis(n, crval, cdelt, crpix):
    pix = np.arange(n)
    return crval + (pix + 1 - crpix) * cdelt

def get_cube_and_axes(hdul, hdu_index):
    hdu = hdul[hdu_index]
    hdr = hdu.header
    data = np.asarray(hdu.data, dtype=float)

    if data.ndim != 3:
        raise ValueError(f"HDU {hdu_index} does not have 3 dimensions. Shape = {data.shape}")

    nx, ny, nwave = data.shape

    wavelength = build_axis(nwave, hdr["CRVAL1"], hdr["CDELT1"], hdr["CRPIX1"])
    solar_y    = build_axis(ny,    hdr["CRVAL2"], hdr["CDELT2"], hdr["CRPIX2"])
    solar_x    = build_axis(nx,    hdr["CRVAL3"], hdr["CDELT3"], hdr["CRPIX3"])
      return data, wavelength, solar_x, solar_y


def nearest_index(axis, value):
    return int(np.argmin(np.abs(axis - value)))


def brightest_x_from_line(data, wavelength, y_index, line_center, half_width=0.2):
    mask = (wavelength >= line_center - half_width) & (wavelength <= line_center + half_width)
    if not np.any(mask):
        raise ValueError("No wavelength points found in the requested line window.")
    x_profile = np.nansum(data[:, y_index, :][:, mask], axis=1)
    return int(np.nanargmax(x_profile))


with fits.open(filename) as hdul:
    mg_data,   mg_wave,   mg_x,   mg_y   = get_cube_and_axes(hdul, HDU_MG)
    cii_data,  cii_wave,  cii_x,  cii_y  = get_cube_and_axes(hdul, HDU_CII)
    siiv_data, siiv_wave, siiv_x, siiv_y = get_cube_and_axes(hdul, HDU_SIIV)

    target_y_arcsec = target_y_values[0]
    y_index = nearest_index(siiv_y, target_y_arcsec)

    if find_brightest_x_at_each_y:
        x_index = brightest_x_from_line(
            siiv_data, siiv_wave, y_index,
            line_center=siiv_line_center,
            half_width=siiv_half_width
        )
    else:
        x_index = x_index_manual

    mg_spec   = mg_data[x_index, y_index, :]
    cii_spec  = cii_data[x_index, y_index, :]
    siiv_spec = siiv_data[x_index, y_index, :]


    fig, axes = plt.subplots(3, 1, figsize=(5, 8))

    # Mg II
    ax = axes[0]
    ax.plot(mg_wave, mg_spec, lw=1.0, color="black")
    for line in mg_lines:
        ax.axvline(line, color="red", linestyle="--", linewidth=1.0)
    ax.set_ylim(bottom=0)
    ax.set_xlim(2794, 2806)
    ax.set_ylabel("Intensity")
    ax.set_title("Mg II k")
    ax.tick_params(labelbottom=False)

    # C II
    ax = axes[1]
    ax.plot(cii_wave, cii_spec, lw=1.0, color="black")
    for line in cii_lines:
        ax.axvline(line, color="red", linestyle="--", linewidth=1.0)
    ax.set_ylim(bottom=0)
    ax.set_xlim(1333.5, 1336.0)
    ax.set_ylabel("Intensity")
    ax.set_title("C II")
    ax.tick_params(labelbottom=False)

    # Si IV
    ax = axes[2]
    ax.plot(siiv_wave, siiv_spec, lw=1.0, color="black")
    for line in siiv_lines:
        ax.axvline(line, color="red", linestyle="--", linewidth=1.0)
    ax.set_ylim(bottom=0)
    ax.set_xlim(1402.1, 1403.5)
    ax.set_xlabel("Wavelength (Å)")
    ax.set_ylabel("Intensity")
    ax.set_title("Si IV 1403 (Å)")

    plt.tight_layout()
    plt.savefig("images/3-plots-y-65-clean.png", bbox_inches="tight")
    plt.show()


