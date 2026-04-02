import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# ============================================================
# User settings
# ============================================================
filename = "iris_l2_20160520_131758_3620110603_raster_t000_r00000.fits"

# Target Y positions for the three rows
target_y_values = [-90.0, -65.0, -40.0]

# HDU mapping from your file
HDU_MG   = 8   # Mg II k
HDU_CII  = 1   # C II 1336 window
HDU_SIIV = 5   # Si IV 1403

# If True, choose X from brightest Si IV 1403 emission at each Y
find_brightest_x_at_each_y = True
x_index_manual = 45

# Search window for choosing brightest Si IV pixel
siiv_line_center = 1402.77
siiv_half_width = 0.20  # Å


# ============================================================
# Helper functions
# ============================================================
def build_axis(n, crval, cdelt, crpix):
    """Build world-coordinate axis from FITS WCS keywords."""
    pix = np.arange(n)
    return crval + (pix + 1 - crpix) * cdelt


def get_cube_and_axes(hdul, hdu_index):
    """
    Return data cube and axes for an IRIS spectral HDU.
    Assumes data shape is (x, y, wavelength).
    """
    hdu = hdul[hdu_index]
    hdr = hdu.header
    data = np.asarray(hdu.data, dtype=float)

    if data.ndim != 3:
        raise ValueError(f"HDU {hdu_index} does not have 3 dimensions. Shape = {data.shape}")

    nx, ny, nwave = data.shape

    nx, ny, nwave = data.shape

    wavelength = build_axis(nwave, hdr["CRVAL1"], hdr["CDELT1"], hdr["CRPIX1"])
    solar_y    = build_axis(ny,    hdr["CRVAL2"], hdr["CDELT2"], hdr["CRPIX2"])
    solar_x    = build_axis(nx,    hdr["CRVAL3"], hdr["CDELT3"], hdr["CRPIX3"])

    return data, wavelength, solar_x, solar_y


def nearest_index(axis, value):
    return int(np.argmin(np.abs(axis - value)))


def brightest_x_from_line(data, wavelength, y_index, line_center, half_width=0.2):
    """
    Find the X pixel with maximum line intensity at the chosen Y.
    """
    mask = (wavelength >= line_center - half_width) & (wavelength <= line_center + half_width)
    if not np.any(mask):
        raise ValueError("No wavelength points found in the requested line window.")

    x_profile = np.nansum(data[:, y_index, :][:, mask], axis=1)
    return int(np.nanargmax(x_profile))


# ============================================================
# Open FITS file and inspect contents
# ============================================================
with fits.open(filename) as hdul:
    print("===== FITS structure =====")
    hdul.info()

    print("\n===== Spectral windows in PRIMARY header =====")
    hdr0 = hdul[0].header
    for i in range(1, 9):
        print(f"HDU {i}: {hdr0.get(f'TDESC{i}')}   {hdr0.get(f'TWAVE{i}')}")

    # Load only the three needed windows: HDU 8, HDU 1, HDU 5
    mg_data,   mg_wave,   mg_x,   mg_y   = get_cube_and_axes(hdul, HDU_MG)
    cii_data,  cii_wave,  cii_x,  cii_y  = get_cube_and_axes(hdul, HDU_CII)
    siiv_data, siiv_wave, siiv_x, siiv_y = get_cube_and_axes(hdul, HDU_SIIV)

    # Basic consistency check
    if not (np.allclose(mg_x, siiv_x) and np.allclose(mg_y, siiv_y) and
            np.allclose(cii_x, siiv_x) and np.allclose(cii_y, siiv_y)):
        print("\nWARNING: Spatial axes are not identical between HDUs.")
        print("The script will still use indices chosen from the Si IV window.")
    # ========================================================
    # Make 3x3 plot grid
    # ========================================================
    fig, axes = plt.subplots(3, 3, figsize=(16, 11))

    for row, target_y_arcsec in enumerate(target_y_values):
        # Find nearest Y pixel using Si IV window
        y_index = nearest_index(siiv_y, target_y_arcsec)

        # Choose X
        if find_brightest_x_at_each_y:
            x_index = brightest_x_from_line(
                siiv_data, siiv_wave, y_index,
                line_center=siiv_line_center,
                half_width=siiv_half_width
            )
        else:
            x_index = x_index_manual

        x_arcsec = siiv_x[x_index]
        y_arcsec = siiv_y[y_index]

        print(f"\nRequested Y = {target_y_arcsec:.1f} arcsec")
        print(f"Nearest pixel: X = {x_index}, Y = {y_index}")
        print(f"Solar coords : X = {x_arcsec:.2f} arcsec, Y = {y_arcsec:.2f} arcsec")

        # Extract spectra
        mg_spec   = mg_data[x_index, y_index, :]
        cii_spec  = cii_data[x_index, y_index, :]
        siiv_spec = siiv_data[x_index, y_index, :]

        # ----------------------------------------------------
        # Column 1: HDU 8 = Mg II k
        # ----------------------------------------------------
        ax = axes[row, 0]
        ax.plot(mg_wave, mg_spec, lw=1.0)
        ax.set_ylim(bottom=0)
        ax.set_xlim(2794, 2806)
        ax.set_xlabel("Wavelength (Å)")
        ax.set_ylabel("Intensity")
        ax.set_title(f"Mg II k\nX={x_arcsec:.2f}\", Y={y_arcsec:.2f}\"")

        # ----------------------------------------------------
        # Column 2: HDU 1 = C II
        # ----------------------------------------------------
        ax = axes[row, 1]
        ax.plot(cii_wave, cii_spec, lw=1.0)
        ax.set_ylim(bottom=0)
        ax.set_xlim(1333.5, 1336.0)
        ax.set_xlabel("Wavelength (Å)")
        ax.set_ylabel("Intensity")
        ax.set_title(f"C II\nX={x_arcsec:.2f}\", Y={y_arcsec:.2f}\"")

        # ----------------------------------------------------
        # Column 3: HDU 5 = Si IV 1403
        # ----------------------------------------------------
        ax = axes[row, 2]
        ax.plot(siiv_wave, siiv_spec, lw=1.0)
        ax.set_ylim(bottom=0)
        ax.set_xlim(1402.1, 1403.5)
        ax.set_xlabel("Wavelength (Å)")
        ax.set_ylabel("Intensity")
        ax.set_title(f"Si IV 1403\nX={x_arcsec:.2f}\", Y={y_arcsec:.2f}\"")

    plt.tight_layout()
    plt.savefig("images/9-plots.png", bbox_inches="tight")
    plt.show()
                                   
