import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.time import Time

#File path for SJI at 1400 Angstrom
#fitsfile = "/mnt/scratch/data/spruksk2/IRIS/20160520_131758/iris_l2_20160520_131758_3620110603_SJI_1400_t000.fits"
#File path for SJI at 2832 showing the slit
fitsfile = "/mnt/scratch/data/spruksk2/IRIS/20160520_131758/iris_l2_20160520_131758_3620110603_SJI_2832_t000.fits"
time_plot = "2016-05-20T14:00:00"
cadence = 16.2
with fits.open(fitsfile) as hdul:
    data = hdul[0].data.astype(float)
    header = hdul[0].header

print("Data shape:", data.shape)
print("DATE_OBS =", header.get("DATE_OBS"))
target_time = Time(time_plot, format="isot", scale="utc")
start_time = Time(header["DATE_OBS"], format="isot", scale="utc")

idx = int(np.round((target_time - start_time).sec / cadence))
idx = np.clip(idx, 0, data.shape[0] - 1)

print("Chosen frame index:", idx)

frame = data[idx]

nx = header["NAXIS1"]
ny = header["NAXIS2"]

crpix1 = header["CRPIX1"]
crpix2 = header["CRPIX2"]

crval1 = header["CRVAL1"]
crval2 = header["CRVAL2"]

cdelt1 = header["CDELT1"]
cdelt2 = header["CDELT2"]

x = (np.arange(nx) + 1 - crpix1) * cdelt1 + crval1
y = (np.arange(ny) + 1 - crpix2) * cdelt2 + crval2

extent = [x.min(), x.max(), y.min(), y.max()]

vmin, vmax = np.nanpercentile(frame, [1, 99.5])

plt.figure(figsize=(8, 8))
im = plt.imshow(
    frame,
    origin="lower",
    cmap="afmhot",
    vmin=vmin,
    vmax=vmax,
    extent=extent,
    aspect="equal"
)

plt.colorbar(im, label="Intensity")
#plt.title(f"IRIS SJI 1400\nFrame {idx}, approx. {time_plot}")
plt.title(f"IRIS SJI 2832\nFrame {idx}, approx. {time_plot}")
plt.xlabel("Solar X (arcsec)")
plt.ylabel("Solar Y (arcsec)")
plt.tight_layout()
#plt.savefig("SJI1400_frame.png", dpi=300, bbox_inches="tight")
plt.savefig("SJI2832_frame.png", dpi=300, bbox_inches="tight")
plt.show()
