##!/usr/bin/env python3
# -*- coding: utf-8 -*-

## Routine originally written by David Long for getting Mg II profile properties using quartiles

import warnings
warnings.filterwarnings('ignore')

import numpy as np
from tqdm import tqdm
from astropy.io import fits
from iris_fitting import extract_irisL2data
from astropy.wcs import WCS


def estimate_mgii_quartiles(wavelength, profile):
    """
    Estimate the quartiles of the Mg II k line profile.
    
    Parameters:
    profile : array-like
        The intensity profile of the Mg II k line.
    wavelength : array-like
        The corresponding wavelengths for the profile.
    
    Returns:
    tuple : (q1_wvl, q2_wvl, q3_wvl)
        The wavelengths corresponding to the first, second (median), and third quartiles.
    """

    if np.count_nonzero(profile)>0:
        cs = np.nancumsum(profile)
        cdf = cs/cs[-1]
        integ_int = cs[-1]

        q1_wvl = wavelength[np.argmin(np.abs(cdf - 0.25))]
        q2_wvl = wavelength[np.argmin(np.abs(cdf - 0.5))]
        q3_wvl = wavelength[np.argmin(np.abs(cdf - 0.75))]

        xrng = np.linspace(wavelength[0],wavelength[-1],2000)
        yrng = np.interp(xrng,wavelength,cdf)
        q1_wvl_interp = xrng[np.argmin(np.abs(yrng - 0.25))]
        q2_wvl_interp = xrng[np.argmin(np.abs(yrng - 0.5))]
        q3_wvl_interp = xrng[np.argmin(np.abs(yrng - 0.75))]

    else:
        integ_int = np.nan
        q1_wvl = np.nan
        q2_wvl = np.nan
        q3_wvl = np.nan
        q1_wvl_interp = np.nan
        q2_wvl_interp = np.nan
        q3_wvl_interp = np.nan
    
    return q1_wvl, q2_wvl, q3_wvl, q1_wvl_interp, q2_wvl_interp, q3_wvl_interp, integ_int


# Main routine to get the Mg II quartiles
def get_mgii_quartiles(file, line_wvl, est_wvl=None, fulldisk=False):

    A_to_nm = 10  # convert wavelength to nm

    if fulldisk:
        # Open the IRIS FITS file
        sp = fits.open(file, memmap=False)

        ind = 0
        header = sp[ind].header
        new_mg = sp[ind].data.T
        # Get the wavelength
        nwave = new_mg.shape[2]
        wavelength = (np.array(range(header['NAXIS3'])) * header['CDELT3'] + 
                      (header['CRVAL3']-((header['NAXIS3']/2)*header['CDELT3'])))/A_to_nm
    else:
        sp = fits.open(file, memmap=False)
        sub_header = sp[1].header

        if sub_header['CDELT3'] == 0:
            ind = np.where(extract_irisL2data.show_lines(file)=="Mg II k 2796")[0][0]
            extension = ind+1
            new_mg = sp[extension].data
            header = sp[extension].header
            nwave = new_mg.shape[2]
        # Get the wavelength
            wavelength = (np.array(range(header['NAXIS1'])) * header['CDELT1'] + header['CRVAL1'])/A_to_nm

        else:
            rast = extract_irisL2data.load(file,window_info=["Mg II k 2796"],verbose=False)
            mg2_loc = np.where(extract_irisL2data.show_lines(file)=='Mg II k 2796')[0][0]
            extension = mg2_loc+1
            head = extract_irisL2data.only_header(file,extension=extension)
            wcs = WCS(head)
            new_mg = rast.raster['Mg II k 2796'].data
            m_to_nm = 1e9  # convert wavelength to nm
            nwave = new_mg.shape[2]
            wavelength = wcs.all_pix2world(np.arange(nwave), [0.], [0.], 0)[0] * m_to_nm

    match line_wvl:
        case 'Mg II k':
            if est_wvl is None:
                est_wvl = 279.635
        case 'Mg II h':
            if est_wvl is None:
                est_wvl = 280.353
    
    k_min = est_wvl - 0.2
    k_max = est_wvl + 0.2

    wvl_crop = wavelength[(wavelength >= k_min) & (wavelength <= k_max)]
    datacube_crop = new_mg[:,:,(wavelength >= k_min) & (wavelength <= k_max)]

    y_size = datacube_crop.shape[0]
    x_size = datacube_crop.shape[1]

    profile = [(datacube_crop[y, x, :]) for y in range(y_size) for x in range(x_size)]

    results = []
    for ind in tqdm(np.arange(0, len(profile))):
        res_array = estimate_mgii_quartiles(wvl_crop, profile[ind])
        results.append(res_array)

    # Reshape the results back into the (y, x, 7) shape (numpy style)
    res = np.array([result for result in results], dtype='object').reshape(y_size, x_size, 7)
    quartiles = res.astype(np.float32)

    return quartiles*A_to_nm