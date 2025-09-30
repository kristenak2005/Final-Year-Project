##!/usr/bin/env python3
# -*- coding: utf-8 -*-

## Routine originally written by David Long for processing IRIS data.
import time
import warnings
warnings.filterwarnings('ignore')

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
from scipy.optimize import curve_fit
from scipy.constants import speed_of_light
from astropy.coordinates import SkyCoord
from sunpy.coordinates import frames
from astropy import units as u
from sunpy.map import Map, make_fitswcs_header
import scipy.constants as const
from general_routines import closest
from scipy.integrate import cumulative_simpson
from astropy.wcs import WCS
from iris_fitting import extract_irisL2data
from astropy.io import fits

class fit_raster:

# Define the program info.
    def __init__(self, filename, window, fulldisk=False):
        self.name = "fit_iris_lines"
        self.version = "0.2"
        self.author = "David M. Long"
        self.email = "david.long@dcu.ie"
        self.filename = filename
        self.window = window
        self.fulldisk = fulldisk

# Function to open different file types
    def open_iris_file(self, open_window):

    # Open the IRIS FITS file
        sp = fits.open(self.filename, memmap=False)

    # Get the line list and match to the chosen line
        main_header = sp[0].header

        if self.fulldisk:
            ind = 0
            data = sp[ind].data.T
            header = sp[ind].header
        # Get the wavelength
            wavelength = (np.array(range(header['NAXIS3'])) * header['CDELT3'] + 
                          (header['CRVAL3']-((header['NAXIS3']/2)*header['CDELT3'])))
        else:
            sub_header = sp[1].header

            if sub_header['CDELT3'] == 0:
                ind = np.where(extract_irisL2data.show_lines(self.filename)==open_window)[0][0]
                extension = ind+1
                data = sp[extension].data
                header = sp[extension].header
        # Get the wavelength
                wavelength = (np.array(range(header['NAXIS1'])) * header['CDELT1'] + header['CRVAL1'])

            else:
                rast = extract_irisL2data.load(self.filename,window_info=[open_window],verbose=False)
                ind = np.where(extract_irisL2data.show_lines(self.filename)==open_window)[0][0]
                extension = ind+1
                header = extract_irisL2data.only_header(self.filename,extension=extension,verbose=False)
                data = rast.raster[open_window].data

        # Get the wavelength
                wcs = WCS(header)
                m_to_ang = 1e10  # convert wavelength to ang
                nwave = data.shape[2]
                wavelength = wcs.all_pix2world(np.arange(nwave), [0.], [0.], 0)[0] * m_to_ang

        return data, main_header, header, wavelength

# Function to get the fit properties for the line fitting
    def get_line_references(self, iris_window):

        # Set initial values and bounds based on the peak
        match iris_window:
            case "Si IV 1393":
                ref_wvl = 1393.76
                inst_wid = 0.0286
                vel = [10,60]
            case "Si IV 1394":
                ref_wvl = 1393.76
                inst_wid = 0.0286
                vel = [10,60]
            case "C II 1334":
                ref_wvl = 1334.53
                inst_wid = 0.0286
                vel = [10,60]
            case "C II 1335":
                ref_wvl = 1335.71
                inst_wid = 0.0286
                vel = [10,60]
            case "C II 1334 1336":
                ref_wvl = 1334.53
                inst_wid = 0.0286
                vel = [10,60]
            case "C II 1335 1336":
                ref_wvl = 1335.71
                inst_wid = 0.0286
                vel = [10,60]
            case "Si IV 1403":
                ref_wvl = 1402.77
                inst_wid = 0.0286
                vel = [10,60]
            case "Si IV 1394 1403":
                ref_wvl = 1402.77
                inst_wid = 0.0286
                vel = [10,60]

        return ref_wvl, inst_wid, vel

# Function to get the fit properties for the line fitting
    def get_line_fit_properties(self, iris_window, iris_prof):

        ref_wvl = None
        inst_wid = None
        vel = None
        
        ref_wvl, inst_wid, vel = self.get_line_references(iris_window)

        # Set initial values and bounds based on the peak
        match iris_window:
            case "Si IV 1393":
                wl_range = [1392.7, 1394.7]
                init_vals = [0., np.nanmax(iris_prof), ref_wvl, 0.03]
                limits = [[0., 0., ref_wvl-0.2, 0.005],[5., 2.*np.nanmax(iris_prof)+1., ref_wvl+0.2, 0.55]]
            case "Si IV 1394":
                wl_range = [1392.7, 1394.7]
                init_vals = [0., np.nanmax(iris_prof), ref_wvl, 0.03]
                limits = [[0., 0., ref_wvl-0.2, 0.005],[5., 2.*np.nanmax(iris_prof)+1., ref_wvl+0.2, 0.55]]
            case "C II 1334":
                wl_range = [1334.0, 1335.0]
                init_vals = [0., np.nanmax(iris_prof), ref_wvl, 0.03]
                limits = [[0., 0., ref_wvl-0.2, 0.005],[5., 2.*np.nanmax(iris_prof)+1., ref_wvl+0.2, 0.55]]
            case "C II 1335":
                wl_range = [1335.2, 1336.2]
                init_vals = [0., np.nanmax(iris_prof), ref_wvl, 0.03]
                limits = [[0., 0., ref_wvl-0.2, 0.005],[5., 2.*np.nanmax(iris_prof)+1., ref_wvl+0.2, 0.55]]
            case "C II 1334 1336":
                wl_range = [1334.0, 1335.0]
                init_vals = [0., np.nanmax(iris_prof), ref_wvl, 0.03]
                limits = [[0., 0., ref_wvl-0.2, 0.005],[5., 2.*np.nanmax(iris_prof)+1., ref_wvl+0.2, 0.55]]
            case "C II 1335 1336":
                wl_range = [1335.2, 1336.2]
                init_vals = [0., np.nanmax(iris_prof), ref_wvl, 0.03]
                limits = [[0., 0., ref_wvl-0.2, 0.005],[5., 2.*np.nanmax(iris_prof)+1., ref_wvl+0.2, 0.55]]
            case "Si IV 1403":
                wl_range = [1401.7, 1403.7]
                init_vals = [0., np.nanmax(iris_prof), ref_wvl, 0.03]
                limits = [[0., 0., ref_wvl-0.2, 0.005],[5., 2.*np.nanmax(iris_prof)+1., ref_wvl+0.2, 0.55]]
            case "Si IV 1394 1403":
                wl_range = [1401.7, 1403.7]
                init_vals = [0., np.nanmax(iris_prof), ref_wvl, 0.03]
                limits = [[0., 0., ref_wvl-0.2, 0.005],[5., 2.*np.nanmax(iris_prof)+1., ref_wvl+0.2, 0.55]]

        return wl_range, init_vals, limits

# Define a function to fit the Gaussian.
    def sum_gaussian(self, x,  *param_gauss):

        """ Calculates a sum of N Gaussians with parameters c, amp1, mu1, sigma1,
        ... ampN, muN, sigmaN, with c as offset constant in Y """

        params_gauss_ok = np.array(param_gauss)[1:]
        params_gauss_ok = params_gauss_ok.reshape(int(params_gauss_ok.size/3),3)
        g = param_gauss[0]
        count = 0
        for i in params_gauss_ok:
            g = g +  i[0] * np.exp(-0.5*((x-i[1])/i[2])**2)
        return g

# Function to fit a single (x, y) spectrum
    def fit_spectrum(self, tasks):

        wavelength, iris_prof = tasks

        wl_range, init_vals, limits = self.get_line_fit_properties(self.window, iris_prof)

        wl_0 = np.argmin(np.abs(wavelength - wl_range[0]))
        wl_1 = np.argmin(np.abs(wavelength - wl_range[1]))
        iris_prof = iris_prof[wl_0:wl_1+1].copy()
        wl_ref = wavelength[wl_0:wl_1+1].copy()
        
        try:
            best_vals, covar = curve_fit(self.sum_gaussian, wl_ref, iris_prof, p0=init_vals, bounds=limits, 
                                         sigma=None, absolute_sigma=False, maxfev=2000, 
                                         nan_policy='omit')
            errors = np.sqrt(np.diag(covar))
            if best_vals[1] < 1:
                best_vals = [np.nan,np.nan,np.nan,np.nan]
                errors = [np.nan,np.nan,np.nan,np.nan]

        except RuntimeError:
            best_vals = [np.nan,np.nan,np.nan,np.nan]
            errors = [np.nan,np.nan,np.nan,np.nan]
    
        results = np.concatenate((best_vals, errors), axis=None)

        return results

# Function to get blue-wing asymmetry of a single (x, y) spectrum
    def get_bluewing_asym(self, tasks):

        wavelength, iris_prof, fit_params = tasks

# Get the wavelength bounds
        cen, inst_wid, vel = self.get_line_references(self.window)

        wl_range, init_vals, limits = self.get_line_fit_properties(self.window, iris_prof)

        wl_0 = np.argmin(np.abs(wavelength - wl_range[0]))
        wl_1 = np.argmin(np.abs(wavelength - wl_range[1]))
        iris_profile = iris_prof[wl_0:wl_1+1].copy()
        wl_ref = wavelength[wl_0:wl_1+1].copy()

# Get the velocity bounds
        min_blue = (cen)/((vel[0])/(speed_of_light/1e3)+1)
        max_blue = (cen)/((vel[1])/(speed_of_light/1e3)+1)

        ind_obs_min = closest(wl_ref, min_blue)
        ind_obs_max = closest(wl_ref, max_blue)

        data_cumul_blue = cumulative_simpson(iris_profile, x=wl_ref, initial=0)
        blue_obs = data_cumul_blue[ind_obs_min]-data_cumul_blue[ind_obs_max]

# Now get the fitted line
        xline = np.arange(wl_ref[0], wl_ref[-1], 0.01)
        fit_line = fit_params[0] +  fit_params[1] * np.exp(-(1.0/2.0)*((xline-fit_params[2])/fit_params[3])**2)

        ind_fit_min = closest(xline, min_blue)
        ind_fit_max = closest(xline, max_blue)

        fit_cumul_blue = cumulative_simpson(fit_line, x=xline, initial=0)
        blue_fit = fit_cumul_blue[ind_fit_min]-fit_cumul_blue[ind_fit_max]

        alpha_b = (blue_obs - blue_fit)/((blue_obs + blue_fit)/2)

        return alpha_b

# Multiprocessing wrapper function for Gaussian fitting
    def fit_gen(self, data_array, wavelength, ncores):
    
        y_size = data_array.shape[0]
        x_size = data_array.shape[1]

        tasks = [(wavelength, data_array[y, x, :]) for y in range(y_size) for x in range(x_size)]

        # Create a delayed task for each (y, x) spectrum
        with Pool(processes=ncores) as pool:
            results = list(tqdm(pool.imap(self.fit_spectrum, tasks), total=len(tasks), desc="Fitting IRIS spectra"))

        # Reshape the results back into the (y, x, 8) shape (numpy style)
        res = np.array([result for result in results], dtype='object').reshape(y_size, x_size, 8)
        results_array = res.astype(np.float32)
    
        return results_array

# Multiprocessing wrapper function for blue-wing asymmetry calculation
    def fit_asym(self, data_array, wavelength, fit_array, ncores):
    
        y_size = data_array.shape[0]
        x_size = data_array.shape[1]

        tasks = [(wavelength, data_array[y, x, :], fit_array[y, x, :]) for y in range(y_size) for x in range(x_size)]

        # Create a delayed task for each (y, x) spectrum (numpy style)
        with Pool(processes=ncores) as pool:
            results = list(tqdm(pool.imap(self.get_bluewing_asym, tasks), total=len(tasks), desc="Calculating blue-wing asymmetries"))

        # Reshape the results back into the (y, x) shape
        res = np.array([result for result in results], dtype='object').reshape(y_size, x_size)
        results_array = res.astype(np.float32)
    
        return results_array

# Function to use IRIS file to make SunPy map of fitted variable
    def mk_iris_map(self, data, header, main_header):

        if self.fulldisk:
            data_array = data

            coord = SkyCoord(header['CRVAL1']*u.arcsec, header['CRVAL2']*u.arcsec, obstime = main_header['date_obs'], 
                             observer = 'earth', frame = frames.Helioprojective)

# Make IRIS maps
            map_header = make_fitswcs_header(data_array, coord, reference_pixel=[header['CRPIX1'], header['CRPIX2']]*u.pixel,
                                             scale=[np.abs(header['CDELT1']), header['CDELT2']]*u.arcsec/u.pixel, telescope='IRIS',
                                             instrument='Spectrograph', wavelength=header['CRVAL3']*u.angstrom)

        else:
            if header['CDELT3'] < 0:
                data_array = np.fliplr(data)
            else:
                data_array = data

            if header['CDELT3'] == 0:
                coord = SkyCoord(header['CRVAL3']*u.arcsec, header['CRVAL2']*u.arcsec, obstime = main_header['date_obs'],
                                 observer = 'earth', frame = frames.Helioprojective)

# Make IRIS maps
                map_header = make_fitswcs_header(data_array, coord, reference_pixel=[header['CRPIX3'], header['CRPIX2']]*u.pixel,
                                                 scale=[1e-9, header['CDELT2']]*u.arcsec/u.pixel, telescope='IRIS',
                                                 instrument='Spectrograph', wavelength=header['CRVAL1']*u.angstrom)

            else:
                coord = SkyCoord(header['CRVAL3']*u.arcsec, header['CRVAL2']*u.arcsec, obstime = main_header['date_obs'], 
                                 observer = 'earth', frame = frames.Helioprojective)

            # Make IRIS maps
                map_header = make_fitswcs_header(data_array, coord, reference_pixel=[header['CRPIX3'], header['CRPIX2']]*u.pixel,
                                                 instrument='Spectrograph', wavelength=header['CRVAL1']*u.angstrom, telescope='IRIS',
                                                 scale=[np.abs(header['CDELT3']), header['CDELT2']]*u.arcsec/u.pixel)

        iris_map = Map(data_array, map_header)

        return iris_map

# Function to get the non-thermal velocity of the line
    def iris_vnt(self, element, ion, wvl, obs_wid, inst_wid, ti=None, nt_fwhm=False, therm_v=False):
        """
        Compute the thermal and non-thermal velocity of a given spectral line in km/s.
    
        Parameters
        ----------
    
        element: string, the element that is being studied. 
    
        ion: string, the ionization state of the line. 
    
        wvl: float, the nominal rest wavelength of the line measured in Angstrom (Å)
    
        obs_wid: float, the observed FWHM of the line, measured in Angstrom (Å)
    
        inst_wid: float, the instrumental FWHM in Angrstrom (Å). For IRIS the instrumental FWHM are:
                        FUV short = 0.0286Å
                        FUV long = 0.0318Å

                        If an instrumental width is not supplied the value zero will be used. Default = None 
    
        ti: float, the peak ion temperature in the ionizations balance, measured in logT. If not supplied,
                    the value from ions.txt will be used. Default = None
                
        nt_fwhm: boolean, if this is set the routine will return the non-thermal FWHM in Angstrom (Å) 
                        along with the non-thermal velocity. Default = False
                    
        therm_v: boolean,  if set the routine will return the thermal velocity of the line in km/s. Default = False
    
        verbose: boolean, it True the routine prints some values of the fit. Default = False
    
        Returns
        -------
        v_nt: float, the non-thermal velocity of the line, in km/s
    
        fwhm_nt: float, if therm_v is True, the non-thermal FWHM of the lime is returned, in Angstrom(Å)
    
        v_t: float, if therm_v is True, the thermal velocity is returned, in km/s
    
    
        Example
        -------
    
        v_nt = iris_vnt('Si', 'iv', 1402.77, obs_fwhm ,inst_wid=0.0286, verbose=True)
    
    
        Version History:
        ----------------
        original version: 17/06/2020, Author Magnus M. Woods, Translated and adapted from Dr Harry Warren's IDL 
                                                                routine eis_width2velocity.pro

        Modified version: 09/04/2025, Author David M. Long, Adapted from original version written by Magnus M. 
                                                                Woods.
    
        """
    
        element = element.upper()
        ion = ion.upper()
        cc = const.c/1000 # c in km/s
    
        # Read in the ions.txt file 
        #This is a duplicate of Dr. Harry Warren's file of temperature and thermal velocity, eis_width2velocity.dat, in SSW. 
        ion_dat = np.loadtxt('ions.txt', dtype='U')
        elements = np.asarray(list(map(str.upper, ion_dat[1:, 0])))
        ions = ion_dat[1:,1]
        t_max = ion_dat[1:,2].astype(float)
        v_therm = ion_dat[1:,3].astype(float)
    
        elms,ins = np.array([i for i, n in enumerate(elements) if n == element]),np.array([i for i, n in enumerate(ions) if n == ion])
    
        match = []
        for i in enumerate(elms):
            for j in enumerate(ins):
                if i[1] == j[1]: match.append(i[1])
        count = len(match)
    
        if count > 0:
            if ti is None:
                ti_max = t_max[match[0]]
                v_t = v_therm[match[0]]
            else:
                # Load info in about various atomic masses
                masses_dat = np.loadtxt('element_masses.txt', dtype='U')
                elem_ms = np.asarray(list(map(str.upper, masses_dat[1:, 0])))
                mass_g = masses_dat[1:, 2].astype(float)
                # Get the mass of the ion we want to study
                mass = mass_g[np.where(elem == match[0])[0][0]]

                v_t = np.sqrt(2*(10.0**ti)*1.6022**-12/11604.0/mass_g)/1.0**5
                ti_max = ti
            
            # Thermal FWHM in Å
            therm_fwhm = np.sqrt(4*np.log(2))*wvl*v_t/cc
        
            if inst_wid is None:
                inst_wid = 0
                print('Warning: Calculating V_nt without instrumental width')
                print()
        
            # Non-thermal FWHM in Å
            fwhm_nt_2 = (obs_wid**2 - inst_wid**2 - therm_fwhm**2)

            # non-thermal velocity
            v_nt = np.full((obs_wid.shape[0], obs_wid.shape[1]), 0.)
            fwhm_nt = np.full((obs_wid.shape[0], obs_wid.shape[1]), 0.)
            for i in range(0, obs_wid.shape[0]):
                for j in range(0, obs_wid.shape[1]):
                    if fwhm_nt_2[i,j] > 0:
                        v_nt[i,j] = cc*(np.sqrt(fwhm_nt_2[i,j])/wvl)/np.sqrt(4*np.log(2))
                        fwhm_nt[i,j] = np.sqrt(fwhm_nt_2[i,j])
                    else:
                        v_nt[i,j] = 0.
                        fwhm_nt[i,j] = 0.
           
        return v_nt

# Main function to fit IRIS raster and produce maps of intensity, Doppler velocity, and nonthermal velocity
    def fit_iris_data(self, v_nontherm=False):

        start_time = time.time()
    
        # Define the number of cores for line fitting
        if os.uname().sysname == 'Darwin':
            ncpu = 7
        else:
            ncpu = len(os.sched_getaffinity(0))

        # Open the IRIS FITS file
        match self.window:
            case "Si IV 1394 1403":
                open_window='Si IV 1403'
            case "C II 1334 1336":
                open_window='C II 1336'
            case "C II 1335 1336":
                open_window='C II 1336'
            case _:
                open_window = self.window

        data, main_header, header, wavelength = self.open_iris_file(open_window) # type: ignore

        # Clean the data array & get its size
        data[data <= 0] = 0
        if header['CDELT3'] == 0:
            y_size = data.shape[1]
            x_size = data.shape[0]
        else:
            y_size = data.shape[0]
            x_size = data.shape[1]
        
        # Fit the IRIS data
        results_array = self.fit_gen(data, wavelength, ncpu)
        
        # Get some constants
        ref_wvl, inst_ref_wid, vel = self.get_line_references(self.window)

        # We're going to define the reference wavelength using the average wavelength of the fits. 
        # This is similar to the approach taken by eis_auto_fit in IDL.
        ref_wvl = np.nanmean(results_array[:,:,2])

        # Now get the line properties from the fit.
        # Get the intensity as a map
        if self.fulldisk:
            intensity = results_array[:,:,1].T
        else:
            if header['CDELT3'] == 0:
                intensity = results_array[:,:,1].T
            else:
                intensity = results_array[:,:,1]
        int_map = self.mk_iris_map(intensity, header, main_header)

        # Get the Doppler velocity as a map
        if self.fulldisk:
            line_pos = results_array[:,:,2].T
            ref_wave = np.full((x_size, y_size), ref_wvl)
        else:
            if header['CDELT3'] == 0:
                line_pos = results_array[:,:,2].T
            else:
                line_pos = results_array[:,:,2]
            ref_wave = np.full((y_size, x_size), ref_wvl)
        v_dopp = ((line_pos - ref_wave)/(ref_wave))*(speed_of_light/1e3)
        doppler_map = self.mk_iris_map(v_dopp, header, main_header)
        
        # Get the line width & nonthermal velocity
        if self.fulldisk:
            line_width = results_array[:,:,3].T
        else:
            if header['CDELT3'] == 0:
                line_width = results_array[:,:,3].T
            else:
                line_width = results_array[:,:,3]
        width_map = self.mk_iris_map(line_width, header, main_header)   

        if v_nontherm:
            fwhm = 2.355 * line_width
            if self.fulldisk:
                inst_wid = np.full((x_size, y_size),inst_ref_wid)
            else:
                inst_wid = np.full((y_size, x_size),inst_ref_wid)
            elem = open_window.split()[0]
            ion = open_window.split()[1].lower()
            v_nt = self.iris_vnt(elem, ion, ref_wvl, fwhm, inst_wid)
            vnt_map = self.mk_iris_map(v_nt, header, main_header)
        else:
            vnt_map = width_map

        # Get the red-blue asymmetry
        if self.fulldisk:
            fit_array = results_array[:,:,0:4]
        else:
            fit_array = results_array[:,:,0:4]

        bw_asym = self.fit_asym(data, wavelength, fit_array, ncpu)

        if self.fulldisk:
            asym = bw_asym.T
        else:
            if header['CDELT3'] == 0:
                asym = bw_asym.T
            else:
                asym = bw_asym
        asym_map = self.mk_iris_map(asym, header, main_header)
        
        print(f"Time taken to fit raster: {time.time() - start_time:.2f} seconds")

        return results_array, int_map, doppler_map, width_map, vnt_map, asym_map

if __name__ == "__main__":
    fit_iris = fit_raster()
    fit_iris.fit_iris_data()
