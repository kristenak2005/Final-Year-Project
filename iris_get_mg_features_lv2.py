"""
Original Author: Vishal Upendran
Contact: uvishal1995@gmail.com

Updates Author: David Long
Contact: david.long@dcu.ie

This code finds Mg II h and k features, and is based on Leenarts et al (Paper II) 2013.

Reference IDL code: https://hesperia.gsfc.nasa.gov/ssw/iris/idl/uio/utils/iris_get_mg_features_lev2.pro

Changelog:
v1.0: Added all the codes for feature identification.
v1.1: Removed Halo beautification, and added flag for multiprocessing
v1.2: Tweaked to enable processing of fulldisk IRIS spectroheliograms (DL, 12-Nov-2024)
"""

import numpy as np 
from astropy.io import fits
#import iris_lmsalpy as iris
from iris_fitting import extract_irisL2data
from astropy.wcs import WCS
from glob import glob
try:
    import multiprocessing
    POOL_FLAG = True 
except:
    POOL_FLAG = False

import warnings
warnings.simplefilter('ignore', np.exceptions.RankWarning)


## Some helper functions
def Nearestind(val,x):
    return np.argmin(np.abs(val-x))
def dopp2wav(vel,ref):
    return vel*ref/299792.458+ref
def wav2dopp(wav,ref):
    return 299792.458*(wav-ref)/ref

def FitParabola_max(args):
    wave = args[0]
    spec = args[1]
    center = args[2]
    base_wvl = args[3]
    #No of points for parabola fit
    N = 3
    minval = np.max([0,center-N])
    maxval = np.min([len(wave)-1,center+N])
    #Parabola fit
    Spec_of_wave = np.poly1d(np.polyfit(wave[minval:maxval],spec[minval:maxval],2))
    #Inference
    new_wave = np.linspace(wave[minval],wave[maxval],50)
    spec_fit = Spec_of_wave(new_wave)
    mloc = new_wave[np.argmax(spec_fit)]
    return [wav2dopp(mloc,base_wvl),np.max(spec_fit)]

def FitParabola_min(args):
    wave = args[0]
    spec = args[1]
    center = args[2]
    base_wvl = args[3]
    #No of points for parabola fit
    N = 3
    #Parabola fit
    minval = np.max([0,center-N])
    maxval = np.min([len(wave)-1,center+N])
    #Parabola fit
    Spec_of_wave = np.poly1d(np.polyfit(wave[minval:maxval],spec[minval:maxval],2))
    #Inference
    new_wave = np.linspace(wave[minval],wave[maxval],50)
    spec_fit = Spec_of_wave(new_wave)
    mloc = new_wave[np.argmin(spec_fit)]
    return [wav2dopp(mloc,base_wvl),np.min(spec_fit)]
    
def Maxmin(d,wave,spec,base_wvl):
    maxv,minv =  np.where((d)<0)[0],np.where((d)>0)[0]
    if len(minv)==0:
        minv=[Nearestind(wave,dopp2wav(5,base_wvl))]
    else:
        minv=[minv[np.argmin(np.abs(wave[minv]-base_wvl))]]
    center = FitParabola_min([wave,spec,minv[0],base_wvl])
    
    if len(maxv)==0:
        blue = np.asarray([np.nan]*2)
        red = np.asarray([np.nan]*2)
    elif len(maxv)==1:
        if wave[maxv[0]]<dopp2wav(center[0],base_wvl):
            blue = FitParabola_max([wave,spec,maxv[0],base_wvl])
            red = np.asarray([np.nan]*2)
        else:
            blue = np.asarray([np.nan]*2)
            red = FitParabola_max([wave,spec,maxv[0],base_wvl])
    elif len(maxv)==2:
        blue = FitParabola_max([wave,spec,maxv[0],base_wvl])
        red = FitParabola_max([wave,spec,maxv[-1],base_wvl])
    else:
        cent = dopp2wav(center[0],base_wvl)
        b1 = np.where(wave[maxv]<cent)[0]
        r1 = np.where(wave[maxv]>cent)[0]
        if len(r1)!=0:
            red = FitParabola_max([wave,spec,maxv[r1[0]],base_wvl])
        else: 
            red = np.asarray([np.nan]*2)
            
        if len(b1)!=0:
            blue = FitParabola_max([wave,spec,maxv[b1[-1]],base_wvl])
        else:
            blue = np.asarray([np.nan]*2)
    return np.asarray([blue,center,red])

def Wrap_maxmin(input_args):
    return Maxmin(input_args[0],input_args[1],input_args[2],input_args[3])

def iris_get_mg_features_lev2(file,vrange=[-40,40],onlyk=False,onlyh=False,fulldisk=False):
    """ Returns the line center, red peak and blue peaks from Mg II h and k lines.

    Args:
        file (string): path to IRIS raster
        vrange (list, [vmin,vmax]): Velocity range to search for peaks. Defaults to [-40,40].
        onlyk (bool): if True, will only calculate properties for the Mg II k line. Defaults to False.
        onlyh (bool, optional): if True, will only calculate properties for the Mg II h line. Defaults to False.
    
    Returns:
        lc: 4-D array (line, feature, slit pos., raster pos.) @ line center
        rp: 4-D array (line, feature, slit pos., raster pos.) @ red peak
        bp: 4-D array (line, feature, slit pos., raster pos.) @ blue peak
    """
   
    Comb_array = []

    if fulldisk:
        # Open the IRIS FITS file
        sp = fits.open(file, memmap=False)

        ind = 0
        header = sp[ind].header
        new_mg = sp[ind].data.T
        # Get the wavelength
        nwave = new_mg.shape[2]
        A_to_nm = 1e-1  # convert wavelength to nm
        wavelength = (np.array(range(header['NAXIS3'])) * header['CDELT3'] + 
                      (header['CRVAL3']-((header['NAXIS3']/2)*header['CDELT3'])))*A_to_nm
    else:
        sp = fits.open(file, memmap=False)
        sub_header = sp[1].header

        if sub_header['CDELT3'] == 0:
            ind = np.where(extract_irisL2data.show_lines(file)=="Mg II k 2796")[0][0]
            extension = ind+1
            new_mg = sp[extension].data
            header = sp[extension].header
            A_to_nm = 10  # convert wavelength to nm
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

    if onlyk:
        BASE_LIST = [279.63509493]
    elif onlyh:
        BASE_LIST = [280.35297192]
    else:
        BASE_LIST = [279.63509493,280.35297192]

    for base_wvl in BASE_LIST:
        #Get the range over which spectral line is present.
        dvp_fit = dopp2wav(vrange[1],base_wvl)
        dvn_fit = dopp2wav(vrange[0],base_wvl)
        #Get nearest index from original spectrum to cutout the line
        indp_fit = Nearestind(dvp_fit,wavelength)
        indn_fit = Nearestind(dvn_fit,wavelength)
        # Save temporary variables of wavelength and spectrum
        wave = wavelength[indn_fit-1:indp_fit+1]
        spectrum=new_mg[:,:,indn_fit-1:indp_fit+1]
#        print(wave.shape,spectrum.shape,wavelength.shape)
        #Get the derivative
        peakvals = np.gradient(spectrum,axis=-1)
        sign = np.sign(peakvals)
        #Get the maxima and minima
        diff = sign[:,:,1:]-sign[:,:,:-1]
#        print(diff.shape,spectrum.shape)
        
        new_spec = spectrum.reshape([-1,spectrum.shape[-1]])
        new_diff = diff.reshape([-1,diff.shape[-1]])
        input_args=[[new_diff[loc],wave,new_spec[loc],base_wvl] for loc in np.arange(len(new_spec))]
#        print(spectrum.shape[0], spectrum.shape[1])
        if POOL_FLAG:
            pool=multiprocessing.Pool(processes=multiprocessing.cpu_count())
            M=np.asarray(pool.map(Wrap_maxmin,input_args)).reshape([spectrum.shape[0],spectrum.shape[1],3,2])
            pool.close()
        else:
            M=np.asarray([Wrap_maxmin(arg) for arg in input_args]).reshape([spectrum.shape[0],spectrum.shape[1],3,2])
        Comb_array.append(M)

    Comb_array = np.asarray(Comb_array)
    bp = Comb_array[:,:,:,0,:]
    lc = Comb_array[:,:,:,1,:]
    rp = Comb_array[:,:,:,2,:]
    return lc,rp,bp
