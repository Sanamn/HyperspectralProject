# -*- coding: utf-8 -*-
"""
Created on Tues Oct 12 11:41:11 2021
@Code script by: Hilda
@author: Sanam
"""

from gc import collect
from PIL import Image
import numpy as np
import os
import spectral as sp # http://www.spectralpython.net/
#import get_plot as plt
from scipy.interpolate import interp1d
import csv
import functions as func
import matplotlib.pyplot as plt



# <summary>
# Direct to the source directory
# </summary>

def get_src():
    return 'F:\\COSI\Semester 3\\Advance Project Work - (IMT4894)\\'



# <summary>
# Getting radiance 10 x 10 patch from the white refrence
# Return average radiance 
#</summary>

def get_rad(ReflWhiteRef):
    
    f = get_src() + ReflWhiteRef 
   
    hdr = sp.envi.open(f)
    c1, c2 = 50, 60      #Selecting a small region to get the radiances
    spectralon = hdr.read_subregion((c1, c2), (c1, c2))   #x start x stop, y start, y stop
    spec_ave = np.average(spectralon.reshape(10*10, hdr.nbands), axis=0)
    
    return spec_ave


# <summary>
# Getting reflectance for the white reference 
# RefWhite.csv -> reflectances values for vendor 
# </summary>

def get_refl(hyp):
    
    
    wvl = hyp.bands.centers
    spectile = np.genfromtxt(get_src() + 'WhiteRefrence\\RefVNIR_varnish.csv', delimiter=',')
    
    file_name = 'RefVNIR_varnish_Inp.csv'
    interpolated = func.get_interp(wvl,spectile,file_name)
    specrefl = []
    
    
    spectile = np.genfromtxt(file_name, delimiter=',')
    t = wvl
    
    for i in t:
        idx = spectile[:, 0].tolist().index(i)
        specrefl.append(spectile[idx, 1])
    specrefl = np.asarray(specrefl) 
    if np.mod(i, 4) == 0:
        print(i, spectile[idx, 0])
            
        
    return specrefl
    
    ##plt.plot_spectra(wvl, np.asarray(specrefl).reshape(1, b), 1, ylimits=[0, 1])
    

# <summary>
# obtaining the light source illuminance by dividing 
# captured radiance by the actual reflectance of the grey patch
# </summary>

def get_lightSource(hyp,ReflWhiteRef):
    
    spec_ave = get_rad(ReflWhiteRef)
    specrefl = get_refl(hyp)
    lightsource = np.divide(spec_ave, specrefl)
    
    return lightsource

# <summary>
# In order to convert from radiance to reflectance, light source spds should be known
# Light source spd is obtained from reflectnace of White reference target and radiance
# This function will find the light source spd first and then obtain reflectance 
# </summary>

def get_refl_from_rad(hdr,ReflWhiteRef):
    
   
    fname = os.path.splitext(f)[0] + '_refl'
    metadata = hdr.metadata.copy()
    metadata['header file'] = fname+'.hdr'
    metadata['interleave'] = 'bil'
    metadata['header offset'] = 0
    metadata['data type'] = str(4)
    reflhyp = sp.envi.create_image(fname+'.hdr', metadata, force=True)
    
    lightsource = get_lightSource(hdr,ReflWhiteRef)
    # Getting the "illuminance" or multiplication factor by dividing 
    # captured radiance by the actual reflectance of the grey patch
    for i in range(hdr.nrows):
        if np.mod(i, 100) == 0:
            print(i, 'of', hdr.nrows)
        for j in range(hdr.ncols):
            pixel = hdr.read_pixel(i, j)
            # Dividing the image (radiance) by the lightsource to get 
            #reflectance
            reflhyp._memmap[i, :, j] = np.divide(pixel, lightsource)
    reflhyp._memmap.flush()



# main function
if __name__ == '__main__':

   
   MainfileName =  'Files\\New Data\\ESR10_mockup1_001_VNIR_1800_SN00841_14998us_2021-10-19T115801_raw_rad.hdr'
   ReflWhiteRef = 'Files\\New Data\\cropped_file_VNIR_211_738 _100_100.hdr'
   src = get_src()
   f = src + MainfileName   
   hdr = sp.envi.open(f)
   get_refl_from_rad(hdr,ReflWhiteRef)
   
   #get_rad(ReflWhiteRef)
   
   collect()
   

    
   
    
   