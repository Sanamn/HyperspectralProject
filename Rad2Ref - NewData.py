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


# <summary>
# Direct to the source directory
# </summary>

def get_src():
    return 'F:\\COSI\Semester 3\\Advance Project Work - (IMT4894)\\Files\\New Data\\'


# <summary>
# The main radinace file call 
# <summary>

def get_MainRadianceFileVNIR():
    
    src = get_src()
    f = src + 'ESR10_mockup1_001_VNIR_1800_SN00841_14998us_2021-10-19T115801_raw_rad.hdr'   
    hdr = sp.envi.open(f)
    return hdr


def get_MainRadianceFileSWIR():
    
    src = get_src()
    f = src + 'ESR10_mockup1_001_SWIR_384_SN3189_7401us_2021-10-19T115801_raw_rad.hdr'   
    hdr = sp.envi.open(f)
    return hdr


# <summary>
# Cropping White Reference tile 
# Radiance values will be saved in 490 x 490 img/hdr file
# </summary>

def crop_WhiteRefrenceVNIR():
    
    src = get_src()
    f = src + 'ESR10_spectralon_001_VNIR_1800_SN00841_14998us_2021-10-19T114944_raw_rad.hdr'   
    hdr = sp.envi.open(f)
    
    meta = hdr.metadata
    x_start, y_start =   211,738            #Choosing the cordinates strating from white tile
    row_size, col_size = 100, 100
    crop_filename = "cropped_file_VNIR_211_738 _100_100.hdr"    
    crop_size = (row_size, col_size, hdr.nbands)
    crop_hdr = sp.envi.create_image(crop_filename,shape=crop_size, metadata=meta, force=True)

    for i in range(row_size):
        for j in range(col_size):
          pixel = hdr.read_pixel(i+x_start, j+y_start)
          # Writing the pixel values to disk, through its memmap interface
          crop_hdr._memmap[i, :, j] = pixel
    crop_hdr._memmap.flush()


# <summary>
# Cropping White Reference tile 
# Radiance values will be saved in 490 x 490 img/hdr file
# </summary>

def crop_WhiteRefrenceSWIR():
    
    src = get_src()
    f = src + 'ESR10_spectralon_001_SWIR_384_SN3189_7401us_2021-10-19T114944_raw_rad.hdr'   
    hdr = sp.envi.open(f)
    
    meta = hdr.metadata
    x_start, y_start =   39,172            #Choosing the cordinates strating from white tile
    row_size, col_size = 20, 20
    crop_filename = "cropped_file_SWIR_39_172_20_20.hdr"    
    crop_size = (row_size, col_size, hdr.nbands)
    crop_hdr = sp.envi.create_image(crop_filename,shape=crop_size, metadata=meta, force=True)

    for i in range(row_size):
        for j in range(col_size):
          pixel = hdr.read_pixel(i+x_start, j+y_start)
          # Writing the pixel values to disk, through its memmap interface
          crop_hdr._memmap[i, :, j] = pixel
    crop_hdr._memmap.flush()



# <summary>
# Getting radiance 10 x 10 patch from the white refrence
# Return average radiance 
#</summary>

def get_rad_VNIR():
    
    f = get_src() + 'cropped_file_VNIR_211_738 _100_100.hdr'  
    hdr = sp.envi.open(f)
    c1, c2 = 50, 60      #Selecting a small region to get the radiances
    spectralon = hdr.read_subregion((c1, c2), (c1, c2))   #x start x stop, y start, y stop
    spec_ave = np.average(spectralon.reshape(10*10, hdr.nbands), axis=0)
    
    return spec_ave


def get_rad_SWIR():
    
    f = get_src() + 'cropped_file_SWIR_39_172_20_20.hdr'  
    hdr = sp.envi.open(f)
    c1, c2 = 5, 20      #Selecting a small region to get the radiances
    spectralon = hdr.read_subregion((c1, c2), (c1, c2))   #x start x stop, y start, y stop
    spec_ave = np.average(spectralon.reshape(10*10, hdr.nbands), axis=0)
    
    return spec_ave


# <summary>
# Getting reflectance for the white reference 
# RefWhite.csv -> reflectances values for vendor 
# </summary>

def get_refl_VNIR():

    hyp = get_MainRadianceFileVNIR()
    wvl = hyp.bands.centers
    spectile = np.genfromtxt(get_src() + 'VNIR_used.csv', delimiter=',')
    
    t = wvl
    specrefl = []
    for i in t:
        idx = spectile[:, 0].tolist().index(i)
        specrefl.append(spectile[idx, 1])
    specrefl = np.asarray(specrefl) 
    if np.mod(i, 4) == 0:
        print(i, spectile[idx, 0])
    ##plt.plot_spectra(wvl, np.asarray(specrefl).reshape(1, b), 1, ylimits=[0, 1])
    
    return specrefl

# <summary>
# obtaining the light source illuminance by dividing 
# captured radiance by the actual reflectance of the grey patch
# </summary>

def get_lightSource_VNIR():
    
    spec_ave = get_rad_VNIR()
    specrefl = get_refl_VNIR()
    lightsource = np.divide(spec_ave, specrefl)
    
    return lightsource

# <summary>
# In order to convert from radiance to reflectance, light source spds should be known
# Light source spd is obtained from reflectnace of White reference target and radiance
# This function will find the light source spd first and then obtain reflectance 
# </summary>

def get_refl_from_rad_VIR():
    
    src = get_src()
    f = src + 'ESR10_mockup1_001_VNIR_1800_SN00841_14998us_2021-10-19T115801_raw_rad.hdr'   
    hyp = sp.envi.open(f)
    # Preparing metadata for the reflectance file
    fname = os.path.splitext(f)[0] + '_refl'
    metadata = hyp.metadata.copy()
    metadata['header file'] = fname+'.hdr'
    metadata['interleave'] = 'bil'
    metadata['header offset'] = 0
    metadata['data type'] = str(4)
    reflhyp = sp.envi.create_image(fname+'.hdr', metadata, force=True)
    
    lightsource = get_lightSource_VNIR()
    # Getting the "illuminance" or multiplication factor by dividing 
    # captured radiance by the actual reflectance of the grey patch
    for i in range(hyp.nrows):
        if np.mod(i, 100) == 0:
            print(i, 'of', hyp.nrows)
        for j in range(hyp.ncols):
            pixel = hyp.read_pixel(i, j)
            # Dividing the image (radiance) by the lightsource to get 
            # reflectance
            reflhyp._memmap[i, :, j] = np.divide(pixel, lightsource)
    reflhyp._memmap.flush()



# <Summary>
# This function is used to interpolate the two signals to have some wavelength
# </Summary
def get_interp():

    
    hyp = get_MainRadianceFileSWIR()
    wvl = hyp.bands.centers
    spectile = np.genfromtxt(get_src() + 'SWIR.csv', delimiter=',')
    
   
    wvl_white = spectile[:, 0].tolist()
    spec_white = spectile[:, 2].tolist()
    
    
    values = interp1d(wvl_white, spec_white)(wvl)
    
    with open('SWIR_GFG.csv', 'w', newline='') as f:
        write = csv.writer(f)  # using csv.writer method from CSV package
        write.writerows(map(lambda x: [x], values))
        
    
    return values

# main function
if __name__ == '__main__':
   #get_interp()
   #crop_WhiteRefrenceVNIR()
   get_refl_from_rad_VIR()
   collect()
   

    
   
    
   