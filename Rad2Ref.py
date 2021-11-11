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

def get_rad():
    
    f = get_src() + 'Files\\cropped_file_-20_650_490_490.hdr'  
    hdr = sp.envi.open(f)
    c1, c2 = 100, 110      #Selecting a small region to get the radiances
    spectralon = hdr.read_subregion((c1, c2), (c1, c2))   #x start x stop, y start, y stop
    spec_ave = np.average(spectralon.reshape(10*10, hdr.nbands), axis=0)
    
    return spec_ave


# <summary>
# Getting reflectance for the white reference 
# RefWhite.csv -> reflectances values for vendor 
# </summary>

def get_refl(hyp):
    
    
    wvl = hyp.bands.centers
    spectile = np.genfromtxt(get_src() + 'WhiteRefrence\\RefWhiteVNIR_color.csv', delimiter=',')
    
    file_name = 'RefWhite_VNIR_color_Inp.csv'
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

def get_lightSource(hyp):
    
    spec_ave = get_rad()
    specrefl = get_refl(hyp)
    lightsource = np.divide(spec_ave, specrefl)
    
    return lightsource

# <summary>
# In order to convert from radiance to reflectance, light source spds should be known
# Light source spd is obtained from reflectnace of White reference target and radiance
# This function will find the light source spd first and then obtain reflectance 
# </summary>

def get_refl_from_rad(hdr):
    
    #src = get_src()
    #f = src + 'ESR10_O_MOCKUPS_VNIR_1800_SN00841_14993us_2021-08-11T143013_raw_rad.hdr'  
    #hyp = get_MainRadianceFile()
    # Preparing metadata for the reflectance file
    fname = os.path.splitext(f)[0] + '_refl'
    metadata = hdr.metadata.copy()
    metadata['header file'] = fname+'.hdr'
    metadata['interleave'] = 'bil'
    metadata['header offset'] = 0
    metadata['data type'] = str(4)
    reflhyp = sp.envi.create_image(fname+'.hdr', metadata, force=True)
    
    lightsource = get_lightSource(hdr)
    # Getting the "illuminance" or multiplication factor by dividing 
    # captured radiance by the actual reflectance of the grey patch
    for i in range(hdr.nrows):
        if np.mod(i, 100) == 0:
            print(i, 'of', hdr.nrows)
        for j in range(hdr.ncols):
            pixel = hdr.read_pixel(i, j)
            # Dividing the image (radiance) by the lightsource to get 
            # reflectance
            reflhyp._memmap[i, :, j] = np.divide(pixel, lightsource)
    reflhyp._memmap.flush()




# def get_interp():

    
#     hyp = get_MainRadianceFile()
#     wvl = hyp.bands.centers
#     spectile = np.genfromtxt(get_src() + 'RefWhite.csv', delimiter=',')
    
   
#     wvl_white = spectile[:, 0].tolist()
#     spec_white = spectile[:, 2].tolist()
    
    
#     values = interp1d(wvl_white, spec_white)(wvl)
    
#     with open('GFG.csv', 'w', newline='') as f:
#         write = csv.writer(f)  # using csv.writer method from CSV package
#         write.writerows(map(lambda x: [x], values))
        
    
#     return values

# main function
if __name__ == '__main__':

   src = get_src()
   f = src + 'Files\\ESR10_O_MOCKUPS_VNIR_1800_SN00841_14993us_2021-08-11T143013_raw_rad.hdr'   
   hdr = sp.envi.open(f)
   get_refl_from_rad(hdr)
   collect()
   

    
   
    
   