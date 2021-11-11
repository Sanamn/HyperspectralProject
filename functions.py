# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 19:22:48 2021

@author: Sanam
"""

import spectral as sp # http://www.spectralpython.net/
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import csv



# <summary>
# Cropping White Reference tile 
# Radiance values will be saved in 490 x 490 img/hdr file
# </summary>

def crop_WhiteRefrence(hdr):
    
    meta = hdr.metadata
    x_start, y_start =   -20,650            #Choosing the cordinates strating from white tile
    row_size, col_size = 490, 490
    crop_filename = "cropped_file_-20_650_490_490.hdr"    
    crop_size = (row_size, col_size, hdr.nbands)
    crop_hdr = sp.envi.create_image(crop_filename,shape=crop_size, metadata=meta, force=True)

    for i in range(row_size):
        for j in range(col_size):
          pixel = hdr.read_pixel(i+x_start, j+y_start)
          # Writing the pixel values to disk, through its memmap interface
          crop_hdr._memmap[i, :, j] = pixel
    crop_hdr._memmap.flush()


# <Summary>
# This function is used to interpolate the two signals to have some wavelength
# </Summary

def get_interp(target_wvl,source_spectile,filename):

    wvl_white = source_spectile[:, 0].tolist()
    spec_white = source_spectile[:, 2].tolist()
    
    values = interp1d(wvl_white, spec_white,fill_value="extrapolate")(target_wvl)
    
    with open(filename, 'w', newline='') as f:
        write = csv.writer(f)  # using csv.writer method from CSV package
        #write.writerows(map(lambda x: [x], zip(target_wvl,values)))
        write.writerows(zip(target_wvl,values))
        
        
    plt.plot(values,target_wvl,'r', label='interp/extrap')
    #plt.plot(spec_white,wvl_white, 'b--', label='data')
    plt.legend()
    plt.show()
        
    return 
