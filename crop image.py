# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 23:42:41 2021

@author: Sanam
"""

import spectral as sp
import numpy as np
import os


cwd = os.getcwd()

MainfileName =  '/Files/VNIR-varnishC1/ESR10_mockup1_001_VNIR_1800_SN00841_14998us_2021-10-19T115801_raw_rad_refl.hdr'
#src = get_src()
f = cwd + MainfileName   
hdr = sp.envi.open(f)
meta = hdr.metadata

#x_start, y_start = 150, 100
#row_size, col_size = 400, 300

# crop_filename = "output\\cropped_file_13.hdr"
# #crop_size = (row_size, col_size, hdr.nbands)
# #crop_hdr = sp.envi.create_image(crop_filename, metadata=meta, force=True)

# # for i in range(row_size):
# #    for j in range(col_size):
# #       pixel = hdr.read_pixel(i+x_start, j+y_start)
# #       # Writing the pixel values to disk, through its memmap interface
# #       crop_hdr._memmap[i, :, j] = pixel.flatten()
# # crop_hdr._memmap.flush()

# crop_hdr = sp.envi.create_image(crop_filename, metadata=meta, force=True)
# x1, x2 = 861, 961
# y1,y2  = 172, 272
 
# spectralon = hdr.read_subregion((x1, x2), (y1, y2))
# spec_ave = np.average(spectralon.reshape(10*10, hdr.nbands), axis=0)
# crop_hdr._memmap[(x1), :, (x2)] = spec_ave.flatten()   #x



x_start, y_start =   996,680            #400, 300
row_size, col_size = 150, 150
crop_filename = "F.hdr"
crop_size = (row_size, col_size, hdr.nbands)
crop_hdr = sp.envi.create_image(crop_filename,shape=crop_size, metadata=meta, force=True)

for i in range(row_size):
    
    
    for j in range(col_size):
      print(i)
      print(j)
      pixel = hdr.read_pixel(x_start+i, y_start+j)
      print(x_start+i) 
      print(y_start+j)
       
      print("")
      # Writing the pixel values to disk, through its memmap interface
      crop_hdr._memmap[i, :, j] = pixel
crop_hdr._memmap.flush()













