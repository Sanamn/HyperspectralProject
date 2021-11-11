# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 09:50:21 2021

@author: Sanam
"""

import spectral as sp # http://www.spectralpython.net/
from gc import collect
from PIL import Image
import numpy as np
import os
import spectral as sp # http://www.spectralpython.net/
from scipy.interpolate import interp1d
import csv
import matplotlib.pyplot as plt
from spectral import imshow, view_cube
import os


# <summary>
# Direct to the source directory
# </summary>

def get_src():
    return '/home/clr1/Sanam/hyperspectral analysis project/Files/'


def plotWhiteReference():
    spectile = np.genfromtxt(get_src() + 'WhiteRefrence\\RefWhite_VNIR_color_Inp.csv', delimiter=',')
    wvl = spectile[:, 0].tolist()
    values = spectile[:, 1].tolist()
    plt.plot(wvl,values)
    #plt.plot(spec_white,wvl_white, 'b--', label='data')
    #plt.legend()
    plt.show()
    return ''



def plotReflectance(filename):
     
    hdr = sp.envi.open(filename)
    wvl = hdr.bands.centers
    w, h = 3, 3
    
    #cordinates
    
    coordinates = [
    (288,1140),    
    (296,1440),  
    (280,1788),
    (256,2069)
    ]
    
   
    labels = ["2.5%", "5% ", "7.5%","10%"]
    #labels = ["1-coat Laropal A64", "2-coat Laropal A64 ", "1-coat  Gum-dammer","2-coat Gum dammer"]
    
    count = 0
    for coordinate in coordinates:
        
        (x, y) = coordinate
        roi = hdr[y:y+h, x:x+w, :]
        intensity = []
        for b in range(roi.shape[2]):
            intensity.append(np.mean(roi[:, :, b]))
        if(count==0):
            plt.plot(wvl, intensity,color='#040351', label=labels[count])
        if(count==1):
            plt.plot(wvl, intensity,color='#040351',linestyle='dashed', label=labels[count])
        if(count==2):
            plt.plot(wvl, intensity,color='#040351', linestyle='dotted',label=labels[count])
        if(count==3):
            plt.plot(wvl, intensity,color='#040351', linestyle='dashdot',label=labels[count])
            
        
        count=count+1
    
    plt.legend(loc='upper left')
    plt.title('Chromium Green')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Reflectance')
    plt.show()     
    
    
    return




if __name__ == '__main__':

   cwd = os.getcwd()
   print(cwd)
   MainfileName =  '/Files/VNIR-varnishC1/ESR10_mockup1_001_VNIR_1800_SN00841_14998us_2021-10-19T115801_raw_rad_refl.hdr'
   #src = get_src()
   f = cwd + MainfileName   
   plotReflectance(f)
   ##plotWhiteReference()
   
   
   collect()
   
   
   


# Notes:
# coordinates = [
#    (376,508),   #1-coat  #Laropal A64 
#    (1527,500),  #2-coat  #Laropal A64 
#    (300,752),   #1-coat  Gum-dammer
#    (1668,756)   #2-coat Gum dammer
#    ]
    
#    labels = ["1-coat Laropal A64", "2-coat Laropal A64 ", "1-coat  Gum-dammer","2-coat Gum dammer"]
#    #000118, #010103,#0f1015,#010206

# 2 -  labels = ["2.5%", "5% ", "7.5%","10%"]

#part 2- missing


#chromium Green 
#part 1
  # coordinates = [
  #   (300,2708),    
  #   (1760,2700),  
  #   (272,3112),
  #   (1716,3108)
  #   ]
  
  
  #yellow part 1
   # coordinates = [
   #  (264,5000),    
   #  (1680,5036),  
   #  (272,5448),
   #  (1700,5452)
   #  ]
  
 #yellow part 2 
   # coordinates = [
   #  (260,5688),    
   #  (256,6040),  
   #  (240,6284),
   #  (216,6644)
   #  ]
    
   
   
    #red part1
    
    # coordinates = [
    # (228,7372),    
    # (1680,7384),  
    # (220,7688),
    # (1688,7732)
    # ]
    
    #red part2
    #  coordinates = [
    # (232,7988),    
    # (204,8304),  
    # (216,8588),
    # (200,8832)
    # ]
    
    
    
    