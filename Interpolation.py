# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 13:39:35 2021

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
import matplotlib.pyplot as plt


def get_src():
    return 'F:\\COSI\Semester 3\\Advance Project Work - (IMT4894)\\Files\\New Data\\'

def get_interp(target_wvl,source_spectile,name):

    wvl_white = source_spectile[:, 0].tolist()
    spec_white = source_spectile[:, 2].tolist()
    
    values = interp1d(wvl_white, spec_white,fill_value="extrapolate")(target_wvl)
    
    # with open(name, 'w', newline='') as f:
    #     write = csv.writer(f)  # using csv.writer method from CSV package
    #     #write.writerows(map(lambda x: [x], zip(target_wvl,values)))
    #     write.writerows(zip(target_wvl,values))
        
        
    plt.plot(values,target_wvl,'r', label='interp/extrap')
    #plt.plot(spec_white,wvl_white, 'b--', label='data')
    plt.legend()
    plt.show()
        
    return True


# main function
if __name__ == '__main__':
    
    src = get_src()  
    
    #Interpolate SWIR data
    f = src + 'ESR10_mockup1_001_SWIR_384_SN3189_7401us_2021-10-19T115801_raw_rad.hdr'   
    hdr = sp.envi.open(f)
    wvl = hdr.bands.centers
    spectile = np.genfromtxt(get_src() + 'SWIR.csv', delimiter=',')
    get_interp(wvl,spectile)
    
    #Interpolate VNIR data
    
    
    
    
    collect()
   