#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 18:14:27 2021

@author: clr1
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




cwd = os.getcwd()
print(cwd)
MainfileName =  '/Files/VNIR-varnishC1/ESR10_mockup1_001_VNIR_1800_SN00841_14998us_2021-10-19T115801_raw_rad_refl.hdr'
   #src = get_src()
f = cwd + MainfileName   
hdr = sp.envi.open(f)
wvl = hdr.bands.centers
w, h = 3, 3
    
#cordinates

coordinates = [

(272,512),
(1688,540),
(272,808),
(1676,836),
(288,1140),
(296,1440),
(280,1788),
(256,2068),
(300,2708),
(1760,2700),
(272,3112),
(1716,3108),
(264,3428),
(252,3748),
(252,4012),
(236,4352),
(264,5000),
(1680,5036),
(272,5448),
(1700,5452),
(260,5688),
(256,6040),
(240,6284),
(216,6644),
(228,7372),
(1680,7384),
(220,7688),
(1668,7732),
(232,7988),
(204,8304),
(216,8588),
(200,8832),
]

count = 0
for coordinate in coordinates:
    
    print(count)
    (x, y) = coordinate
    roi = hdr[y:y+h, x:x+w, :]
    intensity = []
    for b in range(roi.shape[2]):
        intensity.append(np.mean(roi[:, :, b]))
    print(np.mean(intensity))
    
    count=count+1
    

