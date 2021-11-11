# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 20:13:53 2021

@author: Sanam
"""

import spectral as sp


hdr = sp.envi.open("F:/COSI/Semester 3/Advance Project Work - (IMT4894)/For Sanam/ESR10_O_MOCKUPS_VNIR_1800_SN00841_14993us_2021-08-11T143013_raw_rad.hdr")
wvl = hdr.bands.centers
rows, cols, bands = hdr.nrows, hdr.ncols, hdr.nbands
meta = hdr.metadata

step = 5
for x in range(0, rows, step):
   rows = hdr.read_subregion((x, x+step), (0, cols))