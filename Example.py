# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 11:56:32 2021

@author: Sanam
"""

from gc import collect
from PIL import Image

import numpy as np
import os
import spectral as sp


import os

#import get_plot as plt
#import spectral_io as sio


def get_src():
    return 'F:\\COSI\Semester 3\\Advance Project Work - (IMT4894)\\Test set\\'


def process():
    '''
    Getting reflectance from corrected radiance image (xxx_corr.hyspex/hdr)
    '''    
    src = get_src()
    f = src + 'ESR10_O_MOCKUPS_VNIR_1800_SN00841_14993us_2021-08-11T143013_raw_rad.hdr'   
    spectile = np.genfromtxt(get_src() + 'RefWhite.csv', delimiter=',')   
    
    hyp = sp.envi.open(f)
    wvl = hyp.bands.centers       #come from the HDR file of the main image
    r, c, b = hyp.nrows, hyp.ncols, hyp.nbands
    
    # Finding the actual reflectance of the grey patch in the relevant
    # wavelength (from the NEO file)
    #t = map(int, wvl)
    
    t = wvl
    specrefl = []
    for i in t:
        idx = spectile[:, 0].tolist().index(i)
        specrefl.append(spectile[idx, 1])
    specrefl = np.asarray(specrefl) 
    if np.mod(i, 4) == 0:
        print(i, spectile[idx, 0])
    ##plt.plot_spectra(wvl, np.asarray(specrefl).reshape(1, b), 1, ylimits=[0, 1])
        
    # Getting the captured radiance of the grey patch
    c1, c2 = 100, 110
    
    spectralon = hyp.read_subregion((c1, c2), (c1, c2))   #x start x stop, y start, y stop
                                                            #Columns c1 through c2-1 will be read.
    
    spec_ave = np.average(spectralon.reshape(10*10, b), axis=0)
    
    # Dividing the radiance by reflectance, to get SPD of the lightsource
    lightsource = np.divide(spec_ave, specrefl)
    ##plt.plot_spectra(wvl, lightsource.reshape(1, b), 1, title=f)
        
    # Preparing metadata for the reflectance file
    fname = os.path.splitext(f)[0] + '_refl'
    metadata = hyp.metadata.copy()
    metadata['header file'] = fname+'.hdr'
    metadata['interleave'] = 'bil'
    metadata['header offset'] = 0
    metadata['data type'] = str(4)
    reflhyp = sp.envi.create_image(fname+'.hdr', metadata, force=True)
    
    # Getting the "illuminance" or multiplication factor by dividing 
    # captured radiance by the actual reflectance of the grey patch
    for i in range(r):
        if np.mod(i, 100) == 0:
            print(i, 'of', r)
        for j in range(c):
            pixel = hyp.read_pixel(i, j)
            # Dividing the image (radiance) by the lightsource to get 
            # reflectance
            reflhyp._memmap[i, :, j] = np.divide(pixel, lightsource)
    reflhyp._memmap.flush()


def generate_subsets():
    '''
    '''
    size = 250
    coord_pairs = [[100, 172], # Paper
                   [200, 1024], # Textile- Blue
                   [300, 90], # Textile- Pink/red
                   [400, 1002]] # Textile- Beige
    
    src = get_src()
    f = 'ESR10_O_MOCKUPS_VNIR_1800_SN00841_14993us_2021-08-11T143013_raw_rad.hdr'
    color = np.asarray(Image.open(src + os.path.splitext(f)[0] + '.png')).copy()   #Uint8
    
    hyp = sp.envi.open(src + f)
    wvl = hyp.bands.centers
    meta = hyp.metadata
    bands = hyp.nbands
                
    for i, coord in enumerate(coord_pairs):                #enumerate : adds counter to an iterable and returns it.
        fsub = src + 'Texture_Checker_' + str(i+1)
        sub = write_header((size, size, bands), wvl, fsub, metadata=meta)
        
        x, y = coord        
        for r in range(size):
            for c in range(size):
                pixel = hyp.read_pixel(r+x, c+y)
                sub._memmap[r, :, c] = pixel
        sub._memmap.flush()
        
        subcolor = color[x:x+size, y:y+size, :]
        Image.fromarray(subcolor).save(fsub + '.png')
        

def write_cube(img, wvl, fname, metadata=None, dtype=None):
    '''
    Write an image cube array into disk. Give the entire array value in
    argument, this is only used when memory is available.
    
    Note: Writing the CMF is currently not working if the cube is pre-loaded
    
    Arguments:
        img_arr(numpy.array): 
            Contains the value of image cube to be written.
            
        wvl(numpy.array): 
            Wavelengths.
            
        fname(str): 
            Filename, without file extension.
    '''
    r, c, b = img.shape
    md = _get_metadata((r, c, b), wvl[:], metadata, fname=fname)
    hyp = sp.envi.create_image(
            fname + '.hdr', shape=(r, c, b), dtype='float32', metadata=md,
            force=True)
    for i in range(r):
        for j in range(c):
            hyp._memmap[i, :, j] = img[i, j, :]
    hyp._memmap.flush()
    return hyp
        
def write_header(imshape, wvl, fname, metadata=None, dtype='float32'):
    '''
    Writing the hyperspectral header and returning its object.
    
    Arguments:
        imshape(numpy.array or tuple):
            Array determining the shape of the image in (rows, cols, bands)
            
        wvl(numpy.array):
            List of the center of wavelengths
            
        fname(str):
            Filename, incl. path, excl. extension
            
    Returns: The header.
    '''
    md = _get_metadata(imshape, wvl[:], metadata, fname=fname)
    hdr = fname + '.hdr'
    return sp.envi.create_image(
            hdr, shape=imshape, dtype=dtype, metadata=md, force=True)


def _get_metadata(shape, wvl, hypmeta, fname=None):
    '''
    Get the metadata needed for an ENVI file.
    
    Arguments:
        shape(numpy.array): 
            Three-valued variable: row, col, bands.
            
        wvl(numpy.array): 
            The wavelengths.
    
    Returns: The metadata.
    '''
    if hypmeta is None:
        md = {'lines': shape[0],
          'samples': shape[1],
          'bands': shape[2],
          'data type': np.float32,
          'wavelength units': 'nm',
          'wavelength': wvl[:]}
    else:
        md = hypmeta.copy()
        md['lines'] = shape[0]
        md['samples'] = shape[1]
        md['bands'] = shape[2]
        md['data type'] = np.float32
        md['wavelength units'] = 'nm'
        md['wavelength'] = wvl[:]
        if fname is not None:
            if 'hdr' in fname:
                md['header file'] = os.path.basename(fname)
            else:
                md['header file'] = os.path.basename(fname) + '.hdr'
    return md


if __name__ == '__main__':
    process()
    #generate_subsets()
    collect()


