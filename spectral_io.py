# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 14:18:34 2015

(in progress, to replace spectral_proc.py)
there's problem with CMF converter, related to loaded=True/False in imcolor'

@author: Hilda Deborah
"""

from PIL import Image
from scipy.interpolate import interp1d
import numpy as np
import os
import platform
import spectral as sp
#import spectral_plots as splt


def interp_spectra(spectra, old_wavelength, new_wavelength):
    '''
    Interpolate given spectra to the new wavelength.
    
    Arguments:
        spectrum(numpy.array):
            Input spectra to be interpolated, of size n_spectraxn_wavelength
        
        old_wavelength(numpy.array):
            Wavelength of the input spectrum
            
        new_wavelength(numpy.array):
            New wavelength system to interpolate to
            
    Returns: Spectrum interpolated to the `new_wavelength`
    '''
    return interp1d(old_wavelength, spectra)(new_wavelength)

def get_refblack(shape):
    """
    Return reflectance spectrum of zeros, of shape `shape`.
    
    Note: 19/05/2016
        Instead of all zeros it is modified into a very small number.
    """
    return np.ones(shape, dtype=np.double) * 1e-32

def get_refwhite(shape):
    """
    Return reflectance spectrum of ones, of shape `shape`.
    """
    return np.ones(shape, dtype=np.double)
    
def imcolor(image, method='both', bands=([55, 41, 12]), **kwargs):# wvl=None, scale=1., 
#        preloaded=False, fname=None):
    """
    Get the color image of a spectral image.
    Note: For CMF implementation it still do not handle passing the array, it
        only handles pre-loaded header file.
    
    Arguments:
        image(sp.SpyFile or numpy.array): 
            The spectral image file object. Or in case preloaded=True, this
            is an array of the image values.
            
        method(str): 
            Which method to use, 'cmf' or 'fixed' bands or 'both'.
    
    Returns: The color image (numpy.array)
    """
    wvl = kwargs.get('wvl') 
    if not kwargs.get('preloaded') and wvl is None:
        wvl = image.bands.centers
    preloaded = kwargs.get('preloaded')
    if not preloaded:
        assert kwargs.get('fname'), \
            'Filename `fname` must be given when image is not preloaded'
    fname = kwargs.get('fname')
    
    method = method.lower()
    if method == 'cmf':
        srgb = refl_image_to_srgb(image, **kwargs)
        if fname is not None:
            Image.fromarray(srgb).save(fname + '_CMF.png')
        else:
            return srgb
    elif method == 'fixed':
        fixed = _envi_to_RGB_fixed(image, bands, **kwargs)
        if fname is not None:
            Image.fromarray(fixed).save(fname + '_FIXED.png')
        else:
            return fixed
    elif method == 'both':
        srgb = refl_image_to_srgb(image, **kwargs)
        fixed = _envi_to_RGB_fixed(image, bands, **kwargs)
        if fname is not None:
            Image.fromarray(srgb).save(fname + '_CMF.png')
            Image.fromarray(fixed).save(fname + '_FIXED.png')
        else:
            return srgb, fixed

def normalize_array(arr, to=1.):
    '''
    '''
    m1, m2 = arr.min(), arr.max()
    return to * (arr - m1) / (m2 - m1)
    
def normalize_cube(fname, to=1.0):
    '''
    Normalizing a reflectance image to the range of 0 and `to`. This is often
    needed (although not entirely the most proper thing to do) since a given
    reflectance image often exceeds 1.0 or 100 (in percentage unit).
    '''
    hdr = sp.envi.open(fname)
    m1, m2 = hdr._memmap.min(), hdr._memmap.max()
    if m1 >= 0. and m2 <= to:
        print('The range of values is already between 0 and ' + str(to))
        return
    else:
        wvl = hdr.bands.centers
        rows, cols, bands = hdr.nrows, hdr.ncols, hdr.nbands
        nfname = os.path.splitext(fname)[0] + '_test1'
        nhdr = write_header((rows, cols, bands), wvl, nfname, 
                            metadata=hdr.metadata)
        interleave = hdr.metadata['interleave']
        for i in range(rows):
            for j in range(cols):
                pixel = hdr.read_pixel(i, j)
                npixel = np.divide(pixel-m1, m2-m1)
                if interleave == 'bsq':
                    nhdr._memmap[:, i, j] = npixel
                elif interleave == 'bil':
                    nhdr._memmap[i, :, j] = npixel
        nhdr._memmap.flush()
        imcolor(nhdr, wvl=wvl, fname=nfname)
        return nhdr, nfname
        
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
    
def write_spectrallibrary(fname, wvl, names, spectra):
    metadata = {'spectra names': names[:],
                'wavelength units': 'nm',
                'wavelength': wvl[:]}
    speclib = sp.envi.SpectralLibrary(np.asarray(spectra), metadata, 0)
    speclib.save(fname)
    
def rearrange_cube(hyp, indexmap, fname, bandcutoff=0):
    '''
    Given an image header, this function rearrange the pixels within the image
    cube according to the given indexmap. This function is typically used to
    reconstruct the resulting image of a morphological process.
    
    Arguments:
        hyp(spectral.SpyFile): 
            The header of the original image file.
            
        indexmap(numpy.array):
            The index map required to rearrange the cube, of size rowxcolx2.
            
        fname(str): 
            Filepath for the re-arranged cube.
    '''
    
    r, c, b = hyp.nrows, hyp.ncols, hyp.nbands-(2*bandcutoff)
    b1, b2 = bandcutoff, b + bandcutoff
    wvl = hyp.bands.centers[b1:b2]
    md = _get_metadata((r, c, b), wvl[:], hyp.metadata)
    print(fname)
    img = sp.envi.create_image(
            fname + '.hdr', shape=(r, c, b), dtype='float32', metadata=md,
            force=True)
    for i in range(r):
        for j in range(c):
            img._memmap[i, :, j] = hyp.read_pixel(
                    indexmap[i, j, 0], indexmap[i, j, 1])[b1:b2]
    img._memmap.flush()
    return sp.envi.open(fname + '.hdr')
    
def refl_image_to_srgb(hyp, illuminant='D65', **kwargs):
    '''
    modified from ferdinand's
    
    input assumed to be from 0-1
    '''
    r, c = 0, 0
    scale = 1.0 if not kwargs.get('scale') else kwargs.get('scale')
    preloaded = kwargs.get('preloaded')
    wavelength = kwargs.get('wvl')
    if wavelength is None:
        wavelength = hyp.bands.centers
    if preloaded:
        r, c = hyp.shape[:2]
    else:
        r, c = hyp.nrows, hyp.ncols
        
    cmf, cmf_wvl, band_idx, k = _calc_cmf_function(wavelength, illuminant)
    b1, b2 = band_idx[0], band_idx[-1]+1
    
    # Reflectance to xyz
    xyz_image = np.zeros((r, c, 3))
    for i in range(r):
#        if np.mod(i, 100) == 0:
#            print i, 'of', r
        line = None
        if preloaded:
            line = hyp[i, :, b1:b2].squeeze() / scale
        else:
            line = hyp.read_subregion(
                    (i, i+1), (0, c))[:, :, b1:b2].squeeze() / scale
        #image[i, :, :].squeeze() / scale
        line[line > 1] = 1 # Clipping
            
        # Apply CMF
        out_XYZ = np.zeros((c, 3), np.double)
        for j in range(3):
            out_XYZ[:, j] = np.trapz(line*cmf[j, :], cmf_wvl)
        xyz_image[i, :, :] = (k * out_XYZ) / 100.
    
    # xyz to srgb
    # Matrix
    M_srgb_from_xyz = np.array([
             [ 3.2404542, -1.5371385, -0.4985314],
             [-0.9692660,  1.8760108,  0.0415560],
             [ 0.0556434, -0.2040259,  1.0572252]])
    # Assertions and Reshaping
    assert xyz_image.shape[-1] == 3, 'Last dimension of xyz image must be 3'

    if xyz_image.ndim == 3:
        shape = xyz_image.shape
        xyz_image = np.reshape(xyz_image, [-1, 3])
    else :
        shape = 0
    # Applying the matrix
    srgb = np.dot(xyz_image, M_srgb_from_xyz.T)
    # Gamma correction
    gamma_map = (srgb >  0.0031308)
    srgb[gamma_map]   = 1.055 * np.power(srgb[gamma_map], (1. / 2.4)) - 0.055
    srgb[~gamma_map] *= 12.92
    # Scale to output rounding + clipping
    srgb *= 255
    srgb[srgb < 0] = 0
    srgb[srgb > 255] = 255
    srgb = srgb.round()
    srgb = srgb.astype(np.uint8)
    # Reshaping back
    if shape is not None:
        srgb = np.reshape(srgb, shape)
        
    if kwargs.get('return_xyz'):
        return srgb, xyz_image
    else:
        return srgb


def _calc_cmf_function(imgbands, illumination='D65'):
    '''
    Modified from Ferdinand's
    
    Calculates a CMF, based on global `_CIE_XYZ_1931`, multiplied with the 
    given `illumination` - the values are clipped to the range of `imgbands` 
    limits and interpolated to `imgbands` values 

    Parameters:
        imgbands(numpy.array):
            List of the image bands - the CMF will be truncated between the min
            and max

        illumination(numpy.array/string): 
            np.array: SPD of the illumination -> bands must fit _illumination_list['Wavelength']
            string: entry in the illuination database, call list_illuminations() for further details

    Returns:
        cmf(numpy.array):   
            <length(index_bands)x4>
            [:,0] equas bands == bands[index_bands]
            [:,1] values to calculate X (under specified illumination)
            [:,2] values to calculate Y (under specified illumination)
            [:,3] values to calculate Z (under specified illumination)

        index_bands(np.array): indices of bands, which lie within the CMF limits
    '''
    
    cmf_src = '/Users/hildad/' if platform.system() == 'Darwin'\
        else 'C:/Users/hildad/'
    cmf_data = np.load(
        cmf_src + '/OneDrive - NTNU/spectralmorphology/cmf_data.npz')
    cie_xyz_1931 = cmf_data['CIE_XYZ_1931']
    wvl_cmf = cie_xyz_1931['Wavelength']
    
    # Illumination handling, whether it is from the list of standard illumi-
    # nation or a given SPD
    imgbands = np.array(imgbands)
    own_illum = False
    if isinstance(illumination, str):
        illumination_list = cmf_data['list_illumination']
        # print((illumination_list.dtype.names))
        illumination = illumination_list[illumination]
        wvl_illum = illumination_list['Wavelength']
    else:
        wvl_illum = imgbands
        own_illum = True
        
    # Making sure the limit spectral range of the cmf
    cmf_limits = [np.ceil(max(wvl_illum.min(), wvl_cmf.min())), 
                  np.floor(min(wvl_illum.max(), wvl_cmf.max()))]
    # In case the illumination is from the standard list
    true_limits = (np.ceil(max(imgbands.min(), cmf_limits[0])), 
                   np.floor(min(imgbands.max(), cmf_limits[1])))
    
    index_cmf = np.where(
        (wvl_cmf >= true_limits[0]) & (wvl_cmf <= true_limits[1]))[0]
    index_illum = np.where(
        (wvl_illum >= true_limits[0]) & (wvl_illum <= true_limits[1]))[0]
    index_imgbands = np.where(
        (imgbands >= true_limits[0]) & (imgbands <= true_limits[1]))[0]
    imgbands = imgbands[index_imgbands]
    wvl_cmf = wvl_cmf[index_cmf]
    wvl_illum = wvl_illum[index_illum]
    
    # If illuminant from a given standard, interpolate it to imgbands range
    if own_illum:
        illumination = illumination[index_illum]
    else:
        illumination = interp_spectra(
            illumination[index_illum], wvl_illum, imgbands)
        
    cmf = np.zeros((3, len(wvl_cmf)), float)
    cmf[0, :] = cie_xyz_1931['X'][index_cmf]
    cmf[1, :] = cie_xyz_1931['Y'][index_cmf]    
    cmf[2, :] = cie_xyz_1931['Z'][index_cmf]
    interpcmf = interp_spectra(cmf, wvl_cmf, imgbands)
    
    illumination = np.tile(illumination, (3, 1))
    finalcmf = np.multiply(illumination, interpcmf)
    k = 100 / np.trapz(y=finalcmf[1, :], x=imgbands)

    return finalcmf, imgbands, index_imgbands, k
    
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
    
def _envi_to_RGB_fixed(hyp, bands, **kwargs):
    """
    Get the color image of an input spectral image using fixed bands method.
    The image is processed per band.
    
    Arguments:
        hyp(spectral.SpyFile or numpy.array): 
            The hyperspectral input image. If loaded=True, it is a numpy.array
            and already contains the image values.
            
        bands(numpy.array): 
            3 selected bands to be used.
    
    Returns: The constructed color image, a numpy.array object.
    """
    preloaded = kwargs.get('preloaded')
    row = hyp.shape[0] if preloaded else hyp.nrows
    col = hyp.shape[1] if preloaded else hyp.ncols
    # In case data are not in range [0, 1]
    scale = 1.0 if not kwargs.get('scale') else kwargs.get('scale')
    
    out_img = np.zeros((row, col, 3), dtype=np.uint8)
    for i in range(3):
        # Image read per band
        if not preloaded:
            out_img[:, :, i] = hyp.read_band(bands[i]) * 255 / scale
        else:
            out_img[:, :, i] = hyp[:, :, bands[i]] * 255 / scale
    return out_img


if __name__ == '__main__':
    src = '/Users/hildad/OneDrive - NTNU/Hyperspectral Data/Hytexila/ENVI/'\
        'food/food_rice/'
    f = src + 'food_rice.hdr'
    im = sp.envi.open(f)
    wvl = im.bands.centers
    illum = np.array(im.metadata['illuminant'][0].split(';'), float)
    # splt.spectralset(illum, wvl)
    srgb = refl_image_to_srgb(im, illuminant='Calculated_Data_4_LED_with_yellow_LED', wvl=wvl)
    Image.fromarray(srgb).show()
    
    # imcol = np.array(Image.open('/Users/hildad/OneDrive - NTNU/Hyperspectral'\
    #                             ' Data/Hytexila/food_rice_rgb.png'))
    # r,c,b = imcol.shape
    # print((imcol-srgb).sum()/(r*c*b))
    