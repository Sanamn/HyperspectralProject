#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 18:49:32 2021

@author: clr1
"""

import numpy as np
import spectral as sp # http://www.spectralpython.net/
import os





def get_target_spectra(filename):
    
    hdr = sp.envi.open(filename)
    wvl = hdr.bands.centers
    w, h = 3, 3
    (x,y) = (288,1140)  #coordinates
    target = hdr[y:y+h, x:x+w, :]
   
    return target
    


def get_reference_spectra():
    
     x = np.ones((1, 1, 186))
     return(x)
    
    

def replicate_ref(ref, r, c, d):
    """
    Replicate reference spectrum matrix to the size of target spectra
    so that distance computation can be done in matrix operation
    instead of iteration. If ref is already the shape of r, c, d, re-
    turn ref.

    Parameters:
    - `ref`: reference spectrum, of dimension 1x1x(# of wavelengths).
    - `r`: number of rows in the target matrix.
    - `c`: number of columns in the target matrix.
    - `d`: number of wavelengths in the target matrix.

    Returns: replicated reference spectrum, of equal dimension as the
    target spectra matrix.
    """
    if len(ref.shape) == 1:
        ref = ref.reshape(1, 1, ref.shape[0])
    r1, c1 = ref.shape[:2]
    if r1 == r and c1 == c:
        return ref
    else:
        return np.tile(ref, r * c).reshape(r, c, d)
    
    
    
def get_distance_values(
        ref, B, fun, resolution=1., alpha=1., manifold=False, manifold_df=None,
        n_neighbors=10, n_components=2):
    """
    Interprets the distance function stringname and calls its
    corresponding method.

    Parameters:
    - `ref`: reference spectrum to calculate distance from, of
      dimension 1x1x(# of wavelength)
    - `B`: target spectra, of 3d image dimension, i.e.
      (# of rows)x(# of cols)x(# of wavelength)
    - `fun`: distance function stringname. In case of manifold
      distance, this string specifies the embedded distance function
      inside the manifold distance, which measures distance between
      two spectra.
    - `manifold`: if distance in manifold and a distance of dataset
      or region instead of distance between two spectra.
    - `manifold_df`: the manifold distance function name.

    Return: Distance values between reference spectrum and all target
    spectra for the given distance function.
    """
    fun = fun.lower()

    ndim = len(B.shape)
    row, col, wvl = 0, 0, 0
    if ndim == 3:
        row, col, wvl = B.shape
    elif ndim == 2:
        nentries, nwvls = B.shape
        B = B.reshape(1, nentries, nwvls)
        row, col, wvl = B.shape
        ref = ref.reshape(1, 1, nwvls)

    A = replicate_ref(ref, row, col, wvl)
    distance_values = np.zeros((row, col), np.double)

    # Pseudo-divergences
    if fun == 'kl':
        distance_values = KL(normalize_spectra(A), normalize_spectra(B))
    
    elif fun == 'spectral kl' or fun == 'pseudodiv' or fun == 'klpd':
        distance_values = pseudodiv_KL(A, B, resolution=resolution, mode=0)
    elif fun == 'pseudodiv-total' or fun == 'klpd-total':
        distance_values = pseudodiv_KL(A, B, resolution=resolution, mode=3)
    elif fun == 'klpd-shape' or fun == 'pseudodiv-shape':
        distance_values = pseudodiv_KL(A, B, resolution=resolution, mode=1)
    elif fun == 'klpd-energy' or fun == 'pseudodiv-energy':
        distance_values = pseudodiv_KL(A, B, resolution=resolution, mode=2)
    

    else:
        print(fun, 'distance function not found')
        return

    return np.nan_to_num(distance_values)



def pseudodiv_KL(A, B, resolution=1., mode=0):
    """
    Kullback-Leibler pseudo-divergence for spectral data, integration method is
    assumed to be trapezoidal.

    Parameters:
    - `A`: reference matrix, of dimension rowxcolxwavelength.
    - `B`: target matrix, of dimension rowxcolxwavelength.
    - `mode`: whether to return both components (default, 0), shape(1), 
        energy(2), or total (summation, 3)

    Return: distance matrix, of dimension rowxcol
    """    
    if len(A.shape) == 1:
        # This handles pairwise distance with this function as metric callable
        A = A[np.newaxis, np.newaxis, :]
        B = B[np.newaxis, np.newaxis, :]
        # If mode is not setup, by default total klpd is given
        if mode == 0:
            mode = 3
        
    kA, n_A = normalize_spectra(A, get_w=True, resolution=resolution)
    kB, n_B = normalize_spectra(B, get_w=True, resolution=resolution)
    shape = (kA * KL(n_A, n_B, resolution=resolution)) + (
            kB * KL(n_B, n_A, resolution=resolution))
    energy = (kA - kB) * (np.log(kA) - np.log(kB))

    if mode == 0:
        return np.concatenate((shape[:, :, None], energy[:, :, None]), axis=2)
    elif mode == 1:
        return shape
    elif mode == 2:
        return energy
    else:
        return shape + energy

def KL(A, B, resolution=1.):
    """
    Kullback-Leibler, the original divergence. Input is assumed to be
    normalized to one.

    Parameters:
    - `A`: reference matrix, of dimension rowxcolxwavelength.
    - `B`: target matrix, of dimension rowxcolxwavelength.

    Return: divergence matrix, of dimension rowxcol
    """
    scale = 1e6
#    test = len(A[A==0])+len(B[B==0])
#    if test > 0:
#        print '0 values:', test
    part_A = np.nan_to_num(np.log(np.multiply(A, scale)) - np.log(scale))
    part_B = np.nan_to_num(np.log(np.multiply(B, scale)) - np.log(scale))
    div_KL = np.multiply(A, (part_A - part_B))
    return np.trapz(div_KL, dx=resolution, axis=2)


def normalize_spectra(A, get_w=False, resolution=1.):
    '''
    Normalize each spectrum to the sum of values at each of its wavelengths. If
    integration is True, the normalizing factor is integration instead,
    trapezoidal rule is assumed.

    Arguments
        A(np.array):
            Input matrix, of dimension rowxcolxwavelength.

        get_w(bool):
            Whether to return the normalizing factor.

        integration(bool):
            Whether to use integration as normalizing factor.

    Returns: `A` normalized into probability matrix.
    '''
    A[A <= 0.] = 1e-9  # Handling for zero values

    r, c, b = np.shape(A)
    norm_factor = np.trapz(A, dx=resolution, axis=2)
    norm_factor = norm_factor.reshape(r, c, 1)
    norm_factor = np.tile(norm_factor, (1, 1, b))
    if get_w:
        return norm_factor[:, :, 0], np.divide(A, norm_factor)
    else:
        return np.divide(A, norm_factor)
    
    

# main function
if __name__ == '__main__':
   
    cwd = os.getcwd()
    #print(cwd)
    MainfileName =  '/Files/VNIR-varnishC1/ESR10_mockup1_001_VNIR_1800_SN00841_14998us_2021-10-19T115801_raw_rad_refl.hdr'
    f = cwd + MainfileName   
   
    ref = get_reference_spectra() 
    b = get_target_spectra(f)
   
    get_distance_values(ref,b,'spectral kl')
   
    #get_distance_values()
   
   
   
   #src = get_src()
   #f = src + 'Files\\ESR10_O_MOCKUPS_VNIR_1800_SN00841_14993us_2021-08-11T143013_raw_rad.hdr'   
   #hdr = sp.envi.open(f)
   #get_refl_from_rad(hdr)
   #collect()
   





    