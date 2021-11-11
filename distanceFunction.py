import numpy as np


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 18:56:08 2021

@author: clr1
"""

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