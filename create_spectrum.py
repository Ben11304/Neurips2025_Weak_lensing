import os
import json
import time
import zipfile
import datetime
import warnings
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utilis import Utility, Data, Visualization, Score, save_config_and_lr, load_config_and_lr
from chunk_handle import get_chunk_indices, iter_chunks, compute_global_image_stats, compute_global_label_stats, save_transform_and_scaler, load_transform_and_scaler


l_edge = np.logspace(2, 4, 11)
Nbins = len(l_edge)-1

def power_spectrum(x, pixsize, kedge):
    """
    Compute the azimuthally averaged 2D power spectrum of a real-valued 2D field.

    Parameters:
    -----------
    x : 2D numpy array
        Input real-space map (e.g., an image or simulated field).
        Must be a 2D array with shape (N_y, N_x).
    
    pixsize : float
        Physical size of each pixel in the map (e.g., arcmin, Mpc, etc.).
        Units should be consistent with the units used for `kedge`.
    
    kedge : 1D array-like
        Bin edges in wavenumber space (k), used to bin the power spectrum.
        Should be monotonically increasing and cover the k-range of interest.

    Returns:
    --------
    power_k : 1D numpy array
        The average wavenumber in each k bin (excluding the DC bin).
    
    power : 1D numpy array
        The binned, azimuthally averaged power spectrum corresponding to `power_k`.
        Normalized per unit area.
    """

    # Ensure the input array is 2D
    assert x.ndim == 2

    # Compute the 2D FFT of the input map and take its squared magnitude (power spectrum)
    xk = np.fft.rfft2(x)  # Real-to-complex FFT (along last axis)
    xk2 = (xk * xk.conj()).real  # Power spectrum: |FFT|^2

    # Get the shape of the input map
    Nmesh = x.shape

    # Compute the wavenumber grid (k-space)
    k = np.zeros((Nmesh[0], Nmesh[1]//2+1))
    # Square of the frequency in the first axis
    k += np.fft.fftfreq(Nmesh[0], d=pixsize).reshape(-1, 1) ** 2
    # Square of the frequency in the second axis (real FFT)
    k += np.fft.rfftfreq(Nmesh[1], d=pixsize).reshape(1, -1) ** 2
    # Convert from (1/length)^2 to angular frequency in radian units
    k = k ** 0.5 * 2 * np.pi

    # Bin each k value according to the bin edges provided in kedge
    index = np.searchsorted(kedge, k)

    # Bin the power values, number of modes, and wavenumbers
    power = np.bincount(index.flatten(), weights=xk2.flatten())
    Nmode = np.bincount(index.flatten())
    power_k = np.bincount(index.flatten(), weights=k.flatten())

    # Adjust for symmetry in the real FFT: include the mirrored part (excluding Nyquist frequency)
    if Nmesh[1] % 2 == 0:  # Even number of columns
        power += np.bincount(index[...,1:-1].flatten(), weights=xk2[...,1:-1].flatten())
        Nmode += np.bincount(index[...,1:-1].flatten())
        power_k += np.bincount(index[...,1:-1].flatten(), weights=k[...,1:-1].flatten())
    else:  # Odd number of columns
        power += np.bincount(index[...,1:].flatten(), weights=xk2[...,1:].flatten())
        Nmode += np.bincount(index[...,1:].flatten())
        power_k += np.bincount(index[...,1:].flatten(), weights=k[...,1:].flatten())

    # Exclude the first bin (typically corresponds to DC mode)
    power = power[1:len(kedge)]
    Nmode = Nmode[1:len(kedge)]
    power_k = power_k[1:len(kedge)]

    # Average the power and wavenumber in each bin, only where Nmode > 0
    select = Nmode > 0
    power[select] = power[select] / Nmode[select]
    power_k[select] = power_k[select] / Nmode[select]

    # Normalize the power spectrum by the map area
    power *= pixsize ** 2 / Nmesh[0] / Nmesh[1]

    # Return the binned k values and corresponding power spectrum
    return power_k, power


num_chunks=25
indices=np.arange(num_chunks)
pixelsize_arcmin = 2 # pixel size in arcmin
pixelsize_radian = pixelsize_arcmin / 60 / 180 * np.pi



for chunk_idx in tqdm(indices, desc="Chunks", leave=False):
            # Load single chunk (assumes it fits in memory)
            noisy_chunk, label_chunk, _ = next(iter_chunks("./dataset/chunk_kappa_noise_new", indices=[chunk_idx]))
            Ncosmo, Nsys = noisy_chunk.shape[0], noisy_chunk.shape[1]
            Cl =  np.zeros((Ncosmo, Nsys, Nbins))

            for i in range(Ncosmo):
                for j in range(Nsys):
                    l, Cl[i,j] = power_spectrum(noisy_chunk[i,j].astype(np.float32
                ), pixelsize_radian, l_edge)

            if not os.path.exists("./dataset/power_scpectrum"):
                os.makedirs("./dataset/power_scpectrum", exist_ok=True)
            Utility.save_np(data_dir="./dataset/power_scpectrum", file_name=f"Spec_chunk_{chunk_idx}.npy",data=Cl)