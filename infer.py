# inference.py
# This script performs inference on the test data using a pre-trained Spectrum_CNN model.
# It computes power spectra, runs predictions, performs MCMC sampling for posteriors,
# and saves the results in a zipped JSON file.

import os
import json
import time
import zipfile
import datetime
import warnings
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.preprocessing import StandardScaler
from utilis import Utility, Data, Visualization, Score, save_config_and_lr, load_config_and_lr
from chunk_handle import get_chunk_indices, iter_chunks, compute_global_image_stats, compute_global_label_stats, save_transform_and_scaler, load_transform_and_scaler
from Models.Models import Spectrum_CNN  # Assuming this is where Spectrum_CNN is defined
from scipy.interpolate import LinearNDInterpolator
from utilis import CosmologyDataset

# Suppress warnings if needed
warnings.filterwarnings("ignore")

# Define the power spectrum function (copied from the notebook)
def power_spectrum(x, pixsize, kedge):
    """
    Compute the azimuthally averaged 2D power spectrum of a real-valued 2D field.
    """
    assert x.ndim == 2
    xk = np.fft.rfft2(x)
    xk2 = (xk * xk.conj()).real
    Nmesh = x.shape
    k = np.zeros((Nmesh[0], Nmesh[1]//2+1))
    k += np.fft.fftfreq(Nmesh[0], d=pixsize).reshape(-1, 1) ** 2
    k += np.fft.rfftfreq(Nmesh[1], d=pixsize).reshape(1, -1) ** 2
    k = k ** 0.5 * 2 * np.pi
    index = np.searchsorted(kedge, k)
    power = np.bincount(index.flatten(), weights=xk2.flatten())
    Nmode = np.bincount(index.flatten())
    power_k = np.bincount(index.flatten(), weights=k.flatten())
    if Nmesh[1] % 2 == 0:
        power += np.bincount(index[...,1:-1].flatten(), weights=xk2[...,1:-1].flatten())
        Nmode += np.bincount(index[...,1:-1].flatten())
        power_k += np.bincount(index[...,1:-1].flatten(), weights=k[...,1:-1].flatten())
    else:
        power += np.bincount(index[...,1:].flatten(), weights=xk2[...,1:].flatten())
        Nmode += np.bincount(index[...,1:].flatten())
        power_k += np.bincount(index[...,1:].flatten(), weights=k[...,1:].flatten())
    power = power[1:len(kedge)]
    Nmode = Nmode[1:len(kedge)]
    power_k = power_k[1:len(kedge)]
    select = Nmode > 0
    power[select] = power[select] / Nmode[select]
    power_k[select] = power_k[select] / Nmode[select]
    power *= pixsize ** 2 / Nmesh[0] / Nmesh[1]
    return power_k, power

# Define CosmologyDataset (assuming it's in utilis or similar; define here if needed)
# class CosmologyDataset(Dataset):
#     def __init__(self, data, specs=None, labels=None, transform=None):
#         self.data = data
#         self.specs = specs if specs is not None else np.zeros((len(data), 10))  # Placeholder for specs
#         self.labels = labels if labels is not None else np.zeros((len(data), 2))  # Placeholder for labels
#         self.transform = transform

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         image = self.data[idx]
#         spec = self.specs[idx]
#         label = self.labels[idx]
#         if self.transform:
#             image = self.transform(image)
#         return (image, spec), label

# Hardcoded configurations (from Config in notebook)
class Config:
    IMG_HEIGHT = 1424
    IMG_WIDTH = 176
    NUM_TARGETS = 2  # Omega_m, S_8
    BATCH_SIZE = 101
    DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    MODEL_SAVE_PATH = "./side_module_ViT/model_20250921_015923_on/best_model.pth"  # Replace with your model path

config = Config()

# Main inference function
def run_inference():
    # Load transform and label scaler
    transform, label_scaler = load_transform_and_scaler(
        transform_file='./side_module/transform_params.pkl',
        scaler_file='./side_module/label_scaler.pkl'
    )
    # Load model
    model = Spectrum_CNN(config.IMG_HEIGHT, config.IMG_WIDTH, config.NUM_TARGETS)
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE, weights_only=True))
    model.to(config.DEVICE)
    model.eval()
    print(f"Model loaded from {config.MODEL_SAVE_PATH}")

    # Load data
    data_obj = Data(data_dir="./dataset", USE_PUBLIC_DATASET=True)
    data_obj.load_train_data()  # Needed for cosmology
    data_obj.load_test_data()
    print(f"Loaded test data with shape: {data_obj.kappa_test.shape}")

    # Compute power spectrum for test data
    l_edge = np.logspace(2, 4, 11)
    Nbins = len(l_edge) - 1
    pixelsize_arcmin = 2
    pixelsize_radian = pixelsize_arcmin / 60 / 180 * np.pi
    test_Cl = np.zeros((data_obj.Ntest, Nbins))
    for i in tqdm(range(data_obj.Ntest), desc="Computing Power Spectra"):
        _, test_Cl[i] = power_spectrum(data_obj.kappa_test[i].astype(np.float64), pixelsize_radian, l_edge)

    # Create test dataset and loader
    test_dataset = CosmologyDataset(
        data=data_obj.kappa_test,
        specs=test_Cl,
        transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # Inference
    y_pred_list = []
    with torch.no_grad():
        for (images, specs) in tqdm(test_loader, desc="Inference"):
            images = images.to(config.DEVICE, dtype=torch.float32)

            specs = specs.to(config.DEVICE, dtype=torch.float32)
            y_pred = model((images, specs))
            y_pred = label_scaler.inverse_transform(y_pred.cpu().numpy())
            y_pred_list.append(y_pred)
    y_pred_test = np.concatenate(y_pred_list, axis=0)
    print(f"Inference completed. Predictions shape: {y_pred_test.shape}")



    # Load cosmology from train data
    cosmology = data_obj.label[:, 0, :2]  # (Ncosmo, 2)
    Ncosmo = cosmology.shape[0]

    # Interpolators for mean and cov (from validation; assume precomputed or recompute if needed)
    # NOTE: In the notebook, these are computed from validation. For real inference, load or recompute.
    # Here, assuming you have mean_d_vector and cov_d_vector from validation; replace with actual loading.
    # For demo, placeholder - YOU NEED TO LOAD OR COMPUTE THESE.
    # Example: Load from files if saved.
    mean_d_vector = np.load('./path_to_mean_d_vector.npy')  # Replace with actual path
    cov_d_vector = np.load('./path_to_cov_d_vector.npy')    # Replace with actual path

    mean_d_vector_interp = LinearNDInterpolator(cosmology, mean_d_vector, fill_value=np.nan)
    cov_d_vector_interp = LinearNDInterpolator(cosmology, cov_d_vector, fill_value=np.nan)
    logprior_interp = LinearNDInterpolator(cosmology, np.zeros((Ncosmo, 1)), fill_value=-np.inf)

    # Define prior, likelihood, posterior
    def log_prior(x):
        logprior = logprior_interp(x).flatten()
        return logprior

    def loglike(x, d):
        mean = mean_d_vector_interp(x)
        cov = cov_d_vector_interp(x)
        delta = d - mean
        inv_cov = np.linalg.inv(cov)
        cov_det = np.linalg.slogdet(cov)[1]
        return -0.5 * cov_det - 0.5 * np.einsum("ni,nij,nj->n", delta, inv_cov, delta)

    def logp_posterior(x, d):
        logp = log_prior(x)
        select = np.isfinite(logp)
        if np.sum(select) > 0:
            logp[select] += loglike(x[select], d[select])
        return logp

    # MCMC sampling
    Nstep = 10000
    sigma = 0.06
    current = cosmology[np.random.choice(Ncosmo, size=data_obj.Ntest)]
    curr_logprob = logp_posterior(current, y_pred_test)
    states = []
    total_acc = np.zeros(len(current))
    t = time.time()
    for i in tqdm(range(Nstep), desc="MCMC Steps"):
        proposal = current + np.random.randn(*current.shape) * sigma
        proposal_logprob = logp_posterior(proposal, y_pred_test)
        acc_logprob = proposal_logprob - curr_logprob
        acc_logprob[acc_logprob > 0] = 0
        acc_prob = np.exp(acc_logprob)
        acc = np.random.uniform(size=len(current)) < acc_prob
        total_acc += acc_prob
        current[acc] = proposal[acc]
        curr_logprob[acc] = proposal_logprob[acc]
        states.append(np.copy(current)[None])
        if i % (0.1 * Nstep) == 0.1 * Nstep - 1:
            print(f"Step: {len(states)}, Time: {time.time() - t}, Min acc: {np.min(total_acc / (i + 1))}, Mean acc: {np.mean(total_acc / (i + 1))}")
            t = time.time()

    # Process MCMC results
    states = np.concatenate(states[int(0.2 * Nstep):], 0)
    mean = np.mean(states, 0)
    errorbar = np.std(states, 0)
    print(f"MCMC completed. Mean shape: {mean.shape}, Errorbar shape: {errorbar.shape}")

    # Save results
    data = {"means": mean.tolist(), "errorbars": errorbar.tolist()}
    the_date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
    zip_file_name = 'Submission_' + the_date + '.zip'
    zip_file = Utility.save_json_zip(
        submission_dir="submissions",
        json_file_name="result.json",
        zip_file_name=zip_file_name,
        data=data
    )
    print(f"Submission ZIP saved at: {zip_file}")

if __name__ == "__main__":
    run_inference()