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
from utilis import Utility, Data, Visualization, Score, save_config_and_lr, load_config_and_lr, CosmologyDataset
from chunk_handle import get_chunk_indices, iter_chunks, compute_global_image_stats, compute_global_label_stats, save_transform_and_scaler, load_transform_and_scaler
from scipy.interpolate import LinearNDInterpolator
import pickle







def power_spectrum(x, pixsize, kedge):
    assert x.ndim == 2
    xk = np.fft.rfft2(x)  # Real-to-complex FFT (along last axis)
    xk2 = (xk * xk.conj()).real  # Power spectrum: |FFT|^2
    Nmesh = x.shape

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


l_edge = np.logspace(2, 4, 11)
Nbins = len(l_edge)-1
pixelsize_arcmin = 2 # pixel size in arcmin
pixelsize_radian = pixelsize_arcmin / 60 / 180 * np.pi


class Inference():
    @staticmethod
    def inference(model, config, history = None ):
        print("ready for inference")

        # Ensure model is on the requested device
        try:
            model.to(config.DEVICE)
        except Exception:
            # Some models may already be on the correct device or not support .to in this context
            pass

        transform, label_scaler = load_transform_and_scaler(transform_file='./side_module/transform_params.pkl', scaler_file='./side_module/label_scaler.pkl')
        # if history is not None:
        #     total_val_datasets = history['total_val_datasets']
        #     train_losses = history['train_losses']
        #     total_samples = history['total_samples']
        #     total_epochs = history['total_epochs']

        save_dir = f"./dataset/val_set/"
        os.makedirs(save_dir, exist_ok=True)

        # if os.path.exists(val_data_path) and os.path.exists(val_labels_path) and os.path.exists(val_specs_path):
        #     all_val_data = np.load(val_data_path)
        #     all_val_labels = np.load(val_labels_path)
        #     Cl = np.load(val_specs_path)
        #     print(f"Loaded validation data from {save_dir}")
        # # Đường dẫn file lưu trữ
        val_data_path = os.path.join(save_dir, "all_val_data.npy")
        val_labels_path = os.path.join(save_dir, "all_val_labels.npy")
        val_specs_path = os.path.join(save_dir, "all_val_specs.npy")
        metadata_path = os.path.join(save_dir, "validation_metadata.pkl")
        Val=True

        if Val:
            if os.path.exists(val_data_path) and os.path.exists(val_labels_path) and os.path.exists(val_specs_path):
                all_val_data = np.load(val_data_path)
                all_val_labels = np.load(val_labels_path)
                Cl = np.load(val_specs_path)
                print(f"Loaded validation data from {save_dir}")
            else:
                all_val_data = np.concatenate([ds.data for ds in total_val_datasets], axis=0)
                all_val_labels = np.concatenate([ds.labels for ds in total_val_datasets], axis=0)
                Nval = all_val_data.shape[0]
                Cl = np.zeros((Nval, Nbins))


                for i in range(Nval):
                    l, Cl[i] = power_spectrum(all_val_data[i].astype(np.float32), pixelsize_radian, l_edge)
                
                # Lưu dữ liệu vào file
                np.save(val_data_path, all_val_data)
                np.save(val_labels_path, all_val_labels)
                np.save(val_specs_path, Cl)
                print(f"Saved validation data to {save_dir}")

            # Lưu metadata (config, batch_size, v.v.) để tái sử dụng
            metadata = {
                "batch_size": config.BATCH_SIZE,
                "transform": transform,  # Lưu transform nếu có thể serialize
                "shape": all_val_data.shape
            }
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)

            # Tạo dataset và dataloader
            concatenated_val_dataset = CosmologyDataset(
                data=all_val_data,
                labels=all_val_labels,
                specs=Cl # Use transform from first dataset
            )
            val_loader = DataLoader(
                concatenated_val_dataset,
                batch_size=config.BATCH_SIZE,
                shuffle=False
            )
        else:
            concatenated_val_loader = None
            print("No validation datasets to concatenate.")


        model.eval()
        device=config.DEVICE
        y_pred_list = []   
        pbar = tqdm(val_loader, total=len(val_loader), desc="Validating")
        with torch.no_grad():
            for (images, specs), targets in val_loader:
                images = images.to(device, dtype=torch.float32)
                specs = specs.to(device, dtype=torch.float32)
                targets = targets.to(device, dtype=torch.float32)
                y_pred = model((images, specs))
                y_pred = label_scaler.inverse_transform(y_pred.cpu().numpy())
                y_pred_list.append(y_pred) 

        mean_val = np.concatenate(y_pred_list, axis=0)

        # Initialize Data class object
        data_obj = Data(data_dir="./dataset", USE_PUBLIC_DATASET=True)

        # Load train data
        data_obj.load_train_data()

        # Load test data
        data_obj.load_test_data()
        Ncosmo = data_obj.Ncosmo
        Nsys = data_obj.Nsys


        all_val_labels_inv = all_val_labels
        cosmology = data_obj.label[:,0,:2]   
        dists = np.sum((all_val_labels_inv[:, None, :] - cosmology[None, :, :])**2, axis=2)  # (Nval, Ncosmo)
        nearest_idx = np.argmin(dists, axis=1)  # for each validation sample, index of closest cosmology

        # build index lists grouped by cosmology
        index_lists = [[] for _ in range(cosmology.shape[0])]
        for val_idx, cosmo_i in enumerate(nearest_idx):
            index_lists[cosmo_i].append(val_idx)

        val_cosmology_idx = [np.array(lst, dtype=np.int32) for lst in index_lists]
        # The summary statistics of all realizations for all cosmologies in the validation set
        d_vector = []  
        n_d = 2   # Number of summary statistics for each map
        for i in range(Ncosmo):
            d_i =  np.zeros((len(val_cosmology_idx[i]), n_d))  
            for j, idx in enumerate(val_cosmology_idx[i]):
                d_i[j] = mean_val[idx]

            d_vector.append(d_i)

        # mean summary statistics (average over all realizations)
        mean_d_vector = []
        for i in range(Ncosmo):
            # If no realizations for this cosmology, mark with NaNs so it will be ignored by interpolator
            if d_vector[i].size == 0:
                mean_d_vector.append(np.full((n_d,), np.nan))
            else:
                mean_d_vector.append(np.mean(d_vector[i], 0))
        mean_d_vector = np.array(mean_d_vector)

        # covariance matrix
        delta = []
        for i in range(Ncosmo):
            delta.append((d_vector[i] - mean_d_vector[i].reshape(1, n_d))) 

        # Build covariance matrices with safety checks to avoid division by zero / singular matrices
        cov_list = []
        for i in range(Ncosmo):
            denom = (len(delta[i]) - n_d - 2)
            if denom <= 0 or delta[i].size == 0:
                # Not enough samples to estimate covariance; fall back to tiny diagonal covariance
                cov = np.eye(n_d, dtype=float) * 1e-6
            else:
                cov = (delta[i].T @ delta[i]) / denom
                # If covariance contains NaNs or infs, replace with small diagonal
                if not np.all(np.isfinite(cov)):
                    cov = np.eye(n_d, dtype=float) * 1e-6
            cov_list.append(cov[None])
        cov_d_vector = np.concatenate(cov_list, 0)
        mean_d_vector_interp = LinearNDInterpolator(cosmology, mean_d_vector, fill_value=np.nan)
        cov_d_vector_interp = LinearNDInterpolator(cosmology, cov_d_vector, fill_value=np.nan)

        logprior_interp = LinearNDInterpolator(cosmology, np.zeros((Ncosmo, 1)), fill_value=-np.inf)

        def log_prior(x):
            logprior = logprior_interp(x).flatten()  # shape = (Ntest, ) 
            return logprior

        # Gaussian likelihood with interpolated mean and covariance matrix
        def loglike(x, d):
            mean = mean_d_vector_interp(x)
            cov = cov_d_vector_interp(x)
            delta = d - mean

            # cov may be (n_d,) when interpolation returns scalar or malformed — ensure correct shape
            # If cov has shape (n_d,) treat as diagonal
            try:
                # Ensure cov is an array with shape (..., n_d, n_d)
                cov = np.asarray(cov)
            except Exception:
                return -np.inf * np.ones(len(delta))

            # If interpolation returns a single covariance (same for all), broadcast
            if cov.ndim == 2 and cov.shape == (n_d, n_d):
                cov = np.tile(cov[None], (len(delta), 1, 1))

            # Prepare outputs (default to -inf for invalid entries)
            out = np.full(len(delta), -np.inf, dtype=float)

            for idx in range(len(delta)):
                # If mean or delta has NaNs for this entry, skip (leave -inf)
                try:
                    mean_i = mean[idx]
                except Exception:
                    mean_i = None
                if mean_i is None or not np.all(np.isfinite(mean_i)):
                    out[idx] = -np.inf
                    continue
                Ci = cov[idx]
                # Replace non-finite entries in covariance
                if not np.all(np.isfinite(Ci)):
                    Ci = np.eye(n_d, dtype=float) * 1e-6

                # Add small jitter proportional to trace to help numerical stability
                trace = np.trace(Ci)
                if np.isfinite(trace) and trace > 0:
                    eps = 1e-8 * trace
                else:
                    eps = 1e-6
                Ci = Ci + np.eye(n_d) * eps

                try:
                    inv_Ci = np.linalg.inv(Ci)
                    sign, logdet = np.linalg.slogdet(Ci)
                    if sign <= 0 or not np.isfinite(logdet):
                        # fallback to pseudo-inverse
                        inv_Ci = np.linalg.pinv(Ci)
                        logdet = np.log(np.linalg.det(Ci) + 1e-20) if np.linalg.det(Ci) != 0 else 0.0
                except np.linalg.LinAlgError:
                    inv_Ci = np.linalg.pinv(Ci)
                    # compute a stable logdet approximation
                    try:
                        s = np.linalg.svd(Ci, compute_uv=False)
                        logdet = np.sum(np.log(s[s > 0]))
                    except Exception:
                        logdet = 0.0

                # Mahalanobis term
                try:
                    maha = np.einsum("i,ij,j->", delta[idx], inv_Ci, delta[idx])
                except Exception:
                    maha = np.inf

                if np.isfinite(maha):
                    out[idx] = -0.5 * logdet - 0.5 * maha
                else:
                    out[idx] = -np.inf

            return out

        def logp_posterior(x, d):
            logp = log_prior(x)
            select = np.isfinite(logp)
            if np.sum(select) > 0:
                ll = loglike(x[select], d[select])
                # Replace any non-finite log-likelihood values with -inf before adding
                ll = np.where(np.isfinite(ll), ll, -np.inf)
                logp[select] = logp[select] + ll
            return logp

        Nval=all_val_labels_inv.shape[0]


        Nstep = 10000  # Number of MCMC steps (iterations)
        sigma = 0.06   # Proposal standard deviation; should be tuned per method or parameter scale

        # Randomly select initial points from the `cosmology` array for each test case
        # Assumes `cosmology` has shape (Ncosmo, ndim) and `Ntest` is the number of independent chains/samples
        current = cosmology[np.random.choice(Ncosmo, size=Nval)]

        # Compute log-posterior at the initial points
        curr_logprob = logp_posterior(current, mean_val)

        # List to store sampled states (for all chains)
        states = []

        # Track total acceptance probabilities to compute acceptance rates
        total_acc = np.zeros(len(current))

        t = time.time()  # Track time for performance reporting

        # MCMC loop
        for i in range(Nstep):

            # Generate proposals by adding Gaussian noise to current state
            proposal = current + np.random.randn(*current.shape) * sigma    

            # Compute log-posterior at the proposed points
            proposal_logprob = logp_posterior(proposal, mean_val)

            # Compute log acceptance ratio (Metropolis-Hastings)
            acc_logprob = proposal_logprob - curr_logprob
            acc_logprob[acc_logprob > 0] = 0  # Cap at 0 to avoid exp overflow (acceptance prob ≤ 1)

            # Convert to acceptance probabilities
            acc_prob = np.exp(acc_logprob)

            # Decide whether to accept each proposal
            acc = np.random.uniform(size=len(current)) < acc_prob

            # Track acceptance probabilities (not binary outcomes)
            total_acc += acc_prob

            # Update states and log-probs where proposals are accepted
            current[acc] = proposal[acc]
            curr_logprob[acc] = proposal_logprob[acc]

            # Save a copy of the current state
            states.append(np.copy(current)[None])

            # Periodically print progress and acceptance rates
            if i % (0.1*Nstep) == 0.1*Nstep-1:
                print(
                    'step:', len(states),
                    'Time:', time.time() - t,
                    'Min acceptance rate:', np.min(total_acc / (i + 1)),
                    'Mean acceptance rate:', np.mean(total_acc / (i + 1))
                )
                t = time.time()
        y_pred_val=mean_val
        # remove burn-in
        states = np.concatenate(states[int(0.2*Nstep):], 0)

        # mean and std of samples
        mean_val = np.mean(states, 0)
        errorbar_val = np.std(states, 0)
        y_val=all_val_labels_inv

        test_Cl = np.zeros((data_obj.Ntest, Nbins))

        for i in range(data_obj.Ntest):
            l, test_Cl[i] = power_spectrum(data_obj.kappa_test[i].astype(np.float64), data_obj.pixelsize_radian, l_edge)

        test_logCl = np.log10(test_Cl)
        test_dataset = CosmologyDataset(
            data=data_obj.kappa_test, 
            specs=test_Cl,
            transform=transform
        )

        test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
        model.eval()
        y_pred_list = []   
        pbar = tqdm(test_loader, total=len(test_loader), desc="Inference on the test set")
        with torch.no_grad():
            for images, specs in test_loader:
                images = images.to(device, dtype=torch.float32)
                specs = specs.to(device, dtype=torch.float32)
                y_pred = model((images, specs))      
                y_pred = label_scaler.inverse_transform(y_pred.cpu().numpy())
                y_pred_list.append(y_pred) 

        y_pred_test = np.concatenate(y_pred_list, axis=0)


        Nstep = 10000  # Number of MCMC steps (iterations)
        sigma = 0.06   # Proposal standard deviation; should be tuned per method or parameter scale

        # Randomly select initial points from the `cosmology` array for each test case
        # Assumes `cosmology` has shape (Ncosmo, ndim) and `Ntest` is the number of independent chains/samples
        current = cosmology[np.random.choice(Ncosmo, size=data_obj.Ntest)]

        # Compute log-posterior at the initial points
        curr_logprob = logp_posterior(current, y_pred_test)

        # List to store sampled states (for all chains)
        states = []

        # Track total acceptance probabilities to compute acceptance rates
        total_acc = np.zeros(len(current))

        t = time.time()  # Track time for performance reporting

        # MCMC loop
        for i in range(Nstep):

            # Generate proposals by adding Gaussian noise to current state
            proposal = current + np.random.randn(*current.shape) * sigma    

            # Compute log-posterior at the proposed points
            proposal_logprob = logp_posterior(proposal, y_pred_test)

            # Compute log acceptance ratio (Metropolis-Hastings)
            acc_logprob = proposal_logprob - curr_logprob
            acc_logprob[acc_logprob > 0] = 0  # Cap at 0 to avoid exp overflow (acceptance prob ≤ 1)

            # Convert to acceptance probabilities
            acc_prob = np.exp(acc_logprob)

            # Decide whether to accept each proposal
            acc = np.random.uniform(size=len(current)) < acc_prob

            # Track acceptance probabilities (not binary outcomes)
            total_acc += acc_prob

            # Update states and log-probs where proposals are accepted
            current[acc] = proposal[acc]
            curr_logprob[acc] = proposal_logprob[acc]

            # Save a copy of the current state
            states.append(np.copy(current)[None])

            # Periodically print progress and acceptance rates
            if i % (0.1*Nstep) == 0.1*Nstep-1:
                print(
                    'step:', len(states),
                    'Time:', time.time() - t,
                    'Min acceptance rate:', np.min(total_acc / (i + 1)),
                    'Mean acceptance rate:', np.mean(total_acc / (i + 1))
                )
                t = time.time() 
                
        states = np.concatenate(states[int(0.2*Nstep):], 0)

        # mean and std of samples
        mean = np.mean(states, 0)
        errorbar = np.std(states, 0)


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
