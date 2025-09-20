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
import numpy as np
from tqdm import tqdm
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import numpy as np
from typing import Optional, Tuple, Generator, Dict, Any
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
import time
from tqdm import tqdm
import pickle



class Utility:
    @staticmethod
    def add_noise(data, mask, ng, pixel_size=2.):
        """
        Add noise to a noiseless convergence map.

        Parameters
        ----------
        data : np.array
            Noiseless convergence maps.
        mask : np.array
            Binary mask map.
        ng : float
            Number of galaxies per arcmin². This determines the noise level; a larger number means smaller noise.
        pixel_size : float, optional
            Pixel size in arcminutes (default is 2.0).
        """

        return data + np.random.randn(*data.shape) * 0.4 / (2*ng*pixel_size**2)**0.5 * mask
    
    @staticmethod
    def load_np(data_dir, file_name):
        file_path = os.path.join(data_dir, file_name)
        return np.load(file_path)

    @staticmethod
    def save_np(data_dir, file_name, data):
        file_path = os.path.join(data_dir, file_name)
        np.save(file_path, data)

    @staticmethod
    def save_json_zip(submission_dir, json_file_name, zip_file_name, data):
        """
        Save a dictionary with 'means' and 'errorbars' into a JSON file,
        then compress it into a ZIP file inside submission_dir.

        Parameters
        ----------
        submission_dir : str
            Path to the directory where the ZIP file will be saved.
        file_name : str
            Name of the ZIP file (without extension).
        data : dict
            Dictionary with keys 'means' and 'errorbars'.

        Returns
        -------
        str
            Path to the created ZIP file.
        """
        os.makedirs(submission_dir, exist_ok=True)

        json_path = os.path.join(submission_dir, json_file_name)

        # Save JSON file
        with open(json_path, "w") as f:
            json.dump(data, f)

        # Path to ZIP
        zip_path = os.path.join(submission_dir, zip_file_name)

        # Create ZIP containing only the JSON
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(json_path, arcname=json_file_name)

        # Remove the standalone JSON after zipping
        os.remove(json_path)

        return zip_path
    

class Data:
    def __init__(self, data_dir, USE_PUBLIC_DATASET):
        self.USE_PUBLIC_DATASET = USE_PUBLIC_DATASET
        self.data_dir = data_dir
        self.mask_file = 'WIDE12H_bin2_2arcmin_mask.npy'
        self.viz_label_file = 'label.npy'
        if self.USE_PUBLIC_DATASET:
            self.kappa_file = 'WIDE12H_bin2_2arcmin_kappa.npy'
            self.label_file = self.viz_label_file
            self.Ncosmo = 101  # Number of cosmologies in the entire training data
            self.Nsys = 256    # Number of systematic realizations in the entire training data
            self.test_kappa_file = 'WIDE12H_bin2_2arcmin_kappa_noisy_test.npy'
            self.Ntest = 4000  # Number of instances in the test data
        else:
            self.kappa_file = 'sampled_WIDE12H_bin2_2arcmin_kappa.npy'
            self.label_file = 'sampled_label.npy'
            self.Ncosmo = 3    # Number of cosmologies in the sampled training data
            self.Nsys = 30     # Number of systematic realizations in the sampled training data
            self.test_kappa_file = 'sampled_WIDE12H_bin2_2arcmin_kappa_noisy_test.npy'
            self.Ntest = 3     # Number of instances in the sampled test data
        
        self.shape = [1424, 176]  # dimensions of each map
        self.pixelsize_arcmin = 2  # pixel size in arcmin
        self.pixelsize_radian = self.pixelsize_arcmin / 60 / 180 * np.pi  # pixel size in radian
        self.ng = 30  # galaxy number density. This determines the noise level of the experiment.

    def load_train_data(self):
        # Load smaller files (mask, label, viz_label) without chunking
        self.mask = Utility.load_np(data_dir=self.data_dir, file_name=self.mask_file)
        self.label = Utility.load_np(data_dir=self.data_dir, file_name=self.label_file)
        self.viz_label = Utility.load_np(data_dir=self.data_dir, file_name=self.viz_label_file)

        # Initialize kappa array
        self.kappa = np.zeros((self.Ncosmo, self.Nsys, *self.shape), dtype=np.float16)

        # Load kappa file in chunks
        file_path = os.path.join(self.data_dir, self.kappa_file)
        chunk_size = 10  # Number of cosmologies to process per chunk (adjust based on memory)
        try:
            kappa_data = np.load(file_path, mmap_mode='r')  # Load as memory-mapped array
            for i in tqdm(range(0, self.Ncosmo, chunk_size), desc=f"Loading {self.kappa_file}"):
                chunk_end = min(i + chunk_size, self.Ncosmo)
                chunk = kappa_data[i:chunk_end]  # Shape: (chunk_size, Nsys, N_unmasked)
                self.kappa[i:chunk_end, :, self.mask] = chunk  # Assign directly to masked pixels
        finally:
            if 'kappa_data' in locals():
                del kappa_data  # Release file handle

    def load_test_data(self):
        # Initialize kappa_test array
        self.kappa_test = np.zeros((self.Ntest, *self.shape), dtype=np.float16)

        # Load test kappa file in chunks
        file_path = os.path.join(self.data_dir, self.test_kappa_file)
        chunk_size = 100  # Number of test instances to process per chunk (adjust based on memory)
        try:
            kappa_test_data = np.load(file_path, mmap_mode='r')  # Load as memory-mapped array
            for i in tqdm(range(0, self.Ntest, chunk_size), desc=f"Loading {self.test_kappa_file}"):
                chunk_end = min(i + chunk_size, self.Ntest)
                chunk = kappa_test_data[i:chunk_end]  # Shape: (chunk_size, N_unmasked)
                self.kappa_test[i:chunk_end, self.mask] = chunk  # Assign directly to masked pixels
        finally:
            if 'kappa_test_data' in locals():
                del kappa_test_data  # Release file handle


class Visualization:
    
    @staticmethod
    def plot_mask(mask):
        plt.figure(figsize=(30,100))
        plt.imshow(mask.T)
        plt.show()

    @staticmethod
    def plot_noiseless_training_convergence_map(kappa):
        plt.figure(figsize=(30,100))
        plt.imshow(kappa[0,0].T, vmin=-0.02, vmax=0.07)
        plt.show()

    @staticmethod
    def plot_noisy_training_convergence_map(kappa, mask, pixelsize_arcmin, ng):
        plt.figure(figsize=(30,100))
        plt.imshow(Utility.add_noise(kappa[0,0], mask, ng, pixelsize_arcmin).T, vmin=-0.02, vmax=0.07)
        plt.show()

    @staticmethod
    def plot_cosmological_parameters_OmegaM_S8(label):
        plt.scatter(label[:,0,0], label[:,0,1])
        plt.xlabel(r'$\Omega_m$')
        plt.ylabel(r'$S_8$')
        plt.show()

    @staticmethod
    def plot_baryonic_physics_parameters(label):
        plt.scatter(label[0,:,2], label[0,:,3])
        plt.xlabel(r'$T_{\mathrm{AGN}}$')
        plt.ylabel(r'$f_0$')
        plt.show()

    @staticmethod
    def plot_photometric_redshift_uncertainty_parameters(label):
        plt.hist(label[0,:,4], bins=20)
        plt.xlabel(r'$\Delta z$')
        plt.show()


class Score:
    @staticmethod
    def _score_phase1(true_cosmo, infer_cosmo, errorbar):
        """
        Computes the log-likelihood score for Phase 1 based on predicted cosmological parameters.

        Parameters
        ----------
        true_cosmo : np.ndarray
            Array of true cosmological parameters (shape: [n_samples, n_params]).
        infer_cosmo : np.ndarray
            Array of inferred cosmological parameters from the model (same shape as true_cosmo).
        errorbar : np.ndarray
            Array of standard deviations (uncertainties) for each inferred parameter 
            (same shape as true_cosmo).

        Returns
        -------
        np.ndarray
            Array of scores for each sample (shape: [n_samples]).
        """
        
        sq_error = (true_cosmo - infer_cosmo)**2
        scale_factor = 1000  # This is a constant that scales the error term.
        score = - np.sum(sq_error / errorbar**2 + np.log(errorbar**2) + scale_factor * sq_error, 1)
        score = np.mean(score)
        if score >= -10**6: # Set a minimum of the score (to properly display on Codabench)
            return score
        else:
            return -10**6
        


def KL_div_posterior_loss(pred_means, pred_sigmas, truths):
    """
    A KL divergence loss function that directly optimizes the score function
    
    Inputs:
    - pred_means:   2D tensor (batch_size, 2)
    - pred_sigmas:  2D tensor (batch_size, 2) 
    - truths:       2D tensor (batch_size, 2)
    """
    
    residuals_sq = (pred_means - truths)**2  
    
    loss_terms = residuals_sq / (pred_sigmas**2)
    loss_sum = torch.sum(loss_terms, dim=1)
    
    log_sigma_terms = torch.sum(torch.log(pred_sigmas**2), dim=1)


    loss = torch.mean(loss_sum + log_sigma_terms)
    
    return loss





def train_epoch(model, dataloader, loss_fn, optimizer, device):
    """Trains the model for one epoch."""
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, total=len(dataloader), desc="Training")
    for X, y in pbar:
        X, y = X.to(device), y.to(device)
        # Forward pass
        pred_means, pred_sigmas= model(X)

        loss = loss_fn(pred_means, pred_sigmas, y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def validate_epoch(model, dataloader, loss_fn, device):
    """Validates the model on the validation/test set."""
    model.eval()
    total_loss = 0
    pbar = tqdm(dataloader, total=len(dataloader), desc="Validating")
    with torch.no_grad():
        for X, y in pbar:
            X, y = X.to(device), y.to(device)
            pred_means, pred_sigmas = model(X)
            total_loss += loss_fn(pred_means, pred_sigmas, y).item()
            
    return total_loss / len(dataloader)



import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class CosmologyDataset(Dataset):
    """
    Custom PyTorch Dataset with optional data augmentation
    """
    
    def __init__(self, data, labels=None, transform=None, label_transform=None, augment=False):
        """
        Args:
            data: NumPy array or similar, containing image-like data (e.g., shape [N, H, W] or [N, C, H, W])
            labels: Optional labels for the data
            transform: Optional transform to apply to the data
            label_transform: Optional transform to apply to the labels
            augment: Boolean to enable/disable data augmentation (random flips and crops)
        """
        self.data = data
        self.labels = labels
        self.transform = transform
        self.label_transform = label_transform
        self.augment = augment
        
        # Define augmentation pipeline if augment is True
        if self.augment:
            # Compose augmentation transforms
            augmentation_transforms = [
                transforms.RandomHorizontalFlip(p=0.5),  # 50% chance of horizontal flip
                transforms.RandomVerticalFlip(p=0.5),    # 50% chance of vertical flip
            ]
            # Combine user-provided transform with augmentations
            if self.transform:
                self.transform = transforms.Compose(augmentation_transforms + [self.transform])
            else:
                self.transform = transforms.Compose(augmentation_transforms)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx].astype(np.float32)   # Convert from float16 to float32
        if self.transform:
            image = self.transform(image) 
        if self.labels is not None:
            label = self.labels[idx].astype(np.float32)
            label = torch.from_numpy(label)
            if self.label_transform:
                label = self.label_transform(label)
            return image, label
        else:
            return image


# class CosmologyDataset(Dataset):
#     """
#     Custom PyTorch Dataset
#     """
    
#     def __init__(self, data, labels=None,
#                  transform=None,
#                  label_transform=None):
#         self.data = data
#         self.labels = labels
#         self.transform = transform
#         self.label_transform = label_transform

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         image = self.data[idx].astype(np.float32)   # Convert from float16 to float32
#         if self.transform:
#             image = self.transform(image) 
#         if self.labels is not None:
#             label = self.labels[idx].astype(np.float32)
#             label = torch.from_numpy(label)
#             if self.label_transform:
#                 label = self.label_transform(label)
#             return image, label
#         else:
#             return image
        

class Config:
    IMG_HEIGHT = 1424
    IMG_WIDTH = 176
    
    # Parameters to predict (Omega_m, S_8, sigma_Omega_m, sigma_S_8)
    NUM_TARGETS = 4

    # Training hyperparameters
    BATCH_SIZE = 128
    EPOCHS = 25
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 1e-4   # L2 regularization to prevent overfitting
    IMG_RESIZE = 256  # Resize images to 256x256 for ViT
    DEVICE = "mps" if torch.has_mps else "cpu"
    MODEL_SAVE_PATH = None  # Will be set dynamically with timestamp

def save_config_and_lr(config: Config, optimizer: optim.Optimizer, config_file: str):
    """
    Save the Config object and the current learning rate of the optimizer.

    Args:
        config: Config object containing training hyperparameters.
        optimizer: PyTorch optimizer with current learning rate.
        config_file: Path to save the config and learning rate.
    """
    # Get current learning rate from optimizer
    current_lr = optimizer.param_groups[0]['lr']
    
    # Save config attributes and learning rate
    config_dict = {
        'IMG_HEIGHT': config.IMG_HEIGHT,
        'IMG_WIDTH': config.IMG_WIDTH,
        'NUM_TARGETS': config.NUM_TARGETS,
        'BATCH_SIZE': config.BATCH_SIZE,
        'EPOCHS': config.EPOCHS,
        'LEARNING_RATE': config.LEARNING_RATE,
        'WEIGHT_DECAY': config.WEIGHT_DECAY,
        'DEVICE': config.DEVICE,
        'MODEL_SAVE_PATH': config.MODEL_SAVE_PATH,
        'IMG_RESIZE': config.IMG_RESIZE,
        'current_lr': current_lr
    }
    
    os.makedirs(os.path.dirname(config_file) or '.', exist_ok=True)
    with open(config_file, 'wb') as f:
        pickle.dump(config_dict, f)
    print(f"Saved config and learning rate to {config_file}")

def load_config_and_lr(config_file: str) -> Tuple[Config, float]:
    """
    Load the Config object and the last learning rate for continued training.

    Args:
        config_file: Path to the saved config and learning rate.

    Returns:
        Tuple of (Config object, last learning rate).
    """
    with open(config_file, 'rb') as f:
        config_dict = pickle.load(f)
    
    # Reconstruct Config object
    config = Config()
    config.IMG_HEIGHT = config_dict['IMG_HEIGHT']
    config.IMG_WIDTH = config_dict['IMG_WIDTH']
    config.NUM_TARGETS = config_dict['NUM_TARGETS']
    config.BATCH_SIZE = config_dict['BATCH_SIZE']
    config.EPOCHS = config_dict['EPOCHS']
    config.LEARNING_RATE = config_dict['LEARNING_RATE']
    config.WEIGHT_DECAY = config_dict['WEIGHT_DECAY']
    config.DEVICE = config_dict['DEVICE']
    config.MODEL_SAVE_PATH = config_dict['MODEL_SAVE_PATH']
    config.IMG_RESIZE = config_dict['IMG_RESIZE']
    current_lr = config_dict['current_lr']
    
    print(f"Loaded config from {config_file}")
    print(f"Loaded learning rate: {current_lr}")
    return config, current_lr
