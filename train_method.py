import wandb
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
import wandb 
from tqdm import tqdm
import torch
import os
from utilis import Utility, CosmologyDataset,  save_config_and_lr
from chunk_handle import get_chunk_indices, iter_chunks, compute_global_image_stats, compute_global_label_stats, save_transform_and_scaler, load_transform_and_scaler
class Trainer():

    @staticmethod
    def train_epoch(model, train_loader, criterion, optimizer, device, CheckCosmo=False):
        model.train()
        total_loss = 0.0
        # pbar = tqdm(train_loader, total=len(train_loader), desc="Training")
        for (images, specs), targets in train_loader:
            images = images.to(device, dtype=torch.float32)
            specs = specs.to(device, dtype=torch.float32)
            targets = targets.to(device, dtype=torch.float32)
            optimizer.zero_grad()
            outputs = model((images, specs))

            if CheckCosmo:
                print(f"input shape: {images.shape}, target shape: {targets.shape}")
                print(f"num of unique cosmo omega:{len(targets[:,0].unique())}")
                print(f"num of unique cosmo sigma:{len(targets[:,1].unique())}")

            loss = criterion(outputs, targets)  # Gọi hàm mất mát
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        return total_loss / len(train_loader)
    
    @staticmethod
    def validate_epoch(model, dataloader, loss_fn, device):
        """Validates the model on the validation/test set."""
        model.eval()
        total_loss = 0
        # pbar = tqdm(dataloader, total=len(dataloader), desc="Validating")
        with torch.no_grad():
            for (images, specs), targets in dataloader:
                images = images.to(device, dtype=torch.float32)
                specs = specs.to(device, dtype=torch.float32)
                targets = targets.to(device, dtype=torch.float32)
                outputs = model((images, specs))
                total_loss += loss_fn(outputs, targets).item()
        return total_loss / len(dataloader)
    

    @staticmethod # Import Weights & Biases
    def Train_chunk(model: nn.Module, 
                        config,
                        optimizer: optim.Optimizer, 
                        criterion: nn.Module, 
                        chunk_dir: str,
                        val_size: float = 0.2,
                        device: str = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu',
                        verbose: bool = False,
                        log_file: str = None,
                        config_file: str = None,
                        transform: Optional[transforms.Compose] = None,
                        label_scaler: Optional[Any] = None,
                        save_log:bool=False) -> Dict[str, Any]:
        
        
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        # Set file paths using the fixed timestamp
        model_dir = f"./side_module_ViT/model_{timestamp}"
        if log_file is None:
            log_file = f"{model_dir}/training_log.txt"
        if config_file is None:
            config_file = f"{model_dir}/training_config.pkl"
        config.MODEL_SAVE_PATH = f"{model_dir}/best_model.pth"
        if save_log:
            os.makedirs(os.path.dirname(log_file) or '.', exist_ok=True)
            with open(log_file, 'w') as f:
                f.write("Training Log\n")
                f.write(f"Started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Config: EPOCHS={config.EPOCHS}, BATCH_SIZE={config.BATCH_SIZE}, LEARNING_RATE={config.LEARNING_RATE}, DEVICE={config.DEVICE}, MODEL_SAVE_PATH={config.MODEL_SAVE_PATH}\n\n")

        device = config.DEVICE
        model.to(device)
        model.train()
        # total_val_datasets = []
        total_y_vals=[]
        total_X_vals=[]
        total_spect_vals=[]
        
        indices = get_chunk_indices(chunk_dir)
        train_losses = []  # List of lists: losses per epoch per chunk
        val_losses = []  # List of validation losses per epoch  
        total_samples = 0
        
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        scheduler =torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.EPOCHS, eta_min=0)
        best_val_loss = float('inf')

        for epoch in tqdm(range(config.EPOCHS), desc="Epochs", leave=True):

            if save_log:
                with open(log_file, 'a') as f:
                    f.write(f"Epoch {epoch+1}/{config.EPOCHS}\n")


            epoch_train_losses = []
            
            for chunk_idx in tqdm(indices, desc="Chunks", leave=False):
                if verbose:
                    print(f"Processing chunk {chunk_idx}...")
                if save_log:
                    with open(log_file, 'a') as f:
                        f.write(f"  Processing chunk {chunk_idx}...\n")
                
                #load data and spectrum for the chunk
                noisy_chunk, label_chunk, _ = next(iter_chunks(chunk_dir, indices=[chunk_idx]))
                Spect = Utility.load_np(data_dir="./dataset/power_scpectrum", file_name=f"Spec_chunk_{chunk_idx}.npy")  # shape = (Ncosmo, Nsys, 10)
                
                # Split into train/test (stratified if labels are categorical; here simple split)
                Nsys = noisy_chunk.shape[1]
                Ncosmo = noisy_chunk.shape[0]

                NP_idx = np.arange(Nsys)   
                shape = noisy_chunk.shape[2:]

                seed = 113
                train_NP_idx, val_NP_idx = train_test_split(NP_idx, test_size=val_size, random_state=seed)

                noisy_kappa_train = noisy_chunk[:, train_NP_idx]      # shape = (Ncosmo, len(train_NP_idx), H, W)
                label_train = label_chunk[:, train_NP_idx]         # shape = (Ncosmo, len(train_NP_idx), 5)
                noisy_kappa_val = noisy_chunk[:, val_NP_idx]          # shape = (Ncosmo, len(val_NP_idx), H, W)
                label_val = label_chunk[:, val_NP_idx]             # shape = (Ncosmo, len(val_NP_idx), 5)
                Spect_train = Spect[:, train_NP_idx]  # (Ncosmo, len(train_NP_idx), 10)
                Spect_val = Spect[:, val_NP_idx]  # (Ncosmo, len(val_NP_idx), 10)

                Ntrain = label_train.shape[0] * label_train.shape[1]
                Nval = label_val.shape[0] * label_val.shape[1]
                if verbose:
                    print(f'Shape of the split training data = {noisy_kappa_train.shape}')
                    print(f'Shape of the split validation data = {noisy_kappa_val.shape}')
                    print(f'Shape of the split training labels = {label_train.shape}')
                    print(f'Shape of the split validation labels = {label_val.shape}')
                    print(f'Shape of the split training Spect = {Spect_train.shape}')
                    print(f'Shape of the split validation Spect = {Spect_val.shape}')

                # Reshape data
                transposed_train = noisy_kappa_train.transpose(1, 0, *range(2, len(noisy_kappa_train.shape)))
                X_train = transposed_train.reshape(Ntrain, *noisy_kappa_train.shape[2:])
                transposed_val = noisy_kappa_val.transpose(1, 0, *range(2, len(noisy_kappa_val.shape)))
                X_val = transposed_val.reshape(Nval, *noisy_kappa_val.shape[2:])
                # Reshape labels
                transposed_label_train = label_train.transpose(1, 0, 2)  # (len(train_NP_idx), Ncosmo, 5)
                y_train = transposed_label_train.reshape(Ntrain, label_train.shape[2])[:, :2]  # (Ntrain, 2)
                transposed_label_val = label_val.transpose(1, 0, 2)  # (len(val_NP_idx), Ncosmo, 5)
                y_val = transposed_label_val.reshape(Nval, label_val.shape[2])[:, :2]  # (Nval, 2)
                # Reshape specs
                transposed_spect_train = Spect_train.transpose(1, 0, 2)  # (len(train_NP_idx), Ncosmo, 10)
                spec_train = transposed_spect_train.reshape(Ntrain, 10)  # (Ntrain, 10)
                transposed_spect_val = Spect_val.transpose(1, 0, 2)  # (len(val_NP_idx), Ncosmo, 10)
                spec_val = transposed_spect_val.reshape(Nval, 10)  # (Nval, 10)




                train_dataset = CosmologyDataset(data=X_train, specs=spec_train, labels=y_train, transform=transform, label_transform=label_scaler)
                val_dataset = CosmologyDataset(data=X_val, specs=spec_val, labels=y_val, transform=transform, label_transform=label_scaler)
                if epoch == 0:
                    total_y_vals.append(y_val)
                    total_X_vals.append(X_val)
                    total_spect_vals.append(spec_val)
                    # total_val_datasets.append(val_dataset)
                train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
 
                train_loss = Trainer.train_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
                epoch_train_losses.append(train_loss)
                wandb.log({"chunk": epoch*len(indices)+chunk_idx, "train_loss": train_loss})

            train_losses.append(epoch_train_losses)

            # Validate once per epoch using concatenated validation dataset
            if epoch == 0:
                total_spect_vals.append(spec_val)
                all_val_data = np.concatenate([X_val for X_val in total_X_vals], axis=0)
                all_val_specs = np.concatenate([spect for spect in total_spect_vals], axis=0)
                all_val_labels = np.concatenate([y_val for y_val in total_y_vals], axis=0)
                concatenated_val_dataset = CosmologyDataset(
                    data=all_val_data,
                    specs=all_val_specs,
                    labels=all_val_labels,
                    transform=transform,
                    label_transform=label_scaler
                )
                val_loader = DataLoader(
                    concatenated_val_dataset,
                    batch_size=config.BATCH_SIZE,
                    shuffle=False
                )

            val_loss = Trainer.validate_epoch(model, val_loader, criterion, config.DEVICE)
            val_losses.append(val_loss) 
            scheduler.step()

            wandb.log({
                "epoch": epoch + 1,
                "val_loss": val_loss,
                "avg_train_loss": np.mean(epoch_train_losses),
                "learning_rate": optimizer.param_groups[0]['lr']
            })
            
            if save_log:
                with open(log_file, 'a') as f:
                    f.write(f"Epoch {epoch+1}/{config.EPOCHS} | Avg Train Loss: {np.mean(epoch_train_losses):.6f} | Val Loss: {val_loss:.6f} | Learning Rate: {current_lr:.6f}\n")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if config.MODEL_SAVE_PATH is not None:
                    os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH) or '.', exist_ok=True)
                torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
                if save_log:
                    with open(log_file, 'a') as f:
                        f.write(f"  -> New best model saved to {config.MODEL_SAVE_PATH} (Val Loss: {val_loss:.6f})\n")
                wandb.save(config.MODEL_SAVE_PATH)

        save_config_and_lr(config, optimizer, config_file)




        model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, weights_only=True))

        wandb.finish()
        return {
            'train_losses': train_losses,
            'total_val_datasets': concatenated_val_dataset,
            'total_epochs': config.EPOCHS,
            'total_samples': total_samples,
            'val_losses': val_losses    
        }


    @staticmethod
    def train_pipeline(model, config, pretrain=False, previous_path= None):
        transform, label_scaler = load_transform_and_scaler(transform_file='./side_module/transform_params.pkl', scaler_file='./side_module/label_scaler.pkl')

        if pretrain:
            if previous_path is None:
                raise ValueError("Please specify previous_timestamp for pretraining (e.g., '20250918_144500').")
            last_lr=1e-4
            model.load_state_dict(torch.load(f"{previous_path}/best_model.pth", weights_only=True))
            optimizer = optim.Adam(model.parameters(), lr=last_lr, weight_decay=config.WEIGHT_DECAY)
        else:
            optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

        criterion = nn.MSELoss()
        run_name = f"{model.name}_run_{time.strftime('%Y_%m_%d_%H_%M_%S')}"
        wandb.init(
            entity="huy_ben",
            project="cosmology_training",
                # project name (string, not f-string with stray bracket)
            name=run_name,   
            config={
                "IMG_HEIGHT": config.IMG_HEIGHT,
                "IMG_WIDTH": config.IMG_WIDTH,
                "NUM_TARGETS": config.NUM_TARGETS,
                "BATCH_SIZE": config.BATCH_SIZE,
                "EPOCHS": config.EPOCHS,
                "LEARNING_RATE": config.LEARNING_RATE,
                "WEIGHT_DECAY": config.WEIGHT_DECAY,
                "DEVICE": config.DEVICE,
                # "IMG_RESIZE": config.IMG_RESIZE,
                "split_ratio": 0.8,
                "epochs_per_chunk": 1
        })
        # Continue training


        history = Trainer.Train_chunk(
            model=model,
            config=config,
            optimizer=optimizer,
            criterion=criterion,
            chunk_dir='./dataset/chunk_kappa_noise_new',
            val_size=0.2,
            transform=transform,
            label_scaler=label_scaler,
            verbose=False
        )
        return history