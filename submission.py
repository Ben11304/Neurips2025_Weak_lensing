import torch
from train_method import Trainer
from Models.Models import Spectrum_CNN, Simple_CNN, Spectrum_CNNv2
from Models.ViT import CustomViT
from inference import Inference

class Config:
    IMG_HEIGHT = 1424
    IMG_WIDTH = 176

    # Parameters to predict (Omega_m, S_8, sigma_Omega_m, sigma_S_8)
    NUM_TARGETS = 2
    # Training hyperparameters
    BATCH_SIZE = 101
    EPOCHS = 15
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4 
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
    elif getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        DEVICE = torch.device('mps')
    else:
        DEVICE = torch.device('cpu')
    MODEL_SAVE_PATH = None 


def main():
    config = Config()
    # Choose model implementation (use the v2 variant here)
    model= Spectrum_CNN(config.IMG_HEIGHT, config.IMG_WIDTH, config.NUM_TARGETS)
    # model = Spectrum_CNNv2(config.IMG_HEIGHT, config.IMG_WIDTH, config.NUM_TARGETS)
    image_size = (config.IMG_HEIGHT, config.IMG_WIDTH)
    checkpoint="./side_module_ViT/model_20250921_015923_on/best_model.pth"
    # Load checkpoint onto the selected device and move model to that device
    # Load checkpoint to CPU first (more robust across devices like MPS), then move model to target device
    state = torch.load(checkpoint, map_location='cpu')
    # If the checkpoint contains a dict with other keys (e.g., optimizer), handle accordingly
    if isinstance(state, dict) and 'state_dict' in state:
        state_dict = state['state_dict']
    else:
        state_dict = state

    model.load_state_dict(state_dict)
    model.to(config.DEVICE)
    model.eval()
    print("Model loaded and moved to device:", config.DEVICE)

    Inference.inference(model, config)



if __name__ == "__main__":
    main()
