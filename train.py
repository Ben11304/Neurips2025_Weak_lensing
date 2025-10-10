import torch
from train_method import Trainer
from Models.Models import Spectrum_CNN, Simple_CNN, Spectrum_CNNv2
from Models.ViT import CustomViT


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
    model = Spectrum_CNN(config.IMG_HEIGHT, config.IMG_WIDTH, config.NUM_TARGETS)
    model = Spectrum_CNNv2(config.IMG_HEIGHT, config.IMG_WIDTH, config.NUM_TARGETS)
    image_size = (config.IMG_HEIGHT, config.IMG_WIDTH)
    # model = CustomViT(image_size= image_size,
    #                     patch_size=(16, 16),
    #                     dim=1024,
    #                     depth=4,          
    #                     heads=12,       
    #                     mlp_dim=2048,       
    #                     channels=1,         
    #                     dim_head=64,
    #                     dropout=0.1,
    #                     emb_dropout=0.1,
    #                     pool='cls',        
    #                     spec_dim=10,        
    #                     output_dim=2          
    #                 )
    # Run training pipeline
    history = Trainer.train_pipeline(model, config)



if __name__ == "__main__":
    main()
