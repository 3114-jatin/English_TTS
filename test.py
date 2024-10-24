import torch

# Load the HiFi-GAN model
hifigan_model = torch.load(hifigan_model_path)
hifigan_model.eval()  # Set the model to evaluation mode
