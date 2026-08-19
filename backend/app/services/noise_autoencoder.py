import torch
import torch.nn as nn

class SensorNoiseAutoencoder(nn.Module):
    def __init__(self):
        super(SensorNoiseAutoencoder, self).__init__()
        
        # Encoder: Compress the 64x64 noise patch
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1), # 32x32
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # 16x16
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 8x8
            nn.ReLU()
        )
        
        # Decoder: Attempt to reconstruct the noise
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), # 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), # 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1), # 64x64
            nn.Sigmoid()
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

def calculate_reconstruction_error(model, patch_tensor):
    """
    Pass a 64x64 grayscale high-frequency patch into the model.
    Returns the MSE anomaly score.
    """
    model.eval()
    with torch.no_grad():
        reconstructed = model(patch_tensor)
        mse_loss = nn.functional.mse_loss(reconstructed, patch_tensor)
    return mse_loss.item()