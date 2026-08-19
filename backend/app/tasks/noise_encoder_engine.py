import torch
import cv2
import numpy as np
from app.services.noise_autoencoder import SensorNoiseAutoencoder 

class NoiseForensicsEngine:
    def __init__(self, weights_path="app/core/sensor_noise_autoencoder.pth"):
        # 1. Force loading onto CPU for Docker stability
        self.device = torch.device("cpu")
        
        # 2. Initialize the blank architecture
        self.model = SensorNoiseAutoencoder()
        
        # 3. Load the weights you trained on Kaggle
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        
        # 4. Set to evaluation mode (disables dropout and gradients)
        self.model.eval()
        self.model.to(self.device)
        
        # High-Pass Filter for inference
        self.hpf_kernel = np.array([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ], dtype=np.float32)

    # UPDATED: Now accepts the full image and patch_size
    def analyze_image_grid(self, image: np.ndarray, patch_size: int = 64) -> np.ndarray:
        """
        Slices the full image, batches it for a single PyTorch forward pass,
        and returns a 2D grid of MSE reconstruction errors.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        rows = (h - patch_size + 1) // patch_size
        cols = (w - patch_size + 1) // patch_size
        
        patches = []
        
        # 1. Extract all patches from the image
        for r in range(rows):
            for c in range(cols):
                y = r * patch_size
                x = c * patch_size
                patch = gray[y:y+patch_size, x:x+patch_size]
                
                # Apply High-Pass Filter and Normalize
                noise = cv2.filter2D(patch, cv2.CV_32F, self.hpf_kernel)
                noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
                patches.append(noise)
                
        # 2. Vectorize: Stack into a single batch tensor (Batch_Size, 1, 64, 64)
        batch_tensor = torch.from_numpy(np.stack(patches)).unsqueeze(1).float().to(self.device)
        
        # 3. Single Forward Pass (Processes all patches at once)
        with torch.no_grad():
            reconstructed = self.model(batch_tensor)
            
            # 4. Compute MSE per patch (Average over Channels, Height, Width)
            mse_loss_per_patch = torch.mean((reconstructed - batch_tensor)**2, dim=[1, 2, 3])
            
        # 5. Reshape the 1D tensor back into the 2D grid for the C++ graph segmenter
        mse_grid = mse_loss_per_patch.numpy().reshape((rows, cols))
        
        return mse_grid