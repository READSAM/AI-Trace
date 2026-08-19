
import numpy as np

import graph_segmenter_cpp
from app.tasks.noise_encoder_engine import NoiseForensicsEngine


class ComplexSceneForensics:
    def __init__(self):
        # Settings for Local Binary Pattern (LBP) texture extraction
        self.lbp_radius = 1
        self.lbp_points = 8 * self.lbp_radius
        self.noise_engine = NoiseForensicsEngine()

    def map_spatial_anomalies(self, image, patch_size=64, anomaly_threshold=3.5):

        #replacing the Laplacian calculation
        mse_grid = self.noise_engine.analyze_image_grid(image, patch_size)

        global_mse=float(np.mean(mse_grid))

        EXPECTED_AI_MSE_SPIKE = 0.008

        synthetic_risk_score = float(np.clip(global_mse / EXPECTED_AI_MSE_SPIKE, 0.0, 1.0))
         
        std_dev = np.std(mse_grid)

        MIN_STD_DEV_FLOOR = 0.02 
        safe_std_dev = max(std_dev, MIN_STD_DEV_FLOOR)

        z_scores = (mse_grid - np.mean(mse_grid)) / safe_std_dev
            
        # Execute the C++ BFS traversal
        # This runs at near-native C++ speeds
        components = graph_segmenter_cpp.find_forged_components(z_scores, anomaly_threshold)
        
        structural_splices = [c for c in components if c["area"] >=4]
        
        spatial_splice_score = float(np.clip(len(structural_splices) / 25.0, 0.0, 1.0))

        return components, spatial_splice_score,synthetic_risk_score
    
scene_forensics = ComplexSceneForensics()