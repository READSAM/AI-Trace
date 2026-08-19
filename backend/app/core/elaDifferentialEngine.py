import cv2
import numpy as np
from typing import Tuple

class ELADifferentialEngine:
    """
    Error Level Analysis (ELA) Differential Engine.
    Examines localized compression error differentials caused by re-encoding 
    at a target JPEG scale to highlight non-uniform compression regions.
    """

    @staticmethod
    def analyze(
        image_np: np.ndarray, 
        quality_scale: int = 90, 
        scale_multiplier: float = 15.0
    ) -> Tuple[float, np.ndarray]:
        # 1. Re-encode input image to JPEG at designated quality scale in-memory
        success, encoded_buf = cv2.imencode(".jpg", image_np, [int(cv2.IMWRITE_JPEG_QUALITY), quality_scale])
        if not success:
            raise ValueError("Failed to perform JPEG re-encoding for ELA analysis.")
        
        resaved_np = cv2.imdecode(encoded_buf, cv2.IMREAD_COLOR)

        # 2. Calculate Absolute Pixel Difference Matrix: |I_orig - I_resaved|
        diff = cv2.absdiff(image_np, resaved_np).astype(np.float32)

        # 3. Amplify error matrix for visual & quantitative evaluation
        diff_scaled = np.clip(diff * scale_multiplier, 0, 255).astype(np.uint8)

        # 4. Statistical anomaly evaluation: High variance across local blocks indicates splicing/editing
        ela_gray = cv2.cvtColor(diff_scaled, cv2.COLOR_BGR2GRAY)
        mean_energy = float(np.mean(ela_gray))
        std_dev_energy = float(np.std(ela_gray))

        # Authentic images show uniform low-level ELA noise; manipulated/generative show local spikes
        discrepancy_score = float(np.clip((std_dev_energy / 64.0) * (mean_energy / 32.0), 0.0, 1.0))

        # 5. Generate Enhanced ELA Visual Heatmap
        ela_heatmap = cv2.applyColorMap(ela_gray, cv2.COLORMAP_HOT)

        return discrepancy_score, ela_heatmap
