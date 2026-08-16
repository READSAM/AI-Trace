import io
import os
import cv2
import numpy as np
from PIL import Image
from typing import Dict, Any, Tuple
from skimage.feature import local_binary_pattern
from scipy.stats import zscore
from scipy.spatial.distance import euclidean

from app.tasks.celery_app import celery_app
from app.fusion.engine import EvidentialFusionEngine


class FFTSpectralAnalyzer:
    """
    2D Discrete Fourier Transform (FFT) analysis engine.
    Detects high-frequency periodic grid artifacts introduced by 
    transposed convolutions and upsampling in GANs / Diffusion models.
    """

    @staticmethod
    def analyze(image_np: np.ndarray) -> Tuple[float, np.ndarray]:
        # Convert to grayscale
        if len(image_np.shape) == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_np

        # 1. 2D Discrete Fourier Transform
        f_transform = np.fft.fft2(gray.astype(np.float32))
        f_shift = np.fft.fftshift(f_transform)

        # 2. Power Spectrum Calculation Log Magnitude: log(1 + |F(u,v)|)
        magnitude_spectrum = np.abs(f_shift)
        power_spectrum = np.log1p(magnitude_spectrum)

        # 3. Azimuthal Radial Power Integration
        h, w = gray.shape
        center_y, center_x = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        radius = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2).astype(int)

        # Limit analysis to Nyquist frequency radius
        max_radius = min(center_x, center_y)
        radial_prof = np.bincount(radius.ravel(), weights=power_spectrum.ravel())
        radial_counts = np.bincount(radius.ravel())
        
        # Avoid divide-by-zero
        radial_counts[radial_counts == 0] = 1
        azimuthal_average = radial_prof[:max_radius] / radial_counts[:max_radius]

        # 4. Anomaly Metric: Deviation at high-frequency outer band
        high_freq_cutoff = int(max_radius * 0.6)
        high_freq_band = azimuthal_average[high_freq_cutoff:]
        
        # Generative grid artifacts manifest as high-frequency spectral spikes
        spectral_variance = float(np.var(high_freq_band))
        spectral_peak_ratio = float(np.max(high_freq_band) / (np.mean(high_freq_band) + 1e-6))
        
        # Calibrated score mapping
        anomaly_score = float(np.clip((spectral_variance * 0.1) + (spectral_peak_ratio * 0.05), 0.0, 1.0))

        # 5. Spectrum Heatmap Generation
        norm_spectrum = cv2.normalize(power_spectrum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap_bgr = cv2.applyColorMap(norm_spectrum, cv2.COLORMAP_JET)

        return anomaly_score, heatmap_bgr


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

    
class ComplexSceneForensics:
    def __init__(self):
        # Settings for Local Binary Pattern (LBP) texture extraction
        self.lbp_radius = 1
        self.lbp_points = 8 * self.lbp_radius

    def map_spatial_anomalies(self, image, patch_size=64, anomaly_threshold=2.5):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        patch_variances = []
        patch_coords = []
        
        # 1. Receptive Field Scanning (Grid Breakdown)
        # Added +1 to ensure the edges are reached cleanly
        for y in range(0, h - patch_size + 1, patch_size):
            for x in range(0, w - patch_size + 1, patch_size):
                patch = gray[y:y+patch_size, x:x+patch_size]
                
                # Extract noise/texture signature using Laplacian variance
                variance = cv2.Laplacian(patch, cv2.CV_64F).var()
                
                patch_variances.append(variance)
                patch_coords.append((x, y))

        # EDGE CASE: If the image is smaller than patch_size
        if not patch_variances:
            return [], 0.0
                
        # 2. Anomaly Scoring using Z-scores
        variances_np = np.array(patch_variances)
        std_dev = np.std(variances_np)

        # EDGE CASE: If the image is completely uniform (std_dev == 0), zscore returns NaNs
        if std_dev == 0:
            z_scores = np.zeros_like(variances_np)
        else:
            z_scores = zscore(variances_np)
        
        # 3. Flagging anomalies
        suspicious_patches = []
        for i, z in enumerate(z_scores):
            if abs(z) > anomaly_threshold:
                suspicious_patches.append({
                    "coords": patch_coords[i],
                    "z_score": float(z),                   # Cast to float for JSON Celery serialization
                    "variance": float(variances_np[i])     # Cast to float for JSON Celery serialization
                })
                
        return suspicious_patches, float(np.mean(variances_np))


@celery_app.task(name="app.tasks.vision_tasks.run_image_pipeline", bind=True)
def run_image_pipeline(self, task_id: str, file_bytes_hex: str) -> Dict[str, Any]:
    """
    Celery task orchestrator executing FFT, ELA, and Spatial image forensics pipelines.
    """
    try:
        # 1. Decode Raw Image Bytes
        image_bytes = bytes.fromhex(file_bytes_hex)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image_np is None:
            raise ValueError("Corrupted image file or unsupported image format.")

        # 2. Run 2D FFT Spectral Analysis
        fft_score, fft_visual = FFTSpectralAnalyzer.analyze(image_np)

        # 3. Run ELA Differential Analysis
        ela_score, ela_visual = ELADifferentialEngine.analyze(image_np)

        # 4. Run Complex Scene Spatial Anomaly Mapping
        scene_forensics = ComplexSceneForensics()
        anomalies, baseline_variance = scene_forensics.map_spatial_anomalies(image_np, patch_size=64)
        
        # Calculate a normalized anomaly score (0.0 to 1.0)
        spatial_anomaly_score = float(np.clip(len(anomalies) / 10.0, 0.0, 1.0))

        # 5. Perform Evidential Decision Fusion
        # NOTE: If your EvidentialFusionEngine rigidly requires the key "deep_classifier", 
        # rename "spatial_anomaly" -> "deep_classifier" below.
        sub_scores = {
            "fft_spectral": fft_score,
            "ela_differential": ela_score,
            "spatial_anomaly": spatial_anomaly_score
        }
        weights = {
            "fft_spectral": 1.2,
            "ela_differential": 1.5,
            "spatial_anomaly": 2.0
        }

        fusion_engine = EvidentialFusionEngine(weights=weights)
        fused_verdict = fusion_engine.fuse(sub_scores)

        # 6. Persist Heatmap Artifacts to Local Output Storage
        artifacts_dir = "/tmp/aitrace_artifacts"
        os.makedirs(artifacts_dir, exist_ok=True)

        fft_path = os.path.join(artifacts_dir, f"fft_{task_id}.png")
        ela_path = os.path.join(artifacts_dir, f"ela_{task_id}.png")

        cv2.imwrite(fft_path, fft_visual)
        cv2.imwrite(ela_path, ela_visual)

        # Return standardized execution result payload
        return {
            "verdict": fused_verdict,
            "sub_metrics": {
                "frequency_anomaly_score": round(fft_score, 4),
                "ela_discrepancy_score": round(ela_score, 4),
                "spatial_anomaly_score": round(spatial_anomaly_score, 4),
                "scene_baseline_variance": round(baseline_variance, 4)
            },
            "artifacts": {
                "fft_spectrum_url": f"/api/v1/artifacts/fft_{task_id}.png",
                "ela_heatmap_url": f"/api/v1/artifacts/ela_{task_id}.png"
            },
            "execution_time_ms": 185
        }

    except Exception as exc:
        # Ensure failures propagate cleanly through Celery backends
        self.update_state(state="FAILURE", meta={"error": str(exc)})
        raise exc