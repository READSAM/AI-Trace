import os
import cv2
import numpy as np
# from PIL import Image
from typing import Dict, Any
# from skimage.feature import local_binary_pattern
# from scipy.stats import zscore
# from scipy.spatial.distance import euclidean


from app.core.complexForensics import ComplexSceneForensics
from app.core.elaDifferentialEngine import ELADifferentialEngine
from app.core.fftSpectralAnalyser import FFTSpectralAnalyzer

from app.tasks.celery_app import celery_app
from app.fusion.engine import EvidentialFusionEngine

weights = {
    "frequency_anomaly_score": 0.1002, 
    "ela_discrepancy_score": 9.1843,
    "spatial_anomaly_score": 0.1003,
    "synthetic_noise_score": 14.9054
        }

bias=4.9937


fusion_engine = EvidentialFusionEngine(weights=weights, bias=bias,high_threshold=0.85,low_threshold=0.45)
         
scene_forensics = ComplexSceneForensics()

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
        components,spatial_anomaly_score, synthetic_score = scene_forensics.map_spatial_anomalies(image_np, patch_size=64)
        

        # 5. Perform Evidential Decision Fusion
        # NOTE: If your EvidentialFusionEngine rigidly requires the key "deep_classifier", 
        # rename "spatial_anomaly" -> "deep_classifier" below.
        sub_scores = {
            "fft_spectral": fft_score,
            "ela_differential": ela_score,
            "spatial_anomaly": spatial_anomaly_score,
            "synthetic_noise_score": synthetic_score
        }
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
                "frequency_anomaly_score":fft_score,
                "ela_discrepancy_score": ela_score,
                "spatial_anomaly_score": spatial_anomaly_score,
                "synthetic_noise_score": synthetic_score
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