import cv2
import numpy as np
from typing import Tuple

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