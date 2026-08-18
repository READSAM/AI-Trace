import numpy as np
from scipy.optimize import differential_evolution
from typing import List, Tuple, Dict
from engine import EvidentialFusionEngine 

class QuantWeightOptimizer:
    def __init__(self, dataset: List[Tuple[Dict[str, float], int]]):
        self.dataset = dataset

    def _objective_function(self, x: np.ndarray) -> float:
        weights = {
            'frequency_anomaly_score': x[0],
            'ela_discrepancy_score': x[1],
            'spatial_anomaly_score': x[2],
            'synthetic_noise_score': x[3]  # Added the 4th signal
        }
        bias = x[4]

        engine = EvidentialFusionEngine(weights=weights, bias=bias)
        log_loss = 0.0
        epsilon = 1e-15
        
        for sub_scores, true_label in self.dataset:
            result = engine.fuse(sub_scores)
            pred_prob = result['confidence_score']
            pred_prob = np.clip(pred_prob, epsilon, 1 - epsilon)
            
            if true_label == 1:
                log_loss -= 2.0 * np.log(pred_prob) 
            else:
                log_loss -= np.log(1 - pred_prob)
                
        return log_loss / len(self.dataset)

    def optimize(self):
        print("Launching Differential Evolution Optimizer...")
        bounds = [
            (0.1, 15.0),  # x[0] Frequency
            (0.1, 15.0),  # x[1] ELA
            (0.1, 15.0),  # x[2] Spatial
            (0.1, 15.0),  # x[3] Synthetic
            (-10.0, 5.0)  # x[4] Bias
        ]
        
        result = differential_evolution(
            self._objective_function,
            bounds,
            strategy='best1bin',
            maxiter=1500,
            popsize=20,
            tol=1e-4,
            disp=True
        )
        
        print("\n=== OPTIMAL CALIBRATION FOUND ===")
        print(f"frequency_anomaly_score : {result.x[0]:.4f}")
        print(f"ela_discrepancy_score   : {result.x[1]:.4f}")
        print(f"spatial_anomaly_score   : {result.x[2]:.4f}")
        print(f"synthetic_noise_score   : {result.x[3]:.4f}")
        print(f"bias                    : {result.x[4]:.4f}")
        print(f"Final Log-Loss          : {result.fun:.4f}")

if __name__ == "__main__":
    calibration_data = [
        # Label 1 (AI)
        ({"frequency_anomaly_score": 0.0761, "ela_discrepancy_score": 0.1870, "spatial_anomaly_score": 0.0, "synthetic_noise_score": 1.0}, 1),
        ({"frequency_anomaly_score": 0.0576, "ela_discrepancy_score": 0.1121, "spatial_anomaly_score": 0.0, "synthetic_noise_score": 1.0}, 1),
        ({"frequency_anomaly_score": 0.0681, "ela_discrepancy_score": 0.1146, "spatial_anomaly_score": 0.0, "synthetic_noise_score": 0.6701}, 1), 
        ({"frequency_anomaly_score": 0.0581, "ela_discrepancy_score": 0.2044, "spatial_anomaly_score": 0.0, "synthetic_noise_score": 0.7201}, 1), 
        ({"frequency_anomaly_score": 0.0644, "ela_discrepancy_score": 0.2313, "spatial_anomaly_score": 0.0, "synthetic_noise_score": 1.0}, 1),
        ({"frequency_anomaly_score": 0.0642, "ela_discrepancy_score": 0.0850, "spatial_anomaly_score": 0.0, "synthetic_noise_score": 1.0}, 1),
        ({"frequency_anomaly_score": 0.0565, "ela_discrepancy_score": 0.1444, "spatial_anomaly_score": 0.0, "synthetic_noise_score": 1.0}, 1),
        ({"frequency_anomaly_score": 0.0648, "ela_discrepancy_score": 0.2504, "spatial_anomaly_score": 0.0, "synthetic_noise_score": 1.0}, 1),
        ({"frequency_anomaly_score": 0.0533, "ela_discrepancy_score": 0.0640, "spatial_anomaly_score": 0.0, "synthetic_noise_score": 1.0}, 1),
        ({"frequency_anomaly_score": 0.0543, "ela_discrepancy_score": 0.2789, "spatial_anomaly_score": 0.0, "synthetic_noise_score": 1.0}, 1),
        
        # Label 0 (Authentic)
        ({"frequency_anomaly_score": 0.0638, "ela_discrepancy_score": 0.0492, "spatial_anomaly_score": 0.0, "synthetic_noise_score": 0.7631}, 0),
        ({"frequency_anomaly_score": 0.0631, "ela_discrepancy_score": 0.0055, "spatial_anomaly_score": 0.0, "synthetic_noise_score": 0.6813}, 0),
        ({"frequency_anomaly_score": 0.0671, "ela_discrepancy_score": 0.0121, "spatial_anomaly_score": 0.0, "synthetic_noise_score": 0.8401}, 0),
        ({"frequency_anomaly_score": 0.0601, "ela_discrepancy_score": 0.0836, "spatial_anomaly_score": 0.0, "synthetic_noise_score": 0.7272}, 0),
        ({"frequency_anomaly_score": 0.0657, "ela_discrepancy_score": 0.0099, "spatial_anomaly_score": 0.0, "synthetic_noise_score": 0.6684}, 0),
        ({"frequency_anomaly_score": 0.0651, "ela_discrepancy_score": 0.0266, "spatial_anomaly_score": 0.0, "synthetic_noise_score": 0.7346}, 0),
        ({"frequency_anomaly_score": 0.0645, "ela_discrepancy_score": 0.0739, "spatial_anomaly_score": 0.0, "synthetic_noise_score": 0.6426}, 0),
        ({"frequency_anomaly_score": 0.0605, "ela_discrepancy_score": 0.0308, "spatial_anomaly_score": 0.0, "synthetic_noise_score": 0.7510}, 0),
        ({"frequency_anomaly_score": 0.0631, "ela_discrepancy_score": 0.0216, "spatial_anomaly_score": 0.0, "synthetic_noise_score": 0.5675}, 0),
    ]
    
    optimizer = QuantWeightOptimizer(calibration_data)
    optimizer.optimize()