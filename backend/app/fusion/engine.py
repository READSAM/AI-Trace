import numpy as np
from typing import Dict, Any

class DempsterShaferCombiner:
    """
    Combines conflicting evidence sources using Dempster-Shafer Theory.
    
    Frame of Discernment: \Theta = {AI_GENERATED, AUTHENTIC}
    For each source i with probability p_i and reliability w_i in [0, 1]:
      - m_i({AI})        = w_i * p_i
      - m_i({AUTHENTIC}) = w_i * (1 - p_i)
      - m_i({\Theta})    = 1 - w_i  (Uncertainty / Ignorance)
    """

    @staticmethod
    def combine(sub_scores: Dict[str, float], reliabilities: Dict[str, float]) -> Dict[str, float]:
        """
        Iteratively combines Basic Belief Assignments (BBAs) across all sub-engines.
        """
        # Initialize belief with total ignorance
        m_ai = 0.0
        m_auth = 0.0
        m_u = 1.0

        for key, p in sub_scores.items():
            w = reliabilities.get(key, 0.5)
            # Softened clip to avoid destroying mass distributions
            p_clamped = float(np.clip(p, 0.01, 0.99))
            
            # Source BBAs
            m2_ai = w * p_clamped
            m2_auth = w * (1.0 - p_clamped)
            m2_u = 1.0 - w

            # Compute conflict K: mass assigned to empty set
            k = (m_ai * m2_auth) + (m_auth * m2_ai)

            if k >= 0.99:
                # Total contradiction fallback
                m_ai, m_auth, m_u = 0.5, 0.5, 0.0
                break

            # Normalization factor
            norm = 1.0 / (1.0 - k)

            # Combination Rule
            m_ai_new = norm * ((m_ai * m2_ai) + (m_ai * m2_u) + (m_u * m2_ai))
            m_auth_new = norm * ((m_auth * m2_auth) + (m_auth * m2_u) + (m_u * m2_auth))
            m_u_new = norm * (m_u * m2_u)

            m_ai, m_auth, m_u = m_ai_new, m_auth_new, m_u_new

        return {
            "m_ai": float(m_ai),
            "m_authentic": float(m_auth),
            "m_uncertainty": float(m_u)
        }


class CalibratedLogitSigmoidEnsemble:
    """
    Weighted logit-space aggregator with Platt-scaling calibration.
    """

    def __init__(self, weights: Dict[str, float], bias: float = 0.0):
        self.weights = weights
        self.bias = bias

    def predict_probability(self, sub_scores: Dict[str, float]) -> float:
        fused_logit = self.bias
        total_weight = sum(self.weights.get(k, 1.0) for k in sub_scores.keys())

        for key, prob in sub_scores.items():
            w = self.weights.get(key, 1.0)
            
            # 1. Soften the bounds: Stops a 0.0 or 0.1 score from causing logit explosion.
            # This ensures that 1 anomaly (score 0.1) doesn't veto other engines.
            p = float(np.clip(prob, 0.01, 0.99))
            
            logit = np.log(p / (1.0 - p))
            fused_logit += w * logit

        # 2. Temperature Scaling / Averaging: Normalize by total weight.
        # This stops the weights from compounding the logit into ± infinity, 
        # allowing the sigmoid to return a fractional percentage instead of a flat 0.0 or 1.0.
        if total_weight > 0:
            fused_logit = fused_logit / total_weight

        # Calibrated Sigmoid Activation
        calibrated_prob = 1.0 / (1.0 + np.exp(-fused_logit))
        return float(calibrated_prob)


class EvidentialFusionEngine:
    """
    Main Evidential Decision Engine orchestrating Dempster-Shafer belief mapping,
    logit calibration, and risk thresholding.
    """

    def __init__(
        self, 
        weights: Dict[str, float],
        bias: float = 0.0,
        high_threshold: float = 0.85,
        low_threshold: float = 0.45
    ):
        self.weights = weights
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.logit_ensemble = CalibratedLogitSigmoidEnsemble(weights=weights, bias=bias)

    def fuse(self, sub_scores: Dict[str, float]) -> Dict[str, Any]:
        """
        Executes evidential fusion and assigns structural verdict classification.
        """
        # 1. Calibrated Logit Probability (Now returns accurate fractional scores)
        confidence_score = self.logit_ensemble.predict_probability(sub_scores)

        # 2. Dempster-Shafer Mass Distribution
        max_w = max(self.weights.values()) if self.weights else 1.0
        reliabilities = {k: float(np.clip(v / max_w, 0.1, 0.95)) for k, v in self.weights.items()}
        ds_masses = DempsterShaferCombiner.combine(sub_scores, reliabilities)

        # 3. Dynamic Threshold Classification
        if confidence_score >= self.high_threshold:
            classification = "HIGH_CONFIDENCE_AI_GENERATED"
            is_ai = True
        elif confidence_score >= self.low_threshold:
            classification = "INCONCLUSIVE_HYBRID_MODIFICATION"
            # Fallback to DS mass logic if hybrid zone
            is_ai = True if ds_masses["m_ai"] > ds_masses["m_authentic"] else False
        else:
            classification = "HIGH_CONFIDENCE_AUTHENTIC"
            is_ai = False

        return {
            "is_ai_generated": is_ai,
            "confidence_score": round(confidence_score, 4),
            "classification": classification,
            "belief_distribution": {
                "belief_ai": round(ds_masses["m_ai"], 4),
                "belief_authentic": round(ds_masses["m_authentic"], 4),
                "uncertainty": round(ds_masses["m_uncertainty"], 4)
            }
        }