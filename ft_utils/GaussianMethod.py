from .LatentWalkerMethod import LatentWalkerMethod
import numpy as np
from scipy.stats import chi2
from scipy.optimize import root_scalar
from scipy.linalg import cho_factor, cho_solve

class GaussianMethod(LatentWalkerMethod):
    def __init__(self):
        self.__mu = None
        self.__cov = None

    def train(self, latent_codes, scores):
        if len(latent_codes) != len(scores):
            raise ValueError("latent_codes and scores need to have same length")
        if len(latent_codes) < 100_000:
            raise ValueError("at least 10_000 samples are needed")

        top_k_count = 3_000
        sorted_indices = np.argsort(scores.squeeze())
        best_indices = sorted_indices[-top_k_count:]
        latent_vectors_best = latent_codes[best_indices]

        if np.isnan(latent_vectors_best).any():
            raise ValueError("NaN detected in latent_vectors_best")
        if np.allclose(latent_vectors_best, latent_vectors_best[0]):
            raise ValueError("All latent vectors are (almost) identical")

        print(f"latent_vectors_best shape: {latent_vectors_best.shape[0]-1}")

        self.__mu = np.mean(latent_vectors_best, axis=0)
        self.__cov = np.cov(latent_vectors_best, rowvar=False)
        cond = np.linalg.cond(self.__cov)
        print(f"Condition number: {cond:.2e}")

        if cond > 1e3:
            eps = 1e-4 if cond > 1e5 else 1e-6
            print(f"Applying regularization with ε={eps:.1e}...")
            self.__cov += eps * np.eye(self.__cov.shape[0])

        self.__cho_cov = cho_factor(self.__cov)


    def latent_walk(self, original_latent_code):
        if self.__mu is None:
            raise ValueError("Model not trained. Call train() first.")

        x = np.asarray(original_latent_code).reshape(-1)
        d = x.shape[0]

        strength = 0.2
        confidence = 0.90

        # Mahalanobis distance
        delta = self.__mu - x
        if np.isnan(delta).any() or np.isinf(delta).any():
            raise ValueError("delta contains NaN or Inf")

        D2 = delta @ cho_solve(self.__cho_cov, delta)

        t = chi2.ppf(confidence, df=d)

        if D2 <= 0 or not np.isfinite(D2):
            raise ValueError(f"Invalid Mahalanobis D2: {D2}")

        scale = np.sqrt(t / D2) * strength
        x_target = x + scale * delta

        return np.array([x_target])