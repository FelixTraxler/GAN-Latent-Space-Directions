from .LatentWalkerMethod import LatentWalkerMethod
import numpy as np
from scipy.stats import chi2
from scipy.optimize import root_scalar

class GaussianMethod(LatentWalkerMethod):
    def __init__(self):
        self.__mu = None
        self.__cov = None
        self.__inv_cov = None
        self.__logdet_cov = None
        pass

    def train(self, latent_codes, scores):
        if len(latent_codes) != len(scores):
            raise ValueError("latent_codes and scores need to have same length")
        if len(latent_codes) < 100_000:
            raise ValueError("at least 10_000 samples are needed")

        top_k_count = 3_000
        sorted_indices = np.argsort(scores.squeeze())
        best_indices = sorted_indices[-top_k_count:]
        latent_vectors_best = latent_codes[best_indices]

        self.__mu = np.mean(latent_vectors_best, axis=0)
        centered = latent_vectors_best - self.__mu
        self.__cov = (centered.T @ centered) / (latent_vectors_best.shape[0] - 1)
        self.__cov += 1e-6 * np.eye(self.__cov.shape[0])  # Regularization
        self.__inv_cov = np.linalg.inv(self.__cov)
        sign, self.__logdet_cov = np.linalg.slogdet(self.__cov)


    def latent_walk(self, original_latent_code):
        if self.__mu is None:
            raise ValueError("Model not trained. Call train() first.")

        x = np.asarray(original_latent_code).reshape(-1)
        d = x.shape[0]

        strength = 1
        confidence=0.50

        # return np.array([self.nearest_ellipsoid_boundary(x)])

        # Mahalanobis distance
        delta = self.__mu - x  # walk *toward* class mean
        D2 = delta.T @ self.__inv_cov @ delta
        t = chi2.ppf(confidence, df=d)

        # print(f"mu {self.__mu}")
        # print(f"delta {delta}")
        # print(f"D2 {D2}")
        # print(f"t {t}")

        if D2 <= t:
            return np.array([x])  # already inside confidence region

        scale = np.sqrt(t / D2) * strength

        x_target = x + scale * delta  # pull toward class mean

        # print(f"scale {scale}")
        # print(f"x_target {x_target}")

        return np.array([x_target])

    def nearest_ellipsoid_boundary(self, x, confidence=0.5):
        x = np.asarray(x).ravel()
        d = x.shape[0]

        # 1) Mahalanobis target:
        t = chi2.ppf(confidence, df=d)

        # 2) Centered vector and current Mahalanobis distance
        u  = x - self.__mu
        D2 = u @ (self.__inv_cov @ u)

        # 3) If you’re exactly on the boundary, return x
        if np.isclose(D2, t, atol=1e-8):
            return x.copy()

        # 4) If you’re inside, you probably want the "radial" exit point:
        if D2 < t:
            scale = np.sqrt(t / D2)
            return self.__mu + scale * u

        # 5) You’re outside → solve for a negative λ
        #    Bracket λ in [-λ_max, 0]
        eigvals     = np.linalg.eigvalsh(self.__cov)
        lambda_max  = 0.999 * eigvals.min()

        def f(lmbda):
            M = np.eye(d) - lmbda * self.__inv_cov
            v = np.linalg.solve(M, u)
            return v @ (self.__inv_cov @ v) - t

        # Check signs to be sure bracket is valid
        f_lo, f_hi = f(-lambda_max), f(0.0)
        if f_lo * f_hi > 0:
            raise RuntimeError(f"Could not bracket root: f(-{lambda_max})={f_lo:.3g}, f(0)={f_hi:.3g}")

        sol = root_scalar(f,
                        bracket=[-lambda_max, 0.0],
                        method='bisect',
                        xtol=1e-8)
        λ_opt = sol.root

        # 6) Reconstruct the closest point
        M = np.eye(d) - λ_opt * self.__inv_cov
        v = np.linalg.solve(M, u)
        return self.__mu + v