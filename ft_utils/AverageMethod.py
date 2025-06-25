from .LatentWalkerMethod import LatentWalkerMethod
import numpy as np
from sklearn import svm

class AverageMethod(LatentWalkerMethod):
    def __init__(self):
        self.__average_vector = None
        pass

    def train(self, latent_codes, scores):
        if len(latent_codes) != len(scores):
            raise ValueError("latent_codes and scores need to have same length")
        if len(latent_codes) < 10_000:
            raise ValueError("at least 10_000 samples are needed")

        top_k_count = 30_000
        sorted_indices = np.argsort(scores.squeeze())
        best_indices = sorted_indices[-top_k_count:]
        latent_vectors_best = latent_codes[best_indices]

        self.__average_vector = np.mean(latent_vectors_best, axis=0)

    def latent_walk(self, original_latent_code):
        if self.__average_vector is None: raise ValueError("No Average Vector exists. Run train first")

        strength = 0.5

        return strength * self.__average_vector + (1 - strength) * original_latent_code