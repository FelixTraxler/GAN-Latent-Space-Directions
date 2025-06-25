from .LatentWalkerMethod import LatentWalkerMethod
import numpy as np
from sklearn import svm

class InterfaceGANMethod(LatentWalkerMethod):
    def __init__(self):
        self.__boundary = None
        pass

    def train(self, latent_codes, scores):
        if len(latent_codes) != len(scores):
            raise ValueError("latent_codes and scores need to have same length")
        if len(latent_codes) < 10_000:
            raise ValueError("at least 10_000 samples are needed")

        top_k_count = 3_000
        sorted_indices = np.argsort(scores.squeeze())
        worst_indices = sorted_indices[:top_k_count]
        best_indices = sorted_indices[-top_k_count:]
        combined_indices = np.concatenate([best_indices, worst_indices])

        latent_vectors_best_and_worst = latent_codes[combined_indices]

        latent_space_dim = latent_vectors_best_and_worst.shape[1]

        train_label = np.concatenate([np.ones(top_k_count, dtype=np.int32),
                                    np.zeros(top_k_count, dtype=np.int32)], axis=0)

        clf = svm.SVC(kernel='linear')
        classifier = clf.fit(latent_vectors_best_and_worst, train_label)
        print(f'Finish training.')

        a = classifier.coef_.reshape(1, latent_space_dim).astype(np.float32)
        self.__boundary = a / np.linalg.norm(a)
        pass

    def latent_walk(self, original_latent_code):
        if self.__boundary is None: raise ValueError("No Boundary exists. Run find_boundary first")

        assert (original_latent_code.shape[0] == 1 and self.__boundary.shape[0] == 1 and
                len(self.__boundary.shape) == 2 and
                self.__boundary.shape[1] == original_latent_code.shape[-1])

        linspace = np.linspace(3, 3, 1)
        if len(original_latent_code.shape) == 2:
            linspace = linspace - original_latent_code.dot(self.__boundary.T)
            linspace = linspace.reshape(-1, 1).astype(np.float32)
            return original_latent_code + linspace * self.__boundary
        if len(original_latent_code.shape) == 3:
            linspace = linspace.reshape(-1, 1, 1).astype(np.float32)
            return original_latent_code + linspace * self.__boundary.reshape(1, 1, -1)
        raise ValueError(f'Input `original_latent_code` should be with shape '
                        f'[1, latent_space_dim] or [1, N, latent_space_dim] for '
                        f'W+ space in Style GAN!\n'
                        f'But {original_latent_code.shape} is received.')
