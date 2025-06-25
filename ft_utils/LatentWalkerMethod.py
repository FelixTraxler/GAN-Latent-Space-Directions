from abc import ABC, abstractmethod

class LatentWalkerMethod(ABC):
    @abstractmethod
    def train(self, latent_codes, scores):
        pass

    @abstractmethod
    def latent_walk(self, original_latent_code):
        """
        Walks a latent code in the trained direction.
        Throws error if `train` wasn't called before.
        """
        pass
