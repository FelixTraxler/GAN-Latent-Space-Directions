import numpy as np
from tqdm import tqdm
from ft_utils.utils import BATCH_SIZE
from ft_utils.BatchImageClassifier import BatchImageClassifier
from ft_utils.BatchImageGenerator import BatchImageGenerator
from scipy.stats import chi2

class GaussianLatentWalker:
    def __init__(self):
        self.mu = None
        self.cov = None
        self.inv_cov = None
        self.logdet_cov = None
        return

    def fit_gaussian(self, latent_vectors):
        """
        Fit a Gaussian (mean and covariance) to the provided latent vectors.
        """
        self.mu = np.mean(latent_vectors, axis=0)
        centered = latent_vectors - self.mu
        self.cov = (centered.T @ centered) / (latent_vectors.shape[0] - 1)
        self.cov += 1e-6 * np.eye(self.cov.shape[0])  # Regularization
        self.inv_cov = np.linalg.inv(self.cov)
        sign, self.logdet_cov = np.linalg.slogdet(self.cov)
        return

    def logpdf(self, x):
        """
        Return the log-density of x under N(mu, cov).
        """
        x = np.asarray(x).reshape(-1)  # Ensure 1D
        diff = x - self.mu.reshape(-1)
        mahal = diff @ self.inv_cov @ diff
        D = self.mu.shape[0]
        return -0.5 * (D * np.log(2 * np.pi) + self.logdet_cov + mahal)

    def walk_toward_gaussian(self, x, strength=1.0, steps=10):
        """
        Walk from x toward the class mean (mu) in latent space.
        strength: 0 = no change, 1 = move to mu
        steps: number of interpolation steps
        Returns an array of latent vectors along the walk.
        """
        walk = np.stack([
            self.mu + (1 - alpha) * (x - self.mu)
            for alpha in np.linspace(0, strength, steps)
        ], axis=0)
        return walk

    def walk_to_confidence_boundary(self, x, confidence=0.5, steps=10, strength=1.0):
        x = np.asarray(x).reshape(-1)
        d = x.shape[0]
        delta = x - self.mu.reshape(-1)
        D2 = delta.T @ self.inv_cov @ delta
        t = chi2.ppf(confidence, df=d)
        scale = np.sqrt(t / D2) * strength  # <-- scale up the walk
        x_target = self.mu + scale * delta
        alphas = np.linspace(0, 1, steps)[:, None]
        walk = (1 - alphas) * x[None, :] + alphas * x_target[None, :]
        return walk

    # def walk_to_confidence_boundary(self, x, confidence=0.5, steps=10):
    #     """
    #     Walk from x out to the confidence-boundary ellipsoid of the fitted Gaussian.
    #     Args:
    #     x           : 1D latent vector.
    #     confidence  : desired coverage, e.g. 0.5 for the 50% contour.
    #     steps       : how many points to return along the path.
    #     Returns:
    #     An array of shape (steps, dim) of latent vectors,
    #     starting at x and ending at the 50%‐boundary point.
    #     """
    #     x = np.asarray(x).reshape(-1)  # Ensure 1D
    #     d = x.shape[0]
    #     delta = x - self.mu.reshape(-1)
    #     D2 = delta.T @ self.inv_cov @ delta
    #     t = chi2.ppf(confidence, df=d)
    #     scale = np.sqrt(t / D2)
    #     x_target = self.mu + scale * delta
    #     alphas = np.linspace(0, 1, steps)[:, None]
    #     walk = (1 - alphas) * x[None, :] + alphas * x_target[None, :]
    #     return walk

    def walk_by_logpdf(self, x, target_logpdf, tol=1e-6, max_iter=50):
        """
        Move x toward mu until logpdf(x) >= target_logpdf (binary search).
        Returns the adjusted latent vector.
        """
        x = np.asarray(x).reshape(-1)  # Ensure 1D
        if self.logpdf(x) >= target_logpdf:
            return x.copy()
        lo, hi = 0.0, 1.0
        for _ in range(max_iter):
            mid = (lo + hi) / 2
            x_mid = self.mu + (1 - mid) * (x - self.mu)
            if self.logpdf(x_mid) >= target_logpdf:
                hi = mid
            else:
                lo = mid
            if hi - lo < tol:
                break
        return self.mu + (1 - hi) * (x - self.mu)

    def latent_walk(self, attributes, latent_vector):
        batch_classifier = BatchImageClassifier("out_batch_transfer")
        batch_generator = BatchImageGenerator("out_batch_transfer", True)

        def classify(start_seed, batch_size, text):
            return batch_classifier.classify_from_batch(start_seed, batch_size, text)

        scores = []
        latent_vectors_list = []
        text_features = batch_classifier.tokenize_attributes(attributes)

        print("Scoring images")
        for i in tqdm(range(0, round(200_000 / BATCH_SIZE))):
            probs = classify(i * BATCH_SIZE, BATCH_SIZE, text_features)
            scores.extend([t[0, 0].item() for t in probs])
            latent_vectors_list.append(batch_generator.load_w_batch(i * BATCH_SIZE, BATCH_SIZE))

        latent_vectors = np.concatenate(latent_vectors_list, axis=0)
        scores = np.array(scores).reshape(-1, 1)

        # Select top_k_count best (highest score) for the class
        top_k_count = 30_000
        sorted_indices = np.argsort(scores.squeeze())
        best_indices = sorted_indices[-top_k_count:]
        latent_vectors_best = latent_vectors[best_indices]

        self.fit_gaussian(latent_vectors_best)

        # Walk from latent_vector toward the class mean
        return self.walk_toward_gaussian(latent_vector.squeeze(), strength=1.0, steps=64)

    def walk_to_realistic_logpdf(self, attributes, latent_vector, percentile=50, steps=64, strength=1.0):
        """
        Walk from latent_vector toward the mean until it reaches a logpdf at least as high as the given percentile
        (e.g., 50 for the median) of the logpdfs of the training set. Returns the walk and the threshold used.
        """
        batch_classifier = BatchImageClassifier("out_batch_transfer")
        batch_generator = BatchImageGenerator("out_batch_transfer", True)

        def classify(start_seed, batch_size, text):
            return batch_classifier.classify_from_batch(start_seed, batch_size, text)

        scores = []
        latent_vectors_list = []
        text_features = batch_classifier.tokenize_attributes(attributes)

        print("Scoring images")
        for i in tqdm(range(0, round(400_000 / BATCH_SIZE))):
            probs = classify(i * BATCH_SIZE, BATCH_SIZE, text_features)
            scores.extend([t[0, 0].item() for t in probs])
            latent_vectors_list.append(batch_generator.load_w_batch(i * BATCH_SIZE, BATCH_SIZE))

        latent_vectors = np.concatenate(latent_vectors_list, axis=0)
        scores = np.array(scores).reshape(-1, 1)

        # Select top_k_count best (highest score) for the class
        top_k_count = 3_000
        sorted_indices = np.argsort(scores.squeeze())
        best_indices = sorted_indices[-top_k_count:]
        latent_vectors_best = latent_vectors[best_indices]

        self.fit_gaussian(latent_vectors_best)

        # return self.walk_to_confidence_boundary(np.asarray(latent_vector).reshape(-1), confidence=percentile, steps=64, strength=strength)

    def walk_to_avg_until_percentile(self, attributes, latent_vector, percentile=50, steps=16):
        """
        Walk from latent_vector toward the mean, generating a sequence of latent vectors
        that approach the logpdf threshold (e.g., the 50th percentile of the class).
        At each step, use walk_by_logpdf to move to the next logpdf target, and collect
        all intermediate latent vectors. Stops at the first vector that meets or exceeds the threshold.
        Returns the walk (list of latent vectors) and the threshold used.
        """
        batch_classifier = BatchImageClassifier("out_batch_transfer")
        batch_generator = BatchImageGenerator("out_batch_transfer", True)

        def classify(start_seed, batch_size, text):
            return batch_classifier.classify_from_batch(start_seed, batch_size, text)

        scores = []
        latent_vectors_list = []
        text_features = batch_classifier.tokenize_attributes(attributes)

        print("Scoring images")
        for i in tqdm(range(0, round(400_000 / BATCH_SIZE))):
            probs = classify(i * BATCH_SIZE, BATCH_SIZE, text_features)
            scores.extend([t[0, 0].item() for t in probs])
            latent_vectors_list.append(batch_generator.load_w_batch(i * BATCH_SIZE, BATCH_SIZE))

        latent_vectors = np.concatenate(latent_vectors_list, axis=0)
        scores = np.array(scores).reshape(-1, 1)

        # Select top_k_count best (highest score) for the class
        top_k_count = 3_000
        sorted_indices = np.argsort(scores.squeeze())
        best_indices = sorted_indices[-top_k_count:]
        latent_vectors_best = latent_vectors[best_indices]

        self.fit_gaussian(latent_vectors_best)

        # Compute logpdfs for the best latent vectors
        logpdfs = np.array([self.logpdf(vec) for vec in latent_vectors_best])
        threshold = np.percentile(logpdfs, percentile)

        # Create a sequence of target logpdfs from the current logpdf to the threshold
        current_logpdf = self.logpdf(np.asarray(latent_vector).reshape(-1))
        if current_logpdf >= threshold:
            return np.expand_dims(np.asarray(latent_vector).reshape(-1), 0), threshold
        logpdf_targets = np.linspace(current_logpdf, threshold, steps)

        walk = []
        x = np.asarray(latent_vector).reshape(-1)
        for t_logpdf in logpdf_targets:
            x = self.walk_by_logpdf(x, t_logpdf)
            walk.append(x.copy())
            if self.logpdf(x) >= threshold:
                break
        return np.stack(walk, axis=0), threshold 
