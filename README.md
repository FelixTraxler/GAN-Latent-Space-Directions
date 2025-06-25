# Semantic Latent Space Walks in StyleGAN2-ADA

This project analyzes and compares techniques for walking in semantic directions in the latent input space of a GAN, specifically using StyleGAN2-ADA as the baseline generator. The goal is to evaluate two established methods and one novel approach for manipulating generated images along interpretable directions, using OpenCLIP for semantic classification.

## Project Overview

- **Research Goal:**  
  Analyze and compare three techniques for traversing ("walking") in semantic directions in the latent space of StyleGAN2-ADA:

  1. **InterfaceGAN** (existing): Finds hyperplanes in latent space that separate binary semantic features (e.g., "beard" vs. "no beard") and walks along the normal vector to manipulate the feature.
  2. **Average Vector Method** (existing): Uses the mean latent vector of a class as a direction for semantic walks.
  3. **Gaussian/Mahalanobis Method** (novel): Uses a Gaussian model for each class and walks by minimizing Mahalanobis distance to the class distribution.

- **Image Classification:**  
  OpenCLIP is used to classify generated images according to semantic attributes, providing the supervision signal for finding and evaluating semantic directions.

- **Generator:**  
  StyleGAN2-ADA is used for image synthesis, with pretrained weights.

## Code Structure

| Path / File                                         | Description                                                                                    |
| --------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `pipeline.ipynb`                                    | MAIN: Pipeline for scoring different approaches; expects sample image gen/class to be finished |
| `batch_gen.py`                                      | Main script for batch image generation and classification.                                     |
| `bakk_gen.py`                                       | DEPRECATED: Script for generating images and saving latent vectors with StyleGAN2-ADA.         |
| `openclip.py`                                       | Scripts for extracting OpenCLIP features and classifying images.                               |
| `our_approach.py`                                   | Implementation and experiments for the new latent walk technique.                              |
| `results/`                                          | Example output images and result visualizations.                                               |
| `ft_utils/BatchImageGenerator.py`                   | Class for batch image and latent vector generation using StyleGAN2-ADA.                        |
| `ft_utils/BatchImageClassifier.py`                  | Class for batch classification of images using OpenCLIP.                                       |
| `ft_utils/[Gaussian/Average/InterfaceGAN]Method.py` | Implementation of the a method for finding semantic directions.                                |
| `ft_utils/utils.py`                                 | Utility functions for batching, file naming, and image grid creation.                          |
| `interfacegan/`                                     | Original InterfaceGAN code and pretrained boundaries for reference.                            |
| `stylegan2_ada/`                                    | StyleGAN2-ADA code (imported as a submodule or local copy).                                    |

## Techniques

### 1. InterfaceGAN

- **How it works:**

  - Generate a large set of images and corresponding latent vectors.
  - Classify images into binary semantic categories using OpenCLIP.
  - Train a linear SVM to find a separating hyperplane in latent space.
  - The normal vector to this hyperplane is used as the semantic direction for latent walks (e.g., from "no beard" to "beard").

- **Implementation:**  
  See `ft_utils/InterfaceGAN.py` and usage in `batch_gen.py`.

---

### 2. Average Vector Method

- **How it works:**

  - For a given semantic class (e.g., "old"), collect all latent vectors whose generated images are classified as that class.
  - Compute the average (mean) latent vector for the class.
  - The direction from a given latent vector to the class mean is used as the semantic direction. Walking in this direction moves the generated image toward the target class.

- **Implementation:**  
  This method is implemented in scripts such as `our_approach.py` (see the computation of `mu` and the function `shrink_toward_mean`).  
  Example:
  ```python
  def shrink_toward_mean(x, strength):
      # strength ∈ [0,1]: 0 → leave x unchanged, 1 → move x exactly to mu
      return mu + (1 - strength) * (x - mu)
  ```
  Walking toward the mean vector allows smooth interpolation into the semantic class.

---

### 3. Gaussian/Mahalanobis (Novel) Method

- **How it works:**

  - Represent each semantic class as a Gaussian distribution in latent space, using the mean and covariance of the class's latent vectors.
  - For a given latent vector, compute its Mahalanobis distance to the class Gaussian.
  - Walk in latent space by minimizing the Mahalanobis distance (or maximizing the log-probability) with respect to the class Gaussian. This allows for more nuanced, distribution-aware semantic walks, potentially using multiple Gaussians for multi-modal classes.

- **Implementation:**  
  See `our_approach.py` for the computation of the mean (`mu`), covariance (`cov`), and Mahalanobis distance:
  ```python
  inv_cov = np.linalg.inv(cov)
  def logpdf(x):
      diff = x - mu
      mahal = diff @ inv_cov @ diff
      D = mu.shape[0]
      return -0.5 * (D * np.log(2*np.pi) + logdet_cov + mahal)
  ```
  You can walk in the direction that increases the log-probability (i.e., decreases the Mahalanobis distance) to move a latent vector toward the semantic class in a statistically principled way.

---

## Running the Project

1. **Generate Images and Latents:**
   ```bash
   python batch_gen.py
   ```
2. **Classify Images:**
   - Use `openclip.py` or the batch classifier in `batch_gen.py`.
3. **Run InterfaceGAN or New Approach:**
   - See `batch_gen.py` and `our_approach.py` for examples.

## Dependencies

- Python 3.8+
- PyTorch (with MPS or CUDA support)
- OpenCLIP
- StyleGAN2-ADA (local or as a submodule)
- scikit-learn, numpy, PIL, tqdm, click, matplotlib

Install requirements with:

```bash
pip install torch open_clip_torch scikit-learn numpy pillow tqdm click matplotlib
```

## Results

See the `results/` directory for example visualizations of latent walks and semantic manipulations.
