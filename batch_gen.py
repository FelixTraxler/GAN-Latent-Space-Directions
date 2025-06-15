#%%

from tqdm import tqdm
from ft_utils.BatchImageGenerator import BatchImageGenerator
from ft_utils.BatchImageClassifier import BatchImageClassifier
from ft_utils.InterfaceGAN import InterfaceGAN
from ft_utils.utils import create_grid_from_batch, BATCH_SIZE, create_image_grid, image_prefix


#%%
batch_generator = BatchImageGenerator("out_batch_transfer", True)
def gen_image_batch():
    """
    Generates images and latent vectors in batches of 64
    """
    for i in tqdm(range(0, round(1_000_000 / BATCH_SIZE))):
        batch_generator.generate_batch(BATCH_SIZE*i, BATCH_SIZE)
    
# batch_generator.generate_batch(0, BATCH_SIZE)
# gen_image_batch()
#%%
def classify_image_batch():
    """
    Generates the image features in batches of 64
    """
    batch_classifier = BatchImageClassifier("out_batch_transfer")
    for i in tqdm(range(6270, round(1_000_000 / BATCH_SIZE))):
        batch_classifier.generate_image_features(BATCH_SIZE*i, BATCH_SIZE)

# classify_image_batch()
#%%
# Create image overview for a batch
# create_grid_from_batch(0*BATCH_SIZE, BATCH_SIZE, "out_batch_transfer")
# %%
batch_classifier = BatchImageClassifier("out_batch_transfer")

def classify_young(start_seed, batch_size, text):
    return batch_classifier.classify_from_batch(start_seed, batch_size, text)

probs = []

# text_features = batch_classifier.tokenize_attributes(["man", "woman"])
# text_features = batch_classifier.tokenize_attributes(["blonde", "brunette"])

# for i in tqdm(range(170, round(1_000_000 / BATCH_SIZE))):
    # np.save(batch_file_prefix(BATCH_SIZE*i, BATCH_SIZE, "out_batch") + "_score_mw.npy", classify_young(64*i, 64, text_features))
# probs = classify_young(0, 64, text_features)
# print([f"{t[0,0].item():.2f}" for t in probs])
# is_young = [t[0, 0].item() < 0.5 for t in probs]
# print(is_young)

# print(f"Finished classifying {len(probs)} images")
# %%

first_seed_w_vector = batch_generator.load_w_batch(20*BATCH_SIZE, BATCH_SIZE)[0:1]
seed = 20*BATCH_SIZE
w_image_path = image_prefix(seed, "../out_batch_transfer")

interfacegan = InterfaceGAN()
latent_walk = interfacegan.latent_walk(["blonde", "not blonde"], first_seed_w_vector)
print(latent_walk.shape)
print(first_seed_w_vector.shape)

#%%
import numpy as np
import torch

ws = torch.from_numpy(latent_walk[:, np.newaxis, :].repeat(14, axis=1)).to("mps")
paths = []

for (index, lv) in enumerate(ws):
    p = f"walk/lv{index}.png"
    paths.append(p)
    batch_generator.generate_from_w_vec(lv, p)

#%%
print(len(paths))
create_image_grid([w_image_path, paths[-1]], "grid_walk.png", grid_size=(1, 2))

# %%
import matplotlib.pyplot as plt

# %%
def feature_stats(feature_name):
    text_features = batch_classifier.tokenize_attributes(feature_name)

    raw = []
    for i in tqdm(range(0, round(100_000 / BATCH_SIZE))):
        raw = raw + batch_classifier.classify_from_batch(i*BATCH_SIZE, BATCH_SIZE, text_features)

    scores = []
    scores_by_feature = [[] for _ in feature_name]

    for score_tensor in raw:
        # Assuming score_a is the first element and score_b is the second
        # If using PyTorch/TensorFlow tensors, you might need:
        # scores_a.append(score_tensor[0].item())
        # scores_b.append(score_tensor[1].item())
        # For simple lists/tuples:
        for i, feature in enumerate(feature_name):
            scores_by_feature[i].append(score_tensor[0][i])

    fig, axs = plt.subplots(len(feature_name) // 2, 2, figsize=(12, 5), sharey=True) # sharey makes y-axis scales consistent

    for i, feature in enumerate(feature_name):
        axs[i // 2, i % 2].hist(scores_by_feature[i], bins=15, color='skyblue', edgecolor='black')
        axs[i // 2, i % 2].set_title(f'Histogram of {feature} CLIP Scores')
        axs[i // 2, i % 2].set_xlabel(f'Score {feature}')
        axs[i // 2, i % 2].set_ylabel('Frequency')
        axs[i // 2, i % 2].grid(axis='y', alpha=0.75)

    # Adjust layout to prevent overlapping titles/labels
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # rect adjusts for suptitle

    # Display the plot
    plt.show()
# %%
# feature_stats(["female", "male",  "young", "old"])
# %%

from ft_utils.GaussianLatentWalker import GaussianLatentWalker

glw = GaussianLatentWalker()

lw2 = glw.walk_to_realistic_logpdf(["blonde"], first_seed_w_vector, percentile=0.999, strength=1.0)
# %%

ws2 = torch.from_numpy(lw2.astype(np.float32)[:, np.newaxis, :].repeat(14, axis=1)).to("mps")

paths = []
for (index, lv) in enumerate(ws2):
    p = f"walk_gaussian/lv{index}.png"
    paths.append(p)
    batch_generator.generate_from_w_vec(lv, p)
# %%
print(len(paths))
create_image_grid([paths[0], paths[-1]], "grid_walk2.png", grid_size=(1, 2))
# %%
import numpy as np

# Compute the Euclidean distance between the first and last vector in lw2
if isinstance(lw2, np.ndarray):
    first_vec = lw2[0]
    last_vec = lw2[-1]
    distance = np.linalg.norm(last_vec - first_vec)
    print(f"Distance between first and last vector of lw2: {distance}")
else:
    print("lw2 is not a numpy array; cannot compute distance.")

# %%
