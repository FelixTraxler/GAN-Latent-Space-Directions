#%%

%load_ext autoreload
%autoreload 2

#%%

from tqdm import tqdm
from ft_utils.BatchImageGenerator import BatchImageGenerator
from ft_utils.BatchImageClassifier import BatchImageClassifier
from ft_utils.InterfaceGAN import InterfaceGAN
from ft_utils.utils import create_grid_from_batch, BATCH_SIZE, create_image_grid


#%%
batch_generator = BatchImageGenerator("out_batch_transfer", True)
def gen_image_batch():
    """
    Generates images and latent vectors in batches of 64
    """
    for i in tqdm(range(0, round(100_000 / BATCH_SIZE))):
        batch_generator.generate_batch(BATCH_SIZE*i, BATCH_SIZE)
    
# batch_generator.generate_batch(0, BATCH_SIZE)
gen_image_batch()
#%%
# create_grid_from_batch(64*0, 64, "out_batch_transfer")


#%%
def classify_image_batch():
    """
    Generates the image features in batches of 64
    """
    batch_classifier = BatchImageClassifier("out_batch")
    for i in tqdm(range(5353, round(1_000_000 / BATCH_SIZE))):
        print(f"Iteration {i:05d}/{round(1_000_000 / BATCH_SIZE)} {i*BATCH_SIZE}")
        batch_classifier.generate_image_features(BATCH_SIZE*i, BATCH_SIZE)

#%%
# Create image overview for a batch
create_grid_from_batch(20*BATCH_SIZE, BATCH_SIZE, "out_batch")
# %%
batch_classifier = BatchImageClassifier("out_batch")

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

print(f"Finished classifying {len(probs)} images")
# %%

first_seed_w_vector = batch_generator.load_w_batch(20*BATCH_SIZE, BATCH_SIZE)[0:1]

interfacegan = InterfaceGAN()
latent_walk = interfacegan.latent_walk(["blonde", "brunette"], first_seed_w_vector)
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
create_image_grid(paths, "grid_walk.png")

# %%
# text_features = batch_classifier.tokenize_attributes(["male", "female"])

# raw = []
# for i in tqdm(range(0, round(1_000_000 / BATCH_SIZE))):
#     raw = raw + batch_classifier.classify_from_batch(i*BATCH_SIZE, BATCH_SIZE, text_features)

# #%%
# # top = [r for r in raw if r[0][0] > 0.25]
# bottom = [r for r in raw if r[0][1] > 0.3]
# # print(len(top))
# print(len(bottom))
# # print(raw)

# # %%
# import matplotlib.pyplot as plt


# blonde_scores = []
# brunette_scores = []

# for score_tensor in raw:
#     # Assuming score_a is the first element and score_b is the second
#     # If using PyTorch/TensorFlow tensors, you might need:
#     # scores_a.append(score_tensor[0].item())
#     # scores_b.append(score_tensor[1].item())
#     # For simple lists/tuples:
#     blonde_scores.append(score_tensor[0][0])
#     brunette_scores.append(score_tensor[0][1])

# fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True) # sharey makes y-axis scales consistent

# # Histogram for 'a' scores
# axs[0].hist(blonde_scores, bins=15, color='skyblue', edgecolor='black')
# axs[0].set_title('Histogram of Blonde CLIP Scores')
# axs[0].set_xlabel('Score A')
# axs[0].set_ylabel('Frequency')
# axs[0].grid(axis='y', alpha=0.75)

# # Histogram for 'b' scores
# axs[1].hist(brunette_scores, bins=15, color='lightcoral', edgecolor='black')
# axs[1].set_title('Histogram of Brunette CLIP Scores')
# axs[1].set_xlabel('Score B')
# # axs[1].set_ylabel('Frequency') # Not needed if sharey=True
# axs[1].grid(axis='y', alpha=0.75)

# # Add a main title to the figure
# fig.suptitle('Distribution of Scores A and B', fontsize=16)

# # Adjust layout to prevent overlapping titles/labels
# plt.tight_layout(rect=[0, 0, 1, 0.96]) # rect adjusts for suptitle

# # Display the plot
# plt.show()
# # %%

# %%
