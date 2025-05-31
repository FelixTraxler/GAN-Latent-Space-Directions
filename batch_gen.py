#%%

%load_ext autoreload # ignore
%autoreload 2

#%%

from tqdm import tqdm
import numpy as np

from ft_utils.BatchImageGenerator import BatchImageGenerator
from ft_utils.BatchImageClassifier import BatchImageClassifier
from ft_utils.utils import create_grid_from_batch
from ft_utils.utils import batch_file_prefix

BATCH_SIZE = 64

#%%
def gen_image_batch():
    """
    Generates images and latent vectors in batches of 64
    """
    batch_generator = BatchImageGenerator()
    for i in tqdm(range(3439, round(1_000_000 / BATCH_SIZE))):
        print(f"Iteration {i:05d}/{round(1_000_000 / BATCH_SIZE)} {i*BATCH_SIZE}")
        batch_generator.generate_batch(BATCH_SIZE*i, BATCH_SIZE)

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
create_grid_from_batch(64*0, 64, "out_batch")
# %%
batch_classifier = BatchImageClassifier("out_batch")

def classify_young(start_seed, batch_size, text):
    return batch_classifier.classify_from_batch(start_seed, batch_size, text)

probs = []

# text_features = batch_classifier.tokenize_attributes(["man", "woman"])
text_features = batch_classifier.tokenize_attributes(["not old", "old"])

# for i in tqdm(range(170, round(1_000_000 / BATCH_SIZE))):
    # np.save(batch_file_prefix(BATCH_SIZE*i, BATCH_SIZE, "out_batch") + "_score_mw.npy", classify_young(64*i, 64, text_features))
probs = classify_young(0, 64, text_features)
print([f"{t[0,0].item():.2f}" for t in probs])
is_young = [t[0, 0].item() < 0.5 for t in probs]
print(is_young)

print(f"Finished classifying {len(probs)} images")
# %%
