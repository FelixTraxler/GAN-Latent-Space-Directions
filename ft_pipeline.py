
#%%
from tqdm import tqdm
from ft_utils.BatchImageGenerator import BatchImageGenerator
from ft_utils.BatchImageClassifier import BatchImageClassifier
from ft_utils.InterfaceGANMethod import InterfaceGANMethod
from ft_utils.AverageMethod import AverageMethod
from ft_utils.GaussianMethod import GaussianMethod
from ft_utils.utils import create_grid_from_batch, BATCH_SIZE, create_image_grid, image_prefix, diff_percentage
import numpy as np
import torch

#%%

attributes = ["blonde", "brunette"]

batch_classifier = BatchImageClassifier("out_batch_transfer")
batch_generator = BatchImageGenerator("out_batch_transfer", True)

scores = []
latent_vectors_list = []

text_features = batch_classifier.tokenize_attributes(attributes)

print("Scoring images")
for i in tqdm(range(0, round(200_000 / BATCH_SIZE))):
    probs = batch_classifier.classify_from_batch(i*BATCH_SIZE, BATCH_SIZE, text_features)
    scores.extend([t[0,0].item() for t in probs]) # Use extend for efficiency
    latent_vectors_list.append(batch_generator.load_w_batch(i*BATCH_SIZE, BATCH_SIZE))

latent_vectors = np.concatenate(latent_vectors_list, axis=0)
scores = np.array(scores).reshape(-1, 1)

#%%

from ft_utils.InterfaceGANMethod import InterfaceGANMethod
from ft_utils.AverageMethod import AverageMethod
from ft_utils.GaussianMethod import GaussianMethod

methods = {
    "InterfaceGAN": InterfaceGANMethod(),
    "Average": AverageMethod(),
    "Gaussian": GaussianMethod()
}

print("Starting training...")
for method_name, method in methods.items():
    print("{}: Training...".format(method_name))
    method.train(latent_vectors, scores)

print("Finished training")
#%%

print("Measuring...")
batch_idx = 20
seed_idx = 4
first_seed_w_vector = batch_generator.load_w_batch(batch_idx*BATCH_SIZE, BATCH_SIZE)[seed_idx:seed_idx+1]
ws_first = torch.from_numpy(first_seed_w_vector.astype(np.float32)[np.newaxis, :].repeat(14, axis=1)).to("mps")
original_image = batch_generator.generate_from_w_vec(ws_first[0], filename="method_results/comp_original.png")
original_scores = batch_classifier.classify_image_vec(original_image, text_features)

print("Original diff: {:+.2f} %"
    .format(diff_percentage(original_scores[0][0], original_scores[0][1]))
)
print("Original score: {:.2f}"
    .format(original_scores[0][0])
)

for method_name, method in methods.items():
    resulting_w_vector = method.latent_walk(first_seed_w_vector)
    ws = torch.from_numpy(resulting_w_vector.astype(np.float32)[np.newaxis, :].repeat(14, axis=1)).to("mps")
    image = batch_generator.generate_from_w_vec(ws[0], filename=f"method_results/comp_{method_name}.png")
    new_scores = batch_classifier.classify_image_vec(image, text_features)
    print("{} diff: {:+.2f} %".format(method_name, diff_percentage(new_scores[0][0], new_scores[0][1])))
    print("Original score: {:.2f}"
        .format(new_scores[0][0])
    )


# %%
