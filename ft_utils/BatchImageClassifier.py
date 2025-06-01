import torch
from PIL import Image
import open_clip
import numpy as np

from ft_utils.utils import batch_file_prefix, image_prefix

class BatchImageClassifier():
    def __init__(self, outdir):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        self.model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
        self.outdir = outdir
        return


    def generate_image_features(self, start_seed, batch_size):
        with torch.no_grad(), torch.autocast("mps"):
            batch_features = []
            file_prefix = batch_file_prefix(start_seed, batch_size, self.outdir)
            for seed_idx, seed in enumerate(range(start_seed, start_seed + batch_size)):
                output_path = image_prefix(seed, self.outdir)
                image = self.preprocess(Image.open(output_path)).unsqueeze(0) # type: ignore
                image_features = self.model.encode_image(image)
                batch_features.append(image_features)
            np.save(f'{file_prefix}_raw_image_features.npy', batch_features)
            return batch_features[0]

    def load_image_features(self, start_seed, batch_size):
        file_prefix = batch_file_prefix(start_seed, batch_size, self.outdir)
        return np.load(f'{file_prefix}_raw_image_features.npy')
    
    def tokenize_attributes(self, attributes):
        with torch.no_grad(), torch.autocast("mps"):
            text = self.tokenizer(attributes)
            text_features = self.model.encode_text(text)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            return text_features

    def classify_from_batch(self, start_seed, batch_size, text_features):
        with torch.no_grad(), torch.autocast("mps"):

            imgs = self.load_image_features(start_seed, batch_size)

            probabilities = []

            for image_features in imgs:
                image_features /= torch.from_numpy(image_features).norm(dim=-1, keepdim=True)
                probabilities.append((image_features @ text_features.T)) # .softmax(dim=-1))
        return probabilities