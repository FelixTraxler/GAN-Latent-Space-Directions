#%%

import torch
from PIL import Image
import open_clip
import numpy as np
from tqdm import tqdm
from scipy.special import softmax


outdir = "out_bakk_256"
seed_start  = 0
seed_end    = 50000

w_list = []
score_list = []

AGE_SCORE_ITEMS = (6,7)

for seed_idx, seed in tqdm(enumerate(range(seed_start, seed_end)), total = seed_end - seed_start):
    score_path = f'{outdir}/seed{seed:04d}_oc2.npy'

    score = np.load(score_path)[0, AGE_SCORE_ITEMS]

    score_list.append(softmax(score))

score_stack = np.stack(score_list, axis=0)

#%%

old_indices = score_stack[:,1].argsort()

Image.open(f'{outdir}/seed{(old_indices[-2]):06d}.png')

#%%

for seed_idx, seed in tqdm(enumerate(range(seed_start, seed_end)), total = seed_end - seed_start):
    w_path = f'{outdir}/seed{(seed):06d}_w.npy'

    w_sample = np.load(w_path)[0,0,:]
    # np.save(w_sample)

    w_list.append(w_sample)

    # w_sample[0,0,:] == w_sample[0,1,:], can save space

w_stack = np.stack(w_list, axis=0)

w_stack.shape

# %%
old_w_vectors = w_stack[old_indices[-1000:], :]

cov_matrix = np.cov(old_w_vectors, rowvar=False)

mu = np.mean(old_w_vectors, axis=0)
np.save("",mu)


#%%

centered = w_stack - mu
cov = (centered.T @ centered) / (w_stack.shape[0] - 1)
cov += 1e-6 * np.eye(512)

# Precompute once:
inv_cov = np.linalg.inv(cov)
sign, logdet_cov = np.linalg.slogdet(cov)  # more stable than log(det)

def logpdf(x):
    """Return the log-density of x under N(mu, cov)."""
    diff = x - mu                           # shape (512,)
    mahal = diff @ inv_cov @ diff           # scalar Mahalanobis distance
    D = mu.shape[0]                         # 512
    return -0.5 * (D * np.log(2*np.pi) + logdet_cov + mahal)

def l2(x):
    return np.linalg.norm(x - mu)

def cos_dist(x):
    return np.linalg.norm(x - mu)

# Example:
logpdf_vec = np.vectorize(logpdf)

s = w_stack[old_indices, :]

#%%
import os
import re
from typing import List, Optional

import click
from stylegan2_ada import dnnlib
import numpy as np
import PIL.Image
import torch

from stylegan2_ada import legacy

network_pkl = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/paper-fig7c-training-set-sweeps/ffhq140k-paper256-ada-bcr.pkl"
# 1024
# network_pkl = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl"
outdir = "out_bakk_256"
truncation_psi = 1

print('Loading networks from "%s"...' % network_pkl)
device = torch.device('mps')
with dnnlib.util.open_url(network_pkl) as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

def gen_image(x):
    mu_stacked = torch.from_numpy(x.astype(np.float32)).repeat(14, 1).unsqueeze(0).to(device)

    img = G.synthesis(mu_stacked, noise_mode="const")

    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    return PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB')


#%%

import matplotlib.pyplot as plt

lp = np.apply_along_axis(l2, axis=1, arr=s)
plt.plot(np.arange(len(lp)),lp)

#%%

x = score_stack[old_indices,0][::1000]

plt.plot(np.arange(len(x)),x)


#%%

def shrink_toward_mean(x, strength):
    """
    strength ∈ [0,1]:
      0 → leave x unchanged
      1 → move x exactly to mu
    """
    return mu + (1 - strength) * (x - mu)


#%%
img = gen_image(mu)
plt.axis("off")
plt.imshow(img)
plt.show()

#%%

initial = w_stack[0]

interpolated_list = []

for i in np.arange(0,1.1,0.1):
    interpolated_list.append(shrink_toward_mean(initial, i))

img = gen_image(modified_w)
plt.axis("off")
plt.imshow(img)
plt.show()


# def shrink_to_target_logp(x, target_logp, tol=1e-6, max_iter=50):
#     # if it already meets the target, return it
#     if logpdf(x) >= target_logp:
#         return x.copy()

#     lo, hi = 0.0, 1.0
#     for _ in range(max_iter):
#         mid = (lo + hi)/2
#         x_mid = shrink_toward_mean(x, mid)
#         if logpdf(x_mid) >= target_logp:
#             hi = mid
#         else:
#             lo = mid
#         if hi - lo < tol:
#             break

#     return shrink_toward_mean(x, hi)

# # Example: push just enough to get log-p ≥ −100
# x_targeted = shrink_to_target_logp(x_new, target_logp=-100.0)
# print("adjusted log-p:", logpdf(x_targeted))
# # %%

# %%
