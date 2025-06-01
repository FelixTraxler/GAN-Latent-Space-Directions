# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import os
import re
from typing import List, Optional

import click
from stylegan2_ada import dnnlib
import numpy as np
import PIL.Image
import torch

from ft_utils import legacy

#----------------------------------------------------------------------------

def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

#----------------------------------------------------------------------------

@click.command()
@click.pass_context
# @click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
# @click.option('--seeds', type=num_range, help='List of random seeds')
# @click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
# @click.option('--projected-w', help='Projection result file', type=str, metavar='FILE')
# @click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
def generate_images(
    ctx: click.Context,
    # network_pkl: str,
    # seeds: Optional[List[int]],
    # truncation_psi: float,
    noise_mode: str,
    # outdir: str,
    class_idx: Optional[int],
):
    # 256
    network_pkl = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/paper-fig7c-training-set-sweeps/ffhq140k-paper256-ada-bcr.pkl"
    # 1024
    # network_pkl = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl"
    outdir = "out_bakk_256"
    truncation_psi = 1

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('mps')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    os.makedirs(outdir, exist_ok=True)

    # Labels.
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            ctx.fail('Must specify class label with --class when using a conditional network')
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print ('warn: --class=lbl ignored when running on an unconditional network')

    seed_start = 700000
    seed_end = 1000000

    # Generate images.
    for seed_idx, seed in enumerate(range(seed_start, seed_end)):
        if (seed_idx % 50 == 0):
            print(f'Generating image {seed}/{seed_end} ...')

        # Create Z vector (initial latent)
        # G.z_dim is the dimensionality of the Z vector for this model
        z_numpy = np.random.RandomState(seed).randn(1, G.z_dim).astype(np.float32)
        z = torch.from_numpy(z_numpy).to(device)

        # Save Z vector
        # print(f'  Saved Z vector (shape: {z_numpy.shape}) to {outdir}/seed{seed:04d}_z.npy')

        # --- Explicitly call G.mapping to get Ws ---
        # The G.mapping module is an instance of MappingNetwork.
        # Its forward method takes (z, c, truncation_psi, truncation_cutoff, skip_w_avg_update).
        # print(f'  Mapping Z to Ws (truncation_psi={truncation_psi})...')
        ws = G.mapping(z, label, truncation_psi=truncation_psi, truncation_cutoff=None)
        # ws shape is [batch_size, G.num_ws, G.w_dim], e.g., [1, 18, 512] for FFHQ 1024px

        # Save Ws vector
        ws_numpy = ws.cpu().numpy() # ws is a tensor, convert to numpy for saving
        # print(f'  Saved Ws vector (shape: {ws_numpy.shape}) to {outdir}/seed{seed:04d}_w.npy')

        # --- Call G.synthesis with Ws to generate the image ---
        # The G.synthesis module is an instance of SynthesisNetwork.
        # Its forward method takes (ws, **block_kwargs).
        # noise_mode is one of the block_kwargs.
        # print(f'  Synthesizing image from Ws (noise_mode={noise_mode})...')
        img = G.synthesis(ws, noise_mode=noise_mode) # Other **synthesis_kwargs from Generator.forward can be passed here if needed

        # Post-process and save image
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        output_path = f'{outdir}/seed{seed:06d}.png'
        PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(output_path)
        np.save(f'{outdir}/seed{seed:06d}_w.npy', ws_numpy)
        np.save(f'{outdir}/seed{seed:06d}_z.npy', z_numpy)


#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
