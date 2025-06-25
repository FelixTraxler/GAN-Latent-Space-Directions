import os

import sys
import os
sys.path.append("stylegan2_ada")

import dnnlib
import numpy as np
import PIL.Image
import torch
import time

import legacy

class BatchImageGenerator:
    def __init__(self, outdir, transfer_learning):
        if transfer_learning:
            network_pkl = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res256-mirror-paper256-noaug.pkl"
        else:
            network_pkl = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/paper-fig7c-training-set-sweeps/ffhq140k-paper256-ada-bcr.pkl"

        self.outdir = "../" + outdir
        self.truncation_psi = 1

        print('Loading networks from "%s"...' % network_pkl)
        self.device = torch.device('mps')
        with dnnlib.util.open_url(network_pkl) as f:
            self.G = legacy.load_network_pkl(f)['G_ema'].to(self.device) # type: ignore

        self.label = torch.zeros([1, self.G.c_dim], device=self.device)
        os.makedirs(self.outdir, exist_ok=True)

    def load_w_batch(self, start_seed, batch_size):
        return np.load(f'{self.outdir}/batch_{start_seed:06d}-{batch_size:06d}_ws.npy')

    def generate_from_z_batch(self, start_seed, batch_size, save_w):
        z_numpy = np.load(f'{self.outdir}/batch_{start_seed:06d}-{batch_size:06d}_zs.npy')
        z = torch.from_numpy(z_numpy).to(self.device)

        return self.generate_from_z_vectors(start_seed, batch_size, save_w, z)

    def generate_w_from_z_batch(self, start_seed, batch_size):
        z_numpy = np.load(f'{self.outdir}/batch_{start_seed:06d}-{batch_size:06d}_zs.npy')
        z = torch.from_numpy(z_numpy).to(self.device)
        ws = self.G.mapping(z, self.label, truncation_psi=self.truncation_psi, truncation_cutoff=None)
        np.save(f'{self.outdir}/batch_{start_seed:06d}-{batch_size:06d}_ws.npy', ws.cpu().numpy()[:,0,:])

    def generate_from_ws_batch(self, start_seed, batch_size):
        ws_numpy = self.load_w_batch(start_seed, batch_size)
        ws = torch.from_numpy(ws_numpy[:, np.newaxis, :].repeat(14, axis=1)).to(self.device)

        return self.generate_from_w_vectors(start_seed, batch_size, ws)

    def generate_from_w_vec(self, w_vector, filename):
        img = self.G.synthesis(w_vector.unsqueeze(0), noise_mode="const")

        # Post-process and save the image
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        img = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB')
        if (filename is not None):
            img.save(filename)
        return img

    def generate_from_z_vectors(self, start_seed, batch_size, save, z):
        ws = self.G.mapping(z, self.label, truncation_psi=self.truncation_psi, truncation_cutoff=None)
        if save:
            np.save(f'{self.outdir}/batch_{start_seed:06d}-{batch_size:06d}_ws.npy', ws.cpu().numpy()[:,0,:])
        return self.generate_from_w_vectors(start_seed, batch_size, ws)
        
    def generate_from_w_vectors(self, start_seed, batch_size, ws):
        seeds = list(range(start_seed, start_seed + batch_size))
        
        for i, (seed, w_vector) in enumerate(zip(seeds, ws)):
            # Perform inference on a single vector from the batch
            # Add a batch dimension to the w_vector for the synthesis network
            img = self.G.synthesis(w_vector.unsqueeze(0), noise_mode="const")

            # Post-process and save the image
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            output_path = f'{self.outdir}/seed{seed:06d}.png'
            PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(output_path)
        return


    def generate_batch(self, start_seed, batch_size):
        """
        Generates a batch of images
        """
        # Generate all seeds for the batch
        seeds = list(range(start_seed, start_seed + batch_size))
        # Generate all z vectors in a single batch
        z_numpy = np.stack([np.random.RandomState(seed).randn(self.G.z_dim) for seed in seeds]).astype(np.float32)
        z = torch.from_numpy(z_numpy).to(self.device)

        np.save(f'{self.outdir}/batch_{start_seed:06d}-{batch_size:06d}_zs.npy', z_numpy)

        return self.generate_from_z_vectors(start_seed, batch_size, True, z)

    def batch_performance_tests_until(self, max_batch_size: int):
        """
        Makes performance tests for different batch sizes.
        """
        batch_sizes_to_test = 2**np.arange(int(np.log2(max_batch_size)) + 1)
        start_seed = 0

        print("\n--- Performance Tests ---")
        header = f"{'Batch Size':<12} | {'generate_batch (s)':<20} | {'from_z_batch (s)':<18} | {'from_ws_batch (s)':<18} | {'Imgs/s (gen_batch)':<20}"
        print(header)
        print("-" * len(header))

        for bs in batch_sizes_to_test:            
            file_path_z = f'{self.outdir}/batch_{start_seed:06d}-{bs:06d}_zs.npy'
            file_path_w = f'{self.outdir}/batch_{start_seed:06d}-{bs:06d}_ws.npy'

            # Clean up files from previous test runs to ensure fair timing for file I/O
            if os.path.exists(file_path_z):
                os.remove(file_path_z)
            if os.path.exists(file_path_w):
                os.remove(file_path_w)
            
            # Synchronize before timing if on GPU
            torch.mps.synchronize()

            # 1. Test generate_batch
            start_time_gb = time.perf_counter()
            for i in range(int(max_batch_size / bs)):
                self.generate_batch(start_seed + i * bs, bs)
            torch.mps.synchronize()
            end_time_gb = time.perf_counter()
            duration_gb = end_time_gb - start_time_gb
            imgs_per_sec_gb = max_batch_size / duration_gb if duration_gb > 0 else float('inf')

            # 2. Test generate_from_z_batch
            torch.mps.synchronize()
            start_time_fzb = time.perf_counter()
            for i in range(int(max_batch_size / bs)):
                self.generate_from_z_batch(start_seed + i * bs, bs, False)

            torch.mps.synchronize()
            end_time_fzb = time.perf_counter()
            duration_fzb = end_time_fzb - start_time_fzb

            # 3. Test generate_from_ws_batch
            torch.mps.synchronize()
            start_time_fwb = time.perf_counter()
            for i in range(int(max_batch_size / bs)):
                self.generate_from_ws_batch(start_seed + i * bs, bs)

            torch.mps.synchronize()
            end_time_fwb = time.perf_counter()
            duration_fwb = end_time_fwb - start_time_fwb
            
            print(f"{bs:<12} | {duration_gb:<20.4f} | {duration_fzb:<18.4f} | {duration_fwb:<18.4f} | {imgs_per_sec_gb:<20.2f}")

        print("-" * len(header))
        print("Note: 'generate_batch' includes Z gen+save, W map+save, Img synth+save.")
        print("      'from_z_batch' includes Z load, W map, Img synth+save.")
        print("      'from_ws_batch' includes W load, Img synth+save.")
        return