def batch_file_prefix(start_seed, batch_size, outdir):
    return f"{outdir}/batch_{start_seed:06d}-{batch_size:06d}"

def image_prefix(seed, outdir):
    return f'{outdir}/seed{seed:06d}.png'