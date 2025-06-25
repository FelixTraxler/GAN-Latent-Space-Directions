from PIL import Image

BATCH_SIZE = 64

def batch_file_prefix(start_seed, batch_size, outdir):
    return f"{outdir}/batch_{start_seed:06d}-{batch_size:06d}"

def image_prefix(seed, outdir):
    return f'{outdir}/seed{seed:06d}.png'

def create_image_grid(image_paths, output_filename="grid_image.png", grid_size=(8, 8), image_dimensions=(256, 256)):
    """
    Creates a single grid image from a list of image paths.

    Args:
        image_paths (list): A list of 64 paths to 256x256 image files.
        output_filename (str): The name of the output grid image file.
        grid_size (tuple): The dimensions of the grid (rows, columns), e.g., (8, 8).
        image_dimensions (tuple): The dimensions of each individual image (width, height), e.g., (256, 256).
    """

    if len(image_paths) != grid_size[0] * grid_size[1]:
        raise ValueError(f"Expected {grid_size[0] * grid_size[1]} image paths, but got {len(image_paths)}.")

    # Calculate the total size of the grid image
    grid_width = grid_size[1] * image_dimensions[0]
    grid_height = grid_size[0] * image_dimensions[1]

    # Create a new blank image for the grid
    grid_image = Image.new('RGB', (grid_width, grid_height))

    for i, image_path in enumerate(image_paths):
        row = i // grid_size[1]
        col = i % grid_size[1]

        try:
            with Image.open(image_path) as img:
                # Resize if necessary (though the problem states 256x256, it's good practice)
                img = img.resize(image_dimensions)
                
                # Paste the image into the grid
                grid_image.paste(img, (col * image_dimensions[0], row * image_dimensions[1]))
        except FileNotFoundError:
            print(f"Warning: Image not found at {image_path}. Skipping this image.")
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    # Save the final grid image
    grid_image.save(output_filename)
    print(f"Grid image saved as '{output_filename}'")

def create_grid_from_batch(start_seed, batch_size, dir):
    filenames = []
    for i in range(batch_size):
        current_seed = start_seed + i
        filenames.append(image_prefix(current_seed, dir))
    return create_image_grid(filenames)


def diff_percentage(val1, val2):
    return 100 * (val1 - val2) / val2