from ft_utils import BatchImageGenerator, BatchImageClassifier
from tqdm import tqdm

def gen_image_batch():
    batch_generator = BatchImageGenerator.BatchImageGenerator()
    batch_size = 64
    for i in tqdm(range(3439, round(1_000_000 / batch_size))):
        print(f"Iteration {i:05d}/{round(1_000_000 / batch_size)} {i*batch_size}")
        batch_generator.generate_batch(batch_size*i, batch_size)


def classify_image_batch():
    batch_classifier = BatchImageClassifier.BatchImageClassifier("out_batch")
    batch_size = 64
    for i in tqdm(range(5353, round(1_000_000 / batch_size))):
        print(f"Iteration {i:05d}/{round(1_000_000 / batch_size)} {i*batch_size}")
        batch_classifier.generate_image_features(batch_size*i, batch_size)

print("Starting generation")
classify_image_batch()
print("Finished generation")