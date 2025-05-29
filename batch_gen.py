from ft_utils import BatchImageGenerator

batch_generator = BatchImageGenerator.BatchImageGenerator()
print("Starting generation")
batch_generator.batch_performance_tests_until(32)
# batch_size = 64
# for i in range(100):
#     print(f"Iteration {i:02d}: {i*batch_size}")
#     batch_generator.generate_batch(batch_size*i, batch_size)
print("Finished generation")