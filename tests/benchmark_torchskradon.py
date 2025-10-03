import sys
import os
# Add the parent directory to the Python path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import torch
import numpy as np
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, rescale, iradon
from functional import skradon, skiradon
import matplotlib.pyplot as plt

image = shepp_logan_phantom()
image = rescale(image, scale=0.4, mode='reflect', channel_axis=None)
batches = [1, 2, 4, 8, 16, 32,  64, 128, 256, 512]
devices = ['cpu']
if torch.cuda.is_available():
    devices.append('cuda')
benchmarks_torch = np.zeros((len(batches), 2, len(devices)))
for i, batch_size in enumerate(batches):
    input_data = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
    input_data = torch.repeat_interleave(input_data, batch_size, dim=0)
    input_data = torch.repeat_interleave(input_data, 1, dim=1)
    for device in devices:
        print(f"Benchmarking skradon on {device} with batch size: {batch_size}")
        # Measure performance of skradon
        input_data = input_data.to(device)
        start_time = time.time()
        sinogram_torch = skradon(input_data)
        end_time = time.time()
        if device == 'cpu':
            benchmarks_torch[i, 0, 0] = end_time - start_time
        elif device == 'cuda':
            benchmarks_torch[i, 0, 1] = end_time - start_time
        print(f"Benchmarking skiradon on {device} with batch size: {batch_size}")
        # Measure performance of skiradon
        start_time = time.time()
        reconstruction_torch = skiradon(sinogram_torch)
        end_time = time.time()
        if device == 'cpu':
            benchmarks_torch[i, 1, 0] = end_time - start_time
        elif device == 'cuda':
            benchmarks_torch[i, 1, 1] = end_time - start_time


benchmarks_skimage = np.zeros((len(batches), 2))
for i,batch_size in enumerate(batches):
    print(f"Benchmarking (i)radon on cpu with batch size: {batch_size}")
    input_data = np.expand_dims(image, axis=(0,1))
    input_data = np.repeat(input_data, batch_size, 0)
    input_data = np.repeat(input_data, 1, 1)
    start_time = time.time()
    for batch in range(batch_size):
        for channel in range(input_data.shape[1]):
            # Measure performance of radon
            sinogram_torch = radon(input_data[batch, channel])
            end_time_radon = time.time()

            # Measure performance of iradon
            reconstruction_torch = iradon(sinogram_torch)
            end_time_iradon = time.time()
    benchmarks_skimage[i, 0] = end_time_radon - start_time
    benchmarks_skimage[i, 1] = end_time_iradon - start_time


fig, axs = plt.subplots(1, 2, figsize=(10, 4.5))
axs[0].set_title("Radon Transform")
axs[0].set_xlabel("Batch Size")
axs[0].set_ylabel("Time (s)")
axs[0].plot(batches, benchmarks_torch[:, 0, 0], label='torchskradon CPU', marker='o')
axs[0].plot(batches, benchmarks_torch[:, 0, 1], label='torchskradon GPU', marker='o')
axs[0].plot(batches, benchmarks_skimage[:, 0], label='scikit-image CPU', marker='o')
axs[0].set_yscale('log')

axs[1].set_title("Inverse Radon Transform")
axs[1].set_xlabel("Batch Size")
axs[1].plot(batches, benchmarks_torch[:, 1, 0], label='torchskradon CPU', marker='o')
axs[1].plot(batches, benchmarks_torch[:, 1, 1], label='torchskradon GPU', marker='o')
axs[1].plot(batches, benchmarks_skimage[:, 1], label='scikit-image CPU', marker='o')
axs[1].set_yscale('log')
axs[1].legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False)
plt.tight_layout()
plt.savefig(os.path.join('misc', 'benchmark_torchskradon.png'))