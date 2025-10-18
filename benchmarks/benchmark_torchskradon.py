import os

import pytest
import torch
import torchvision
import torchvision.transforms as transforms
from skimage.transform import iradon, radon
from torch.utils.data import DataLoader

from torchskradon.functional import skiradon, skradon


# Load MNIST once per module and reuse
@pytest.fixture(scope="module")
def mnist_images():
    os.makedirs(os.path.join("data"), exist_ok=True)
    dataset = torchvision.datasets.MNIST(
        root=os.path.join("data"),
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )
    loader = DataLoader(
        dataset, batch_size=len(dataset), shuffle=False, num_workers=0, pin_memory=True
    )
    images, _ = next(iter(loader))
    return images


# Parameters
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES, ids=lambda d: f"device={d}")
@pytest.mark.parametrize("name", ["torchskradon", "skimage"], ids=lambda n: f"{n}")
def benchmark_radon(benchmark, name, device, mnist_images):
    batch_size = mnist_images.size(0)
    benchmark.group = "Radon Transform on MNIST test dataset"
    device_name = "CPU" if device == "cpu" else "GPU"
    benchmark.name = f"{name} ({device_name})"

    if name == "skimage" and device != "cpu":
        pytest.skip("skimage runs on CPU only")

    if name == "torchskradon":
        x = mnist_images.to(device, non_blocking=True)

        # Warm-up
        with torch.inference_mode():
            _ = skradon(x[:1], circle=False)
        if device == "cuda":
            torch.cuda.synchronize()

        def run():
            with torch.inference_mode():
                y = skradon(x, circle=False)
            if device == "cuda":
                torch.cuda.synchronize()
            return y
    else:  # skimage
        x = mnist_images.detach().cpu().numpy()

        # Warm-up
        _ = radon(x[0, 0], circle=False)

        def run():
            for b in range(batch_size):
                _ = radon(x[b, 0], circle=False)
            return None

    benchmark(run)


@pytest.mark.parametrize("device", DEVICES, ids=lambda d: f"device={d}")
@pytest.mark.parametrize("name", ["torchskradon", "skimage"], ids=lambda n: f"{n}")
def benchmark_iradon(benchmark, name, device, mnist_images):
    batch_size = mnist_images.size(0)
    benchmark.group = "Inverse Radon Transform on MNIST test dataset"
    device_name = "CPU" if device == "cpu" else "GPU"
    benchmark.name = f"{name} ({device_name})"

    if name == "skimage" and device != "cpu":
        pytest.skip("skimage runs on CPU only")

    if name == "torchskradon":
        x = mnist_images.to(device, non_blocking=True)

        # Precompute sinograms and warm-up
        with torch.inference_mode():
            sino = skradon(x, circle=False)
            _ = skiradon(sino[:1], circle=False)
        if device == "cuda":
            torch.cuda.synchronize()

        def run():
            with torch.inference_mode():
                y = skiradon(sino, circle=False)
            if device == "cuda":
                torch.cuda.synchronize()
            return y
    else:  # skimage
        x = mnist_images.detach().cpu().numpy()

        # Precompute sinograms and warm-up
        sinos = [radon(x[b, 0], circle=False) for b in range(batch_size)]
        _ = iradon(sinos[0], circle=False)

        def run():
            for b in range(batch_size):
                _ = iradon(sinos[b], circle=False)
            return None

    benchmark(run)
