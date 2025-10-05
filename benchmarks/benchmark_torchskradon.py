import pytest
import torch
from skimage.data import shepp_logan_phantom
from skimage.transform import iradon, radon, rescale

from torchskradon.functional import skiradon, skradon

# Parameters
BATCH_SIZES = [128]
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

PHANTOM = shepp_logan_phantom()[::2, ::2]
PHANTOM = rescale(PHANTOM, 0.5, order=1, mode="constant", anti_aliasing=False, channel_axis=None)
PHANTOM = torch.from_numpy(PHANTOM).unsqueeze(0).unsqueeze(0)


def _make_batch(image, batch_size, device, np=False):
    image = image.repeat(batch_size, 1, 1, 1).to(device, non_blocking=True)
    if np:
        image = image.detach().cpu().numpy()
    return image


@pytest.mark.parametrize("batch_size", BATCH_SIZES, ids=lambda b: f"batch_size={b}")
@pytest.mark.parametrize("device", DEVICES, ids=lambda d: f"device={d}")
@pytest.mark.parametrize("name", ["torchskradon", "skimage"], ids=lambda n: f"{n}")
def benchmark_radon(benchmark, name, device, batch_size):
    benchmark.group = f"Radon Transform with Batch Size {batch_size}"
    if device == "cpu":
        device_name = "CPU"
    elif device == "cuda":
        device_name = "GPU"
    benchmark.name = f"{name} ({device_name})"
    if name == "skimage" and device != "cpu":
        pytest.skip("skimage runs on CPU only")
    if name == "torchskradon":
        x = _make_batch(PHANTOM, batch_size, device)

        # Warm-up
        with torch.inference_mode():
            _ = skradon(x)
        if device == "cuda":
            torch.cuda.synchronize()

        def run():
            with torch.inference_mode():
                y = skradon(x)
            if device == "cuda":
                torch.cuda.synchronize()
            return y
    elif name == "skimage":
        x = _make_batch(PHANTOM, batch_size, device, np=True)

        # Warm-up
        _ = radon(x[0, 0])

        def run():
            for b in range(batch_size):
                _ = radon(x[b, 0])
            return None

    benchmark(run)


@pytest.mark.parametrize("batch_size", BATCH_SIZES, ids=lambda b: f"batch_size={b}")
@pytest.mark.parametrize("device", DEVICES, ids=lambda d: f"device={d}")
@pytest.mark.parametrize("name", ["torchskradon", "skimage"], ids=lambda n: f"{n}")
def benchmark_iradon(benchmark, name, device, batch_size):
    benchmark.group = f"Inverse Radon Transform with Batch Size {batch_size}"
    if device == "cpu":
        device_name = "CPU"
    elif device == "cuda":
        device_name = "GPU"
    benchmark.name = f"{name} ({device_name})"
    if name == "skimage" and device != "cpu":
        pytest.skip("skimage runs on CPU only")
    if name == "torchskradon":
        x = _make_batch(PHANTOM, batch_size, device)

        # Precompute sinograms and warm-up
        with torch.inference_mode():
            sino = skradon(x)
            _ = skiradon(sino)
        if device == "cuda":
            torch.cuda.synchronize()

        def run():
            with torch.inference_mode():
                y = skiradon(sino)
            if device == "cuda":
                torch.cuda.synchronize()
            return y
    elif name == "skimage":
        x = _make_batch(PHANTOM, batch_size, device, np=True)

        # Precompute sinograms and warm-up
        sinos = [radon(x[b, 0]) for b in range(batch_size)]
        _ = iradon(sinos[0])

        def run():
            for b in range(batch_size):
                _ = iradon(sinos[b])
            return None

    benchmark(run)
