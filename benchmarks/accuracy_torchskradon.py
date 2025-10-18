import os

import torch
import torchvision
import torchvision.transforms as transforms
from skimage.transform import iradon, radon
from torch.utils.data import DataLoader

from torchskradon.functional import skiradon, skradon


def _mnist_images():
    os.makedirs(os.path.join("benchmarks", "data"), exist_ok=True)
    dataset = torchvision.datasets.MNIST(
        root=os.path.join("benchmarks", "data"),
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

IMAGES = _mnist_images()
SINOGRAMS_SK = torch.zeros((IMAGES.size()[0], 1, 40, 180), dtype=IMAGES.dtype)
RECOS_SK = torch.zeros_like(IMAGES)


for b in range(IMAGES.size()[0]):
    # Radon transform
    SINOGRAMS_SK[b, 0] = torch.from_numpy(radon(IMAGES[b, 0].detach().cpu().numpy(), circle=False))
    # Inverse Radon transform
    RECOS_SK[b, 0] = torch.from_numpy(
        iradon(SINOGRAMS_SK[b, 0].detach().cpu().numpy(), circle=False)
    )

SINOGRAMS = torch.zeros((len(DEVICES), IMAGES.size()[0], 1, 40, 180), dtype=IMAGES.dtype)
RECOS = torch.zeros(
    (len(DEVICES), IMAGES.size()[0], IMAGES.size()[1], IMAGES.size()[2], IMAGES.size()[3]),
    dtype=IMAGES.dtype,
)


for i, device in enumerate(DEVICES):
    # Radon transform
    SINOGRAMS[i] = skradon(IMAGES.to(device), circle=False)
    # Inverse Radon transform
    RECOS[i] = skiradon(SINOGRAMS_SK.to(device), circle=False)


# We need a custom quantile function as torch.quantile works only on tensors with limited sizes
def _quantile(tensor, q, dim=None, keepdim=False):
    """
    Computes the quantile of the input tensor along the specified dimension; by mklacho (https://github.com/pytorch/pytorch/issues/64947).

    Parameters:
    tensor (torch.Tensor): The input tensor.
    q (float): The quantile to compute, should be a float between 0 and 1.
    dim (int): The dimension to reduce. If None, the tensor is flattened.
    keepdim (bool): Whether to keep the reduced dimension in the output.
    Returns:
    torch.Tensor: The quantile value(s) along the specified dimension.
    """
    assert 0 <= q <= 1, "\n\nquantile value should be a float between 0 and 1.\n\n"

    if dim is None:
        tensor = tensor.flatten()
        dim = 0

    sorted_tensor, _ = torch.sort(tensor, dim=dim)
    num_elements = sorted_tensor.size(dim)
    index = q * (num_elements - 1)
    lower_index = int(index)
    upper_index = min(lower_index + 1, num_elements - 1)
    lower_value = sorted_tensor.select(dim, lower_index)
    upper_value = sorted_tensor.select(dim, upper_index)
    # linear interpolation
    weight = index - lower_index
    quantile_value = (1 - weight) * lower_value + weight * upper_value

    return quantile_value.unsqueeze(dim) if keepdim else quantile_value


# Summaries for Absolute Error tables
def _summarize(t: torch.Tensor):
    f = t.flatten()
    minv = f.min().item()
    maxv = f.max().item()
    meanv = f.mean()
    stdv = f.std(unbiased=False)
    q1 = _quantile(f, 0.25)
    q3 = _quantile(f, 0.75)
    medianv = _quantile(f, 0.5).item()
    iqr = q3 - q1
    out_sd = ((f < meanv - stdv) | (f > meanv + stdv)).sum().item()
    out_iqr = ((f < q1 - 1.5 * iqr) | (f > q3 + 1.5 * iqr)).sum().item()
    return {
        "min": minv,
        "max": maxv,
        "mean": meanv.item(),
        "std": stdv.item(),
        "median": medianv,
        "iqr": iqr.item(),
        "out_sd": out_sd,
        "out_iqr": out_iqr,
    }


# Inverse Radon absolute errors (RECOS vs RECOS_SK)
_inv_cpu = _summarize(torch.abs(RECOS[0].detach().cpu() - RECOS_SK.detach().cpu()))
_inv_gpu = (
    _summarize(torch.abs(RECOS[1].detach().cpu() - RECOS_SK.detach().cpu()))
    if torch.cuda.is_available()
    else None
)

# Radon absolute errors (SINOGRAMS vs SINOGRAMS_SK)
_rad_cpu = _summarize(torch.abs(SINOGRAMS[0].detach().cpu() - SINOGRAMS_SK.detach().cpu()))
_rad_gpu = (
    _summarize(torch.abs(SINOGRAMS[1].detach().cpu() - SINOGRAMS_SK.detach().cpu()))
    if torch.cuda.is_available()
    else None
)

# Column widths for aligned output
NAME_W, NUM_W, OUT_W = 22, 13, 13


def _fmt_row(name: str, s: dict) -> str:
    outliers = f"{s['out_sd']};{s['out_iqr']}"
    return (
        f"{name:<{NAME_W}} "
        f"{s['min']:{NUM_W}.4e} "
        f"{s['max']:{NUM_W}.4e} "
        f"{s['mean']:{NUM_W}.4e} "
        f"{s['std']:{NUM_W}.4e} "
        f"{s['median']:{NUM_W}.4e} "
        f"{s['iqr']:{NUM_W}.4e} "
        f"{outliers:>{OUT_W}}"
    )


HEADER = (
    f"{'Name':<{NAME_W}} "
    f"{'Min':>{NUM_W}} "
    f"{'Max':>{NUM_W}} "
    f"{'Mean':>{NUM_W}} "
    f"{'StdDev':>{NUM_W}} "
    f"{'Median':>{NUM_W}} "
    f"{'IQR':>{NUM_W}} "
    f"{'Outliers':>{OUT_W}}"
)
SEP = "-" * len(HEADER)

print(f"""
----------------------- Absolute Error of Inverse Radon Transform on MNIST test dataset: {"2" if torch.cuda.is_available() else "1"} tests -----------------------
{HEADER}
{SEP}
{_fmt_row("torchskradon (CPU)", _inv_cpu)}
{_fmt_row("torchskradon (GPU)", _inv_gpu) if _inv_gpu is not None else ""}
{SEP}

--------------------------- Absolute Error of Radon Transform on MNIST test dataset: {"2" if torch.cuda.is_available() else "1"} tests ---------------------------
{HEADER}
{SEP}
{_fmt_row("torchskradon (CPU)", _rad_cpu)}
{_fmt_row("torchskradon (GPU)", _rad_gpu) if _rad_gpu is not None else ""}
{SEP}

Legend:
  Outliers: 1 Standard Deviation from Mean; 1.5 IQR (InterQuartile Range) from 1st Quartile and 3rd Quartile.
""")
