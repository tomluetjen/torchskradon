from warnings import warn

import torch
import torch.nn.functional as F

from torchskradon.helpers import (
    convert_to_float,
    get_fourier_filter,
    interp,
    sinogram_circle_to_square,
    warp,
)


def skradon(image, theta=None, circle=True, preserve_range=False):
    image = convert_to_float(image, preserve_range)

    if image.ndim != 4:
        raise ValueError("The input image must be 4-D")
    if theta is None:
        theta = torch.arange(180, dtype=torch.float32)
    else:
        theta = theta.detach().clone().to(device=image.device, dtype=image.dtype)

    if circle:
        shape_min = min(image.size()[2:])
        radius = shape_min // 2
        img_shape = torch.tensor(image.size()[2:])
        x, y = torch.meshgrid(
            torch.arange(image.size()[2], device=image.device),
            torch.arange(image.size()[3], device=image.device),
            indexing="ij",
        )
        center = torch.tensor([image.size()[2] // 2, image.size()[3] // 2], device=image.device)
        coords = torch.stack([x, y], dim=0)
        dist = ((coords - center.view(-1, 1, 1)) ** 2).sum(0)
        outside_reconstruction_circle = dist > radius**2
        if torch.any(image[:, :, outside_reconstruction_circle]):
            warn(
                "Radon transform: image must be zero outside the reconstruction circle",
                UserWarning,
                2,
            )
        excess = img_shape - shape_min
        slices = tuple(
            (
                slice(
                    int(torch.ceil(e / 2).item()),
                    int(torch.ceil(e / 2).item() + shape_min),
                )
                if e.item() > 0
                else slice(None)
            )
            for e in excess
        )
        padded_image = image[:, :, slices[0], slices[1]]
    else:
        diagonal = torch.sqrt(torch.tensor(2)) * max(image.size()[2:])
        pad = [int(torch.ceil(diagonal - s)) for s in image.size()[2:]]
        new_center = [(s + p) // 2 for s, p in zip(image.size()[2:], pad, strict=False)]
        old_center = [s // 2 for s in image.size()[2:]]
        pad_before = [nc - oc for oc, nc in zip(old_center, new_center, strict=False)]
        pad_width = [(pb, p - pb) for pb, p in zip(pad_before, pad, strict=False)]
        padded_image = torch.nn.functional.pad(
            image,
            (pad_width[1][0], pad_width[1][1], pad_width[0][0], pad_width[0][1]),
            mode="constant",
            value=0,
        )
    if padded_image.size()[2] != padded_image.size()[3]:
        raise ValueError("padded_image must be a square")
    if padded_image.size()[2] % 2 == 0:
        center = 0.5
    else:
        center = 0.0
    radon_image = torch.zeros(
        (
            padded_image.size()[0],
            padded_image.size()[1],
            padded_image.size()[2],
            len(theta),
        ),
        dtype=padded_image.dtype,
        device=padded_image.device,
    )
    for i, angle in enumerate(torch.deg2rad(theta)):
        cos_a, sin_a = torch.cos(angle), torch.sin(angle)
        R = torch.tensor(
            [
                [
                    cos_a,
                    sin_a,
                    -center * (cos_a + sin_a - 1) / (padded_image.size()[2] / 2),
                ],
                [
                    -sin_a,
                    cos_a,
                    -center * (cos_a - sin_a - 1) / (padded_image.size()[3] / 2),
                ],
                [0, 0, 1],
            ],
            dtype=padded_image.dtype,
            device=padded_image.device,
        )
        rotated = warp(padded_image, R)
        radon_image[:, :, :, i] = rotated.sum(2)
    return radon_image


def skiradon(
    radon_image,
    theta=None,
    output_size=None,
    filter_name="ramp",
    interpolation="linear",
    circle=True,
    preserve_range=True,
):
    if radon_image.ndim != 4:
        raise ValueError("The input image must be 4-D")

    device = radon_image.device
    radon_image = convert_to_float(radon_image, preserve_range)
    dtype = radon_image.dtype

    if theta is None:
        theta = torch.linspace(0, 180, radon_image.size()[3] + 1, dtype=dtype, device=device)[:-1]
    else:
        theta = theta.detach().clone().to(device=device, dtype=dtype)

    angles_count = len(theta)
    if angles_count != radon_image.size()[3]:
        raise ValueError(
            "The given ``theta`` does not match the number of projections in ``radon_image``."
        )

    interpolation_types = ("linear", "nearest", "cubic")
    if interpolation not in interpolation_types:
        raise ValueError(f"Unknown interpolation: {interpolation}")

    filter_types = ("ramp", "shepp-logan", "cosine", "hamming", "hann", None)
    if filter_name not in filter_types:
        raise ValueError(f"Unknown filter: {filter_name}")

    img_shape = radon_image.size()[2]
    if output_size is None:
        if circle:
            output_size = img_shape
        else:
            output_size = int(torch.floor(torch.sqrt((torch.tensor(img_shape)) ** 2 / 2.0)).item())

    if circle:
        radon_image = sinogram_circle_to_square(radon_image)
        img_shape = radon_image.size()[2]

    projection_size_padded = max(
        64, int(2 ** torch.ceil(torch.log2(torch.tensor(2 * img_shape))).item())
    )
    pad_width = (0, projection_size_padded - img_shape)
    img = F.pad(radon_image, (0, 0, pad_width[0], pad_width[1]), mode="constant", value=0)
    fourier_filter = get_fourier_filter(projection_size_padded, filter_name, device=img.device)
    projection = torch.fft.fft(img, dim=2) * fourier_filter[None, None, :, :]
    radon_filtered = torch.real(torch.fft.ifft(projection, dim=2)[:, :, :img_shape, :])
    reconstructed = torch.zeros(
        (radon_image.size()[0], radon_image.size()[1], output_size, output_size),
        device=device,
        dtype=dtype,
    )
    radius = output_size // 2
    xpr, ypr = torch.meshgrid(
        torch.arange(output_size, device=device) - radius,
        torch.arange(output_size, device=device) - radius,
        indexing="ij",
    )

    for i, angle in enumerate(torch.deg2rad(theta)):
        col = radon_filtered[:, :, :, i]
        t = ypr * torch.cos(angle) - xpr * torch.sin(angle)
        t_flat = t.flatten().unsqueeze(0).unsqueeze(0)
        t_flat = torch.repeat_interleave(t_flat, radon_image.size()[0], dim=0)
        # We do not need to define sampling points (always cartesian)
        col_interp = interp(t_flat, img_shape, col, mode=interpolation)
        reconstructed += col_interp.view(
            radon_image.size()[0], radon_image.size()[1], output_size, output_size
        )

    if circle:
        out_reconstruction_circle = (xpr**2 + ypr**2) > radius**2
        reconstructed[:, :, out_reconstruction_circle] = 0.0

    return reconstructed * torch.pi / (2 * angles_count)
