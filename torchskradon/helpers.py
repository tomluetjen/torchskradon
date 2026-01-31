import torch
import torch.nn.functional as F


def convert_to_float(image, preserve_range):
    if image.dtype == torch.float16:
        return image.float()
    if preserve_range:
        # Convert image to double only if it is not single or double
        # precision float
        if image.dtype not in [torch.float32, torch.float64]:
            image = image.double()
    else:
        # Convert to float with appropriate scaling
        if not image.dtype.is_floating_point:
            if image.dtype == torch.bool:
                # Boolean: False -> 0.0, True -> 1.0
                image = image.double()
            elif image.dtype in [torch.uint8]:
                # Unsigned integers -> [0.0, 1.0]
                imax_in = torch.iinfo(image.dtype).max
                image = image / imax_in
                image = image.double()
            elif image.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
                # Signed integers -> [-1.0, 1.0] (following DirectX conversion rules)
                imax_in = torch.iinfo(image.dtype).max
                image = image / imax_in
                image = image.double()
                # Clamp the most negative value to -1.0 (DirectX style)
                torch.clamp(image, min=-1.0, out=image)
            else:
                # Fallback for other types
                image = image + 0.5
                image = image * 2 / (torch.iinfo(image.dtype).max - torch.iinfo(image.dtype).min)
                image = image.double()

    return image


def warp(image, warp_matrix):
    warp_matrix = warp_matrix[:2, :].unsqueeze(0).repeat(image.size()[0], 1, 1)  # [B,2,3]
    grid = F.affine_grid(warp_matrix, image.size(), align_corners=False)
    return F.grid_sample(image, grid, align_corners=False)


def sinogram_circle_to_square(sinogram):
    diagonal = int(torch.ceil(torch.sqrt(torch.tensor(2.0)) * sinogram.shape[2]).item())
    pad = diagonal - sinogram.shape[2]
    old_center = sinogram.shape[2] // 2
    new_center = diagonal // 2
    pad_before = new_center - old_center
    pad_width = (pad_before, pad - pad_before)
    # pad_width for torch.nn.functional.pad is (left, right, top, bottom) for 2D
    # So we pad rows: (pad_before, pad - pad_before), columns: (0, 0)
    return torch.nn.functional.pad(
        sinogram, (0, 0, pad_width[0], pad_width[1]), mode="constant", value=0
    )


def get_fourier_filter(size, filter_name, device=None, dtype=torch.float32):
    n = torch.cat(
        [
            torch.arange(1, size // 2 + 1, 2, device=device, dtype=dtype),
            torch.arange(size // 2 - 1, 0, -2, device=device, dtype=dtype),
        ]
    )
    f = torch.zeros(size, device=device, dtype=dtype)
    f[0] = 0.25
    f[1::2] = -1 / (torch.pi * n) ** 2

    fourier_filter = 2 * torch.real(torch.fft.fft(f))  # ramp filter

    if filter_name == "ramp":
        pass
    elif filter_name == "shepp-logan":
        omega = torch.pi * torch.fft.fftfreq(size, device=device, dtype=dtype)[1:]
        fourier_filter[1:] *= torch.sin(omega) / omega
    elif filter_name == "cosine":
        freq = torch.linspace(0, torch.pi, size + 1, device=device, dtype=dtype)[:-1]
        cosine_filter = torch.fft.fftshift(torch.sin(freq))
        fourier_filter *= cosine_filter
    elif filter_name == "hamming":
        hamming = torch.hamming_window(size, device=device, dtype=dtype, periodic=False)
        fourier_filter *= torch.fft.fftshift(hamming)
    elif filter_name == "hann":
        hann = torch.hann_window(size, device=device, dtype=dtype, periodic=False)
        fourier_filter *= torch.fft.fftshift(hann)
    elif filter_name is None:
        fourier_filter[:] = 1

    return fourier_filter.view(-1, 1)


def interp(x, img_shape, fp, mode="linear"):
    mode_map = {
        "nearest": "nearest",
        "linear": "bilinear",
        "cubic": "bicubic",
    }
    mode = mode_map[mode]
    N, C, W = fp.size()
    W_out = x.size()[-1]

    grid = torch.stack(
        (
            torch.zeros_like(x),
            2.0 * (x + fp.shape[-1] // 2 + 0.5) / fp.shape[-1] - 1,
        ),
        dim=-1,
    ).reshape(N, W_out, 1, 2)

    # grid_sample implemented for 4D (or 5D)
    col_in = fp.reshape(N, C, W, 1)

    out = F.grid_sample(
        col_in,
        grid,
        mode=mode,
        padding_mode="zeros",
        align_corners=False,
    )

    # We need to mask out values outside the original range to match numpy interp
    mask = (x < -img_shape // 2) | (x > img_shape - 1 - img_shape // 2)
    out = out.masked_fill(mask.unsqueeze(-1), 0)

    return out.squeeze(-2)
