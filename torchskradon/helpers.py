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


def interp(
    x: torch.Tensor,
    xp: torch.Tensor,
    fp: torch.Tensor,
    dim: int = -1,
    extrapolate: str = "constant",
) -> torch.Tensor:
    """One-dimensional linear interpolation between monotonically increasing sample
    points, with extrapolation beyond sample points; by MoritzLange (https://github.com/pytorch/pytorch/issues/50334).
    Modified to to constant zero extrapolation.

    Returns the one-dimensional piecewise linear interpolant to a function with
    given discrete data points :math:`(xp, fp)`, evaluated at :math:`x`.

    Args:
        x: The :math:`x`-coordinates at which to evaluate the interpolated
            values.
        xp: The :math:`x`-coordinates of the data points, must be increasing.
        fp: The :math:`y`-coordinates of the data points, same shape as `xp`.
        dim: Dimension across which to interpolate.
        extrapolate: How to handle values outside the range of `xp`. Options are:
            - 'linear': Extrapolate linearly beyond range of xp values.
            - 'constant': Use the boundary value of `fp` for `x` values outside `xp`.

    Returns:
        The interpolated values, same size as `x`.
    """
    # Move the interpolation dimension to the last axis
    x = x.movedim(dim, -1)
    xp = xp.movedim(dim, -1)
    fp = fp.movedim(dim, -1)

    m = torch.diff(fp) / torch.diff(xp)  # slope
    b = fp[..., :-1] - m * xp[..., :-1]  # offset
    indices = torch.searchsorted(xp, x, right=False)

    if extrapolate == "constant":
        # Pad m and b to get constant values outside of xp range
        m = torch.cat([torch.zeros_like(m)[..., :1], m, torch.zeros_like(m)[..., :1]], dim=-1)
        # We want to zero pad with boundary values
        b = torch.cat([torch.zeros_like(fp)[..., :1], b, torch.zeros_like(fp)[..., -1:]], dim=-1)
    else:  # extrapolate == 'linear'
        indices = torch.clamp(indices - 1, 0, m.shape[-1] - 1)

    values = m.gather(-1, indices) * x + b.gather(-1, indices)

    return values.movedim(-1, dim)
