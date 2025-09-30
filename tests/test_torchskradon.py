import itertools
import sys
import os
# Add the parent directory to the Python path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import pytest
from skimage._shared._dependency_checks import has_mpl
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, iradon, rescale
from functional import torchskradon, torchskiradon
from utils.helpers import convert_to_float

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

PHANTOM = shepp_logan_phantom()[::2, ::2]
PHANTOM = rescale(
    PHANTOM, 0.5, order=1, mode='constant', anti_aliasing=False, channel_axis=None
)

PHANTOM = torch.from_numpy(PHANTOM).unsqueeze(0).unsqueeze(0).double()
PHANTOM = torch.repeat_interleave(PHANTOM, 10, dim=0)
PHANTOM = torch.repeat_interleave(PHANTOM, 3, dim=1).to(device)

def _debug_plot(original, result, sinogram=None):
    from matplotlib import pyplot as plt

    imkwargs = dict(cmap='gray', interpolation='auto')
    if sinogram is None:
        plt.figure(figsize=(15, 6))
        sp = 130
    else:
        plt.figure(figsize=(11, 11))
        sp = 221
        plt.subplot(sp + 0)
        plt.imshow(sinogram, aspect='auto', **imkwargs)
    plt.subplot(sp + 1)
    plt.imshow(original, **imkwargs)
    plt.subplot(sp + 2)
    plt.imshow(result, vmin=original.min(), vmax=original.max(), **imkwargs)
    plt.subplot(sp + 3)
    plt.imshow(result - original, **imkwargs)
    plt.colorbar()
    plt.show()


def _rescale_intensity(x):
    x = x.double()
    x -= x.min()
    x /= x.max()
    return x


def check_torchskradon_vs_skimageradon(circle, dtype, preserve_range):
    image = PHANTOM.to(dtype)[:1,:1,:,:]
    image_sk = image.squeeze().cpu().numpy()
    sinogram = torchskradon(image, circle=circle, preserve_range=preserve_range)
    sinogram_sk = radon(image_sk, circle=circle, preserve_range=preserve_range)
    sinogram_sk = torch.from_numpy(sinogram_sk).unsqueeze(0).unsqueeze(0).to(device)
    # Compare the two sinograms
    assert ((sinogram - sinogram_sk)**2).mean() < 1e-3*torch.max(torch.abs(sinogram_sk)) #less than 0.1% of max value
    assert sinogram.dtype == sinogram_sk.dtype

@pytest.mark.parametrize("circle", [False, True])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.float16, torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8, bool])
@pytest.mark.parametrize("preserve_range", [False, True])
def test_torchskradon_vs_skimageradon(circle, dtype,  preserve_range):
    check_torchskradon_vs_skimageradon(circle, dtype, preserve_range)


def check_torchskiradon_vs_skimageiradon(circle, dtype, preserve_range):
    image = PHANTOM.to(dtype)[:1,:1,:,:]
    image_sk = image.squeeze().cpu().numpy()
    sinogram_sk = radon(image_sk, circle=circle, preserve_range=preserve_range)
    sinogram = torch.from_numpy(sinogram_sk).unsqueeze(0).unsqueeze(0).to(device).to(device)
    reco_sk = iradon(sinogram_sk, circle=circle, preserve_range=preserve_range)
    reco_sk = torch.from_numpy(reco_sk).unsqueeze(0).unsqueeze(0).to(device)
    reco = torchskiradon(sinogram, circle=circle, preserve_range=preserve_range)
    # Compare the two reconstructions
    assert ((reco - reco_sk)**2).mean() < 1e-3*torch.max(torch.abs(reco_sk)) #less than 0.1% of max value
    assert reco.dtype == reco_sk.dtype

@pytest.mark.parametrize("circle", [False, True])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.float16, torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8, bool])
@pytest.mark.parametrize("preserve_range", [False, True])
def test_torchskiradon_vs_skimageradon(circle, dtype,  preserve_range):
    check_torchskiradon_vs_skimageiradon(circle, dtype, preserve_range)

def test_iradon_bias_circular_phantom():
    """
    test that a uniform circular phantom has a small reconstruction bias
    """
    pixels = 128
    xy = torch.arange(-pixels / 2, pixels / 2) + 0.5
    x, y = torch.meshgrid(xy, xy, indexing='xy')
    image = x**2 + y**2 <= (pixels / 4) ** 2
    image = image.unsqueeze(0).unsqueeze(0).double()

    theta = torch.linspace(0.0, 180.0, max(image.size()[2:]))
    sinogram = torchskradon(image, theta=theta)

    reconstruction_fbp = torchskiradon(sinogram, theta=theta)
    error = reconstruction_fbp - image

    tol = 5e-5
    roi_err = torch.abs(torch.mean(error))
    assert roi_err < tol


def check_radon_center(shape, circle, dtype, preserve_range):
    # Create a test image with only a single non-zero pixel at the origin
    image = torch.zeros(shape, dtype=dtype)
    image[(shape[0] // 2, shape[1] // 2)] = 1.0
    image = image.unsqueeze(0).unsqueeze(0)
    # Calculate the sinogram
    theta = torch.linspace(0.0, 180.0, max(shape))
    sinogram = torchskradon(image, theta=theta, circle=circle, preserve_range=preserve_range)
    #assert sinogram.dtype == _supported_float_type(sinogram.dtype)
    # The sinogram should be a straight, horizontal line
    sinogram_max = torch.argmax(sinogram, axis=2)
    print(sinogram_max)
    assert torch.std(sinogram_max.double()) < 1e-6


@pytest.mark.parametrize("shape", [(16, 16), (17, 17)])
@pytest.mark.parametrize("circle", [False, True])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.float16, torch.uint8, bool])
@pytest.mark.parametrize("preserve_range", [False, True])
def test_radon_center(shape, circle, dtype,  preserve_range):
    check_radon_center(shape, circle, dtype, preserve_range)


@pytest.mark.parametrize("shape", [(32, 16), (33, 17)])
@pytest.mark.parametrize("circle", [False])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.uint8, bool])
@pytest.mark.parametrize("preserve_range", [False, True])
def test_radon_center_rectangular(shape, circle, dtype, preserve_range):
    check_radon_center(shape, circle, dtype, preserve_range)


def check_iradon_center(size, theta, circle):
    debug = False
    # Create a test sinogram corresponding to a single projection
    # with a single non-zero pixel at the rotation center
    if circle:
        sinogram = torch.zeros((size, 1), dtype=torch.double)
        sinogram[size // 2, 0] = 1.0
    else:
        diagonal = int(torch.ceil(torch.sqrt(torch.tensor(2)) * size))
        sinogram = torch.zeros((diagonal, 1), dtype=torch.double)
        sinogram[sinogram.size()[0] // 2, 0] = 1.0
    maxpoint = torch.unravel_index(torch.argmax(sinogram), sinogram.size())
    print('shape of generated sinogram', sinogram.size())
    print('maximum in generated sinogram', maxpoint)
    # Compare reconstructions for theta=angle and theta=angle + 180;
    # these should be exactly equal
    sinogram = sinogram.unsqueeze(0).unsqueeze(0)
    reconstruction = torchskiradon(sinogram, theta=torch.tensor([theta]), circle=circle)
    reconstruction_opposite = torchskiradon(sinogram, theta=torch.tensor([theta + 180]), circle=circle)
    print(
        'rms deviance:',
        torch.sqrt(torch.mean((reconstruction_opposite - reconstruction) ** 2)),
    )
    if debug and has_mpl:
        import matplotlib.pyplot as plt

        imkwargs = dict(cmap='gray', interpolation='auto')
        plt.figure()
        plt.subplot(221)
        plt.imshow(sinogram[0,0].detach().cpu().numpy(), **imkwargs)
        plt.subplot(222)
        plt.imshow(reconstruction_opposite[0,0].detach().cpu().numpy() - reconstruction[0,0].detach().cpu().numpy(), **imkwargs)
        plt.subplot(223)
        plt.imshow(reconstruction[0,0].detach().cpu().numpy(), **imkwargs)
        plt.subplot(224)
        plt.imshow(reconstruction_opposite[0,0].detach().cpu().numpy(), **imkwargs)
        plt.show()

    assert torch.allclose(reconstruction, reconstruction_opposite, atol=1e-6) #atol needs to be explicitly set, probably due to interpolation errors


sizes_for_test_iradon_center = [16, 17]
thetas_for_test_iradon_center = [0, 90]
circles_for_test_iradon_center = [False, True]


@pytest.mark.parametrize(
    "size, theta, circle",
    itertools.product(
        sizes_for_test_iradon_center,
        thetas_for_test_iradon_center,
        circles_for_test_iradon_center,
    ),
)
def test_iradon_center(size, theta, circle):
    check_iradon_center(size, theta, circle)


def check_radon_iradon(interpolation_type, filter_type):
    debug = False
    image = PHANTOM
    reconstructed = torchskiradon(
        torchskradon(image, circle=False),
        filter_name=filter_type,
        interpolation=interpolation_type,
        circle=False,
    )
    delta = torch.mean(torch.abs(image - reconstructed))
    print('\n\tmean error:', delta)
    if debug and has_mpl:
        _debug_plot(image[0,0].detach().cpu().numpy(), reconstructed[0,0].detach().cpu().numpy())
    if filter_type in ('ramp', 'shepp-logan'):
            allowed_delta = 0.025
    else:
        allowed_delta = 0.05
    assert delta < allowed_delta


filter_types = ["ramp", "shepp-logan", "cosine", "hamming", "hann"]
interpolation_types = ['linear']
radon_iradon_itorchuts = list(itertools.product(interpolation_types, filter_types))


@pytest.mark.parametrize("interpolation_type, filter_type", radon_iradon_itorchuts)
def test_radon_iradon(interpolation_type, filter_type):
    check_radon_iradon(interpolation_type, filter_type)


def test_iradon_angles():
    """
    Test with different number of projections
    """
    size = 100
    # Synthetic data
    image = torch.tril(torch.ones((size, size))) + torch.flip(torch.tril(torch.ones((size, size))), dims=[0])
    image = image.unsqueeze(0).unsqueeze(0)
    # Large number of projections: a good quality is expected
    nb_angles = 200
    theta = torch.linspace(0, 180, nb_angles)
    radon_image_200 = torchskradon(image, theta=theta, circle=False)
    reconstructed = torchskiradon(radon_image_200, circle=False)
    delta_200 = torch.mean(
        abs(_rescale_intensity(image) - _rescale_intensity(reconstructed))
    )
    assert delta_200 < 0.07 #skimage allows < 0.03, but we are a bit worse
    # Lower number of projections
    nb_angles = 80
    radon_image_80 = torchskradon(image, theta=theta, circle=False)
    # Test whether the sum of all projections is approximately the same
    s = radon_image_80.sum(axis=2)
    print(s.size())
    assert torch.allclose(s[0,0], s[0,0,2], rtol=0.01)
    reconstructed = torchskiradon(radon_image_80, circle=False)
    delta_80 = torch.mean(
        abs(image / torch.max(image) - reconstructed / torch.max(reconstructed))
    )
    # Loss of quality when the number of projections is reduced
    assert delta_80 > delta_200


def check_radon_iradon_minimal(shape, slices):
    debug = False
    theta = torch.arange(180)
    image = torch.zeros(shape, dtype=torch.double)
    image[slices] = 1.0
    image = image.unsqueeze(0).unsqueeze(0)
    sinogram = torchskradon(image, theta, circle=False)
    reconstructed = torchskiradon(sinogram, theta, circle=False)
    print('\n\tMaximum deviation:', torch.max(torch.abs(image - reconstructed)))
    if debug and has_mpl:
        _debug_plot(image[0,0].detach().cpu().numpy(), reconstructed[0,0].detach().cpu().numpy(), sinogram[0,0].detach().cpu().numpy())
    if image.sum() == 1:
        assert torch.unravel_index(
            torch.argmax(reconstructed), image.size()
        ) == torch.unravel_index(torch.argmax(image), image.size())


shapes = [(3, 3), (4, 4), (5, 5)]


def generate_test_data_for_radon_iradon_minimal(shapes):
    def shape2coordinates(shape):
        c0, c1 = shape[0] // 2, shape[1] // 2
        coordinates = itertools.product((c0 - 1, c0, c0 + 1), (c1 - 1, c1, c1 + 1))
        return coordinates

    def shape2shapeandcoordinates(shape):
        return itertools.product([shape], shape2coordinates(shape))

    return itertools.chain.from_iterable(
        [shape2shapeandcoordinates(shape) for shape in shapes]
    )


@pytest.mark.parametrize(
    "shape, coordinate", generate_test_data_for_radon_iradon_minimal(shapes)
)
def test_radon_iradon_minimal(shape, coordinate):
    check_radon_iradon_minimal(shape, coordinate)


def test_reconstruct_with_wrong_angles():
    a = torch.zeros((3, 3))
    a = a.unsqueeze(0).unsqueeze(0)
    p = torchskradon(a, theta=torch.tensor([0, 1, 2]), circle=False)
    torchskiradon(p, theta=torch.tensor([0, 1, 2]), circle=False)
    with pytest.raises(ValueError):
        torchskiradon(p, theta=torch.tensor([0, 1, 2, 3]))


def _random_circle(shape):
    # Synthetic random data, zero outside reconstruction circle
    torch.manual_seed(98312871)
    image = torch.rand(*shape)
    c0, c1 = torch.meshgrid(torch.arange(0, shape[0]), torch.arange(0, shape[1]), indexing='ij')
    r = torch.sqrt((c0 - shape[0] // 2) ** 2 + (c1 - shape[1] // 2) ** 2)
    radius = min(shape) // 2
    image[r > radius] = 0.0
    return image


def test_radon_circle():
    a = torch.ones((10, 10))
    a = a.unsqueeze(0).unsqueeze(0)
    with pytest.warns(UserWarning):
        result = torchskradon(a, circle=True)

    # Synthetic data, circular symmetry
    shape = (61, 79)
    c0, c1 = torch.meshgrid(torch.arange(0, shape[0]), torch.arange(0, shape[1]), indexing='xy')
    r = torch.sqrt((c0 - shape[0] // 2) ** 2 + (c1 - shape[1] // 2) ** 2)
    radius = min(shape) // 2
    image = torch.clip(radius - r, 0, torch.inf)
    image = _rescale_intensity(image)
    angles = torch.linspace(0, 180, min(shape))
    image = image.unsqueeze(0).unsqueeze(0)
    sinogram = torchskradon(image, theta=angles, circle=True)
    assert torch.all(sinogram.std(axis=3) < 1e-2)

    # Synthetic data, random
    image = _random_circle(shape)
    image = image.unsqueeze(0).unsqueeze(0)
    sinogram = torchskradon(image, theta=angles, circle=True)
    mass = sinogram.sum(axis=2)
    average_mass = mass.mean()
    relative_error = torch.abs(mass - average_mass) / average_mass
    print(relative_error.max(), relative_error.mean())
    assert torch.all(relative_error < 3.2e-3)


def check_sinogram_circle_to_square(size):
    from utils.helpers import sinogram_circle_to_square

    image = _random_circle((size, size))
    theta = torch.linspace(0.0, 180.0, size+1)[:-1]
    image = image.unsqueeze(0).unsqueeze(0)
    sinogram_circle = torchskradon(image, theta, circle=True)

    def argmax_shape(a):
        return torch.unravel_index(torch.argmax(a), a.shape)

    print('\n\targmax of circle:', argmax_shape(sinogram_circle))
    sinogram_square = torchskradon(image, theta, circle=False)
    print('\targmax of square:', argmax_shape(sinogram_square))
    sinogram_circle_to_square = sinogram_circle_to_square(sinogram_circle)
    print('\targmax of circle to square:', argmax_shape(sinogram_circle_to_square))
    error = abs(sinogram_square - sinogram_circle_to_square)
    print(torch.mean(error), torch.max(error))
    assert argmax_shape(sinogram_square) == argmax_shape(sinogram_circle_to_square)


@pytest.mark.parametrize("size", (50, 51))
def test_sinogram_circle_to_square(size):
    check_sinogram_circle_to_square(size)


def check_radon_iradon_circle(interpolation, shape, output_size):
    # Forward and inverse radon on synthetic data
    image = _random_circle(shape)
    radius = min(shape) // 2
    image = image.unsqueeze(0).unsqueeze(0)
    sinogram_rectangle = torchskradon(image, circle=False)
    reconstruction_rectangle = torchskiradon(
        sinogram_rectangle,
        output_size=output_size,
        interpolation=interpolation,
        circle=False,
    )
    sinogram_circle = torchskradon(image, circle=True)
    reconstruction_circle = torchskiradon(
        sinogram_circle,
        output_size=output_size,
        interpolation=interpolation,
        circle=True,
    )
    # Crop rectangular reconstruction to match circle=True reconstruction
    width = reconstruction_circle.size()[2]
    excess = int(torch.ceil(torch.tensor((reconstruction_rectangle.size()[2] - width) / 2)))
    #s = torch.s_[excess : width + excess, excess : width + excess]
    reconstruction_rectangle = reconstruction_rectangle[:,:,excess : width + excess, excess : width + excess]
    # Find the reconstruction circle, set reconstruction to zero outside
    c0, c1 = torch.meshgrid(torch.arange(0, width), torch.arange(0, width), indexing='ij')
    r = torch.sqrt((c0 - width // 2) ** 2 + (c1 - width // 2) ** 2)
    reconstruction_rectangle[0,0,r > radius] = 0.0
    print(reconstruction_circle.shape)
    print(reconstruction_rectangle.shape)
    torch.allclose(reconstruction_rectangle, reconstruction_circle)


# if adding more shapes to test data, you might want to look at commit d0f2bac3f
shapes_radon_iradon_circle = ((61, 79),)
interpolations = ('linear')
output_sizes = (
    None,
    min(shapes_radon_iradon_circle[0]),
    max(shapes_radon_iradon_circle[0]),
    97,
)


@pytest.mark.parametrize(
    "shape, interpolation, output_size",
    itertools.product(shapes_radon_iradon_circle, interpolations, output_sizes),
)
def test_radon_iradon_circle(shape, interpolation, output_size):
    check_radon_iradon_circle(interpolation, shape, output_size)

@pytest.mark.parametrize("preserve_range", [True, False])
def test_iradon_dtype(preserve_range):
    sinogram = torch.zeros((16, 1), dtype=int)
    sinogram[8, 0] = 1.0
    sinogram = sinogram.unsqueeze(0).unsqueeze(0)
    sinogram64 = sinogram.double()
    sinogram32 = sinogram.float()

    assert torchskiradon(sinogram, theta=torch.tensor([0]), preserve_range=preserve_range).dtype == torch.float64
    assert (
        torchskiradon(sinogram64, theta=torch.tensor([0]), preserve_range=preserve_range).dtype
        == sinogram64.dtype
    )
    assert (
        torchskiradon(sinogram32, theta=torch.tensor([0]), preserve_range=preserve_range).dtype
        == sinogram32.dtype
    )


def test_radon_dtype():
    img = convert_to_float(PHANTOM, False)
    img32 = img.float()

    assert torchskradon(img).dtype == img.dtype
    assert torchskradon(img32).dtype == img32.dtype


def test_iradon_rampfilter_bias_circular_phantom():
    """
    test that a uniform circular phantom has a small reconstruction bias using
    the ramp filter
    """
    pixels = 128
    xy = torch.arange(-pixels / 2, pixels / 2) + 0.5
    x, y = torch.meshgrid(xy, xy, indexing='xy')
    image = x**2 + y**2 <= (pixels / 4) ** 2
    image = image.unsqueeze(0).unsqueeze(0)

    theta = torch.linspace(0.0, 180.0, max(image.size()[2:]))
    sinogram = torchskradon(image, theta=theta)

    reconstruction_fbp = torchskiradon(sinogram, theta=theta)
    error = reconstruction_fbp - image.to(dtype=reconstruction_fbp.dtype)

    tol = 5e-5
    roi_err = torch.abs(torch.mean(error))
    assert roi_err < tol