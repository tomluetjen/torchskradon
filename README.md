# torchskradon

[![LICENSE](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE) [![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue)](#) [![PyPI](https://img.shields.io/pypi/v/torchskradon.svg?label=PyPI&logo=pypi)](https://pypi.org/project/torchskradon/) [![CI](https://github.com/tomluetjen/torchskradon/actions/workflows/python-app.yml/badge.svg?branch=main)](https://github.com/tomluetjen/torchskradon/actions/workflows/python-app.yml) [![Coverage](https://codecov.io/gh/tomluetjen/torchskradon/branch/main/graph/badge.svg)](https://codecov.io/gh/tomluetjen/torchskradon)


## About
`torchskradon` mimics the implementation of [`radon`](https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.radon) and [`iradon`](https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.iradon) from [`scikit-image`](https://scikit-image.org). All transforms work with batched multi-channel data and are fully differentiable. This allows backpropagation through `torchskradon` functions to train neural networks or solve optimization problems with [`torch.optim`](https://docs.pytorch.org/docs/stable/optim.html) (see [examples](#examples)).

## Installation
```console
pip install torchskradon
```

## Basic Usage
```python
import torch
from skimage.data import shepp_logan_phantom
from skimage.transform import rescale
from torchskradon.functional import skradon, skiradon

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

image = shepp_logan_phantom()
image = rescale(image, scale=0.4, mode='reflect', channel_axis=None)
image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(device)
theta = torch.linspace(0.0, 180.0, max(image.size()[2:])+1)[:-1].to(device)
sinogram = skradon(image, theta=theta)
reconstruction_fbp = skiradon(sinogram, theta=theta, filter_name='ramp')
```

## Examples
For more detailed examples and use cases, see the `examples` directory:

- [`examples/plot_radon_transform.py`](examples/plot_radon_transform.py) - Basic forward and inverse Radon transforms
- [`examples/ct_model.ipynb`](examples/ct_model.ipynb) - Model-based CT reconstruction

## Accuracy
````console
----------------------- Absolute Error of Inverse Radon Transform on MNIST test dataset: 2 tests -----------------------
Name                             Min           Max          Mean        StdDev        Median           IQR      Outliers
------------------------------------------------------------------------------------------------------------------------
torchskradon (CPU)        0.0000e+00    2.0862e-06    3.1433e-07    2.1123e-07    2.5891e-07    2.5879e-07 2117701;247180
torchskradon (GPU)        0.0000e+00    1.9670e-06    1.3048e-07    1.0576e-07    1.0803e-07    1.1735e-07 1800730;379622
------------------------------------------------------------------------------------------------------------------------

--------------------------- Absolute Error of Radon Transform on MNIST test dataset: 2 tests ---------------------------
Name                             Min           Max          Mean        StdDev        Median           IQR      Outliers
------------------------------------------------------------------------------------------------------------------------
torchskradon (CPU)        0.0000e+00    6.2466e-05    1.0514e-06    1.9718e-06    0.0000e+00    1.4305e-06 7713886;6355122
torchskradon (GPU)        0.0000e+00    6.1989e-05    1.0435e-06    1.9558e-06    0.0000e+00    1.4305e-06 7633118;6271687
------------------------------------------------------------------------------------------------------------------------

Legend:
  Outliers: 1 Standard Deviation from Mean; 1.5 IQR (InterQuartile Range) from 1st Quartile and 3rd Quartile.
````
## Performance 
````console
--------------------------------------------------------- benchmark 'Inverse Radon Transform on MNIST test dataset': 3 tests ---------------------------------------------------------
Name (time in s)           Min                Max               Mean            StdDev             Median               IQR            Outliers      OPS            Rounds  Iterations
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
skimage (CPU)          18.7919 (206.14)   18.8811 (167.55)   18.8254 (197.13)   0.0346 (4.10)     18.8162 (204.12)   0.0433 (39.79)         1;0   0.0531 (0.01)          5           1
torchskradon (CPU)      5.0316 (55.20)     5.2680 (46.75)     5.1462 (53.89)    0.0983 (11.64)     5.1858 (56.26)    0.1541 (141.53)        2;0   0.1943 (0.02)          5           1
torchskradon (GPU)      0.0912 (1.0)       0.1127 (1.0)       0.0955 (1.0)      0.0084 (1.0)       0.0922 (1.0)      0.0011 (1.0)           1;1  10.4715 (1.0)           6           1
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

------------------------------------------------------------- benchmark 'Radon Transform on MNIST test dataset': 3 tests ------------------------------------------------------------
Name (time in s)           Min                Max               Mean            StdDev             Median               IQR            Outliers     OPS            Rounds  Iterations
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
skimage (CPU)          39.3553 (72.82)    40.3932 (73.88)    39.6877 (73.08)    0.4297 (178.15)   39.4695 (72.65)    0.5418 (173.05)        1;0  0.0252 (0.01)          5           1
torchskradon (CPU)     12.2509 (22.67)    12.3949 (22.67)    12.3444 (22.73)    0.0604 (25.03)    12.3783 (22.78)    0.0845 (27.00)         1;0  0.0810 (0.04)          5           1
torchskradon (GPU)      0.5405 (1.0)       0.5467 (1.0)       0.5431 (1.0)      0.0024 (1.0)       0.5433 (1.0)      0.0031 (1.0)           2;0  1.8413 (1.0)           5           1
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Legend:
  Outliers: 1 Standard Deviation from Mean; 1.5 IQR (InterQuartile Range) from 1st Quartile and 3rd Quartile.
  OPS: Operations Per Second, computed as 1 / Mean
````
You can run benchmarks yourself by running:
```console
  python -m pytest --benchmark-only --benchmark-sort="name"
```
## Other Packages
For users interested in more flexible implementations of projection transforms check out:

1. [`ASTRA Toolbox`](https://github.com/astra-toolbox/astra-toolbox)

2. [`torch-radon`](https://github.com/matteo-ronchetti/torch-radon)
## Acknowledgements
This package is inspired by implementations of the Radon transform and its' inverse in [`skimage.transform`](https://github.com/scikit-image/scikit-image/tree/main/src/skimage/transform), which are based on [1-3].

## References
1. JK Romberg, ["Image Projections and the Radon Transform"](https://www.clear.rice.edu/elec431/projects96/DSP/bpanalysis.html)

2. AC Kak, M Slaney, “Principles of Computerized Tomographic Imaging”, IEEE Press 1988.

3. B.R. Ramesh, N. Srinivasa, K. Rajgopal, “An Algorithm for Computing the Discrete Radon Transform With Some Applications”, Proceedings of the Fourth IEEE Region 10 International Conference, TENCON ‘89, 1989  