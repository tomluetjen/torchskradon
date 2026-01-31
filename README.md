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

## Accuracy on MNIST
````console
----------------------- Absolute Error of Inverse Radon Transform on MNIST test dataset: 2 tests -----------------------
Name                             Min           Max          Mean        StdDev        Median           IQR      Outliers
------------------------------------------------------------------------------------------------------------------------
torchskradon (CPU)        0.0000e+00    2.2054e-06    3.1475e-07    2.1740e-07    2.5937e-07    2.6217e-07 2159268;272232
torchskradon (GPU)        0.0000e+00    1.9670e-06    1.4094e-07    1.2985e-07    1.0803e-07    1.2759e-07 1415341;427665
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
## Performance on MNIST
````console
--------------------------------------------------------- benchmark 'Inverse Radon Transform on MNIST test dataset': 3 tests --------------------------------------------------------
Name (time in s)           Min                Max               Mean            StdDev             Median               IQR            Outliers     OPS            Rounds  Iterations
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
skimage (CPU)          16.8473 (157.73)   16.8693 (130.60)   16.8537 (150.81)   0.0091 (1.06)     16.8511 (155.21)   0.0096 (4.75)          1;0  0.0593 (0.01)          5           1
torchskradon (CPU)      4.8955 (45.83)     5.2628 (40.75)     4.9870 (44.63)    0.1554 (18.10)     4.9192 (45.31)    0.1215 (59.92)         1;1  0.2005 (0.02)          5           1
torchskradon (GPU)      0.1068 (1.0)       0.1292 (1.0)       0.1118 (1.0)      0.0086 (1.0)       0.1086 (1.0)      0.0020 (1.0)           1;1  8.9483 (1.0)           6           1
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

------------------------------------------------------------- benchmark 'Radon Transform on MNIST test dataset': 3 tests ------------------------------------------------------------
Name (time in s)           Min                Max               Mean            StdDev             Median               IQR            Outliers     OPS            Rounds  Iterations
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
skimage (CPU)          36.6816 (78.12)    37.3837 (78.01)    36.8518 (77.68)    0.2996 (79.20)    36.7269 (77.48)    0.2385 (41.00)         1;1  0.0271 (0.01)          5           1
torchskradon (CPU)     10.4956 (22.35)    10.6328 (22.19)    10.5554 (22.25)    0.0532 (14.07)    10.5572 (22.27)    0.0763 (13.12)         2;0  0.0947 (0.04)          5           1
torchskradon (GPU)      0.4695 (1.0)       0.4792 (1.0)       0.4744 (1.0)      0.0038 (1.0)       0.4740 (1.0)      0.0058 (1.0)           2;0  2.1080 (1.0)           5           1
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