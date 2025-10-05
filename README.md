# torchskradon

(WORK IN PROGRESS) A differentiable PyTorch implementation of the forward and inverse Radon transform.

## About
`torchskradon` mimics the implementation of [`radon`](https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.radon) and [`iradon`](https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.iradon) from [`scikit-image`](https://scikit-image.org). All transforms work with batched multi-channel data and are fully differentiable. This allows backpropagation through `torchskradon` functions to train neural networks or solve simple optimization tasks (see [examples](#examples)).

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
For more detailed examples and use cases, see the `examples/` directory:

- [`examples/basic_usage.py`](examples/basic_usage.py) - Basic forward and inverse Radon transforms
- [`examples/torchskradon_model.py`](examples/torchskradon_model.py) - Model-based image reconstruction
- [`examples/torchskiradon_model.py`](examples/torchskiradon_model.py) - Model-based sinogram reconstruction

## Accuracy
The mean squared error between the transforms in `torchskradon` and their respective [`scikit-image`](https://scikit-image.org) counterparts is typically less than 0.0001% of the maximum of scikit-image's output array.

## Performance 
````console
-------------------------------------------------------- benchmark 'Inverse Radon Transform with Batch Size 128': 3 tests --------------------------------------------------------
Name (time in s)          Min               Max              Mean            StdDev            Median               IQR            Outliers      OPS            Rounds  Iterations
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
skimage (CPU)          1.0410 (27.56)    1.0545 (12.56)    1.0481 (16.83)    0.0057 (1.0)      1.0498 (16.59)    0.0095 (8.68)          2;0   0.9541 (0.06)          5           1
torchskradon (CPU)     0.6675 (17.67)    0.7644 (9.11)     0.7237 (11.62)    0.0359 (6.36)     0.7334 (11.59)    0.0410 (37.42)         2;0   1.3818 (0.09)          5           1
torchskradon (GPU)     0.0378 (1.0)      0.0839 (1.0)      0.0623 (1.0)      0.0114 (2.02)     0.0633 (1.0)      0.0011 (1.0)           4;5  16.0578 (1.0)          16           1
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

------------------------------------------------------------ benchmark 'Radon Transform with Batch Size 128': 3 tests -----------------------------------------------------------
Name (time in s)          Min               Max              Mean            StdDev            Median               IQR            Outliers     OPS            Rounds  Iterations
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
skimage (CPU)          2.0392 (8.92)     2.0689 (8.01)     2.0472 (8.35)     0.0122 (1.37)     2.0425 (8.33)     0.0080 (1.0)           1;1  0.4885 (0.12)          5           1
torchskradon (CPU)     0.6258 (2.74)     0.6447 (2.49)     0.6351 (2.59)     0.0090 (1.0)      0.6368 (2.60)     0.0171 (2.13)          3;0  1.5746 (0.39)          5           1
torchskradon (GPU)     0.2287 (1.0)      0.2584 (1.0)      0.2450 (1.0)      0.0107 (1.20)     0.2452 (1.0)      0.0110 (1.37)          2;0  4.0809 (1.0)           5           1
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Legend:
  Outliers: 1 Standard Deviation from Mean; 1.5 IQR (InterQuartile Range) from 1st Quartile and 3rd Quartile.
  OPS: Operations Per Second, computed as 1 / Mean
````
You can run benchmarks yourself by running:
```console
  python -m pytest --benchmark-only --benchmark-sort="name"
```

## Acknowledgements
This package is inspired by implementations of the Radon transform and its' inverse in [`skimage.transform`](https://github.com/scikit-image/scikit-image/tree/main/src/skimage/transform), which are based on [1-3].

## References
1. JK Romberg, ["Image Projections and the Radon Transform"](https://www.clear.rice.edu/elec431/projects96/DSP/bpanalysis.html)

2. AC Kak, M Slaney, “Principles of Computerized Tomographic Imaging”, IEEE Press 1988.

3. B.R. Ramesh, N. Srinivasa, K. Rajgopal, “An Algorithm for Computing the Discrete Radon Transform With Some Applications”, Proceedings of the Fourth IEEE Region 10 International Conference, TENCON ‘89, 1989  