# torchskradon

(WORK IN PROGRESS) A simple and easy to use PyTorch implementation of the Radon transform and the filtered back projection.

## About
This package mimics the implementation of [`radon`](https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.radon) and [`iradon`](https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.iradon) from scikit-image. 
It supports batched, multi-channel 2D tensors as input to the Radon and inverse Radon transforms.

## Basic Usage
```python
import torch
from skimage.data import shepp_logan_phantom
from skimage.transform import rescale
from functional import torchskradon, torchskiradon

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

image = shepp_logan_phantom()
image = rescale(image, scale=0.4, mode='reflect', channel_axis=None)
image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(device)
theta = torch.linspace(0.0, 180.0, max(image.size()[2:])+1)[:-1].to(device)
sinogram = torchskradon(image, theta=theta)
reconstruction_fbp = torchskiradon(sinogram, theta=theta, filter_name='ramp')
```
## Performance 
![Example Reconstruction (Shepp-Logan)](misc/benchmark_torchskradon.png)
## Acknowledgements
This package is inspired by implementations of the Radon transform and its' inverse in [`skimage.transform`](https://github.com/scikit-image/scikit-image/tree/main/src/skimage/transform), which are based on [1-3].

## References
1. JK Romberg, ["Image Projections and the Radon Transform"](https://www.clear.rice.edu/elec431/projects96/DSP/bpanalysis.html)

2. AC Kak, M Slaney, “Principles of Computerized Tomographic Imaging”, IEEE Press 1988.

3. B.R. Ramesh, N. Srinivasa, K. Rajgopal, “An Algorithm for Computing the Discrete Radon Transform With Some Applications”, Proceedings of the Fourth IEEE Region 10 International Conference, TENCON ‘89, 1989  