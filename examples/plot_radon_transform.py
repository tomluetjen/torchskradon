import matplotlib.pyplot as plt
import torch
from skimage.data import shepp_logan_phantom
from skimage.transform import rescale

from torchskradon.functional import skiradon, skradon

# https://scikit-image.org/docs/stable/auto_examples/transform/plot_radon_transform.html

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image = shepp_logan_phantom()
image = rescale(image, scale=0.4, mode="reflect", channel_axis=None)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5))

ax1.set_title("Original")
ax1.imshow(image, cmap=plt.cm.Greys_r)

image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(device)
theta = torch.linspace(0.0, 180.0, max(image.size()[2:]) + 1)[:-1].to(device)
sinogram = skradon(image, theta=theta)
dx, dy = 0.5 * 180.0 / max(image.shape), 0.5 / sinogram.shape[0]
ax2.set_title("Radon transform\n(Sinogram)")
ax2.set_xlabel("Projection angle (deg)")
ax2.set_ylabel("Projection position (pixels)")
ax2.imshow(
    sinogram.squeeze().cpu().numpy(),
    cmap=plt.cm.Greys_r,
    extent=(-dx, 180.0 + dx, -dy, sinogram.shape[0] + dy),
    aspect="auto",
)

fig.tight_layout()
plt.show()
reconstruction_fbp = skiradon(sinogram, theta=theta, filter_name="ramp")
error = reconstruction_fbp - image
print(f"FBP rms reconstruction error: {torch.sqrt(torch.mean(error**2)).item():.3g}")

imkwargs = dict(vmin=-0.2, vmax=0.2)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5), sharex=True, sharey=True)
ax1.set_title("Reconstruction\nFiltered back projection")
ax1.imshow(reconstruction_fbp.squeeze().cpu().numpy(), cmap=plt.cm.Greys_r)
ax2.set_title("Reconstruction error\nFiltered back projection")
ax2.imshow(
    (reconstruction_fbp - image).squeeze().cpu().numpy(),
    cmap=plt.cm.Greys_r,
    **imkwargs,
)
plt.show()
