import sys
import os
# Add the parent directory to the Python path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from functional import skradon, skiradon
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load MNIST dataset
os.makedirs(os.path.join('examples', 'data'), exist_ok=True)
test_dataset = torchvision.datasets.MNIST(
    root=os.path.join('examples', 'data'), 
    train=False, 
    download=True, 
    transform=transforms.ToTensor()
)

batch_size = 8

test_loader = DataLoader(
    test_dataset, 
    batch_size=batch_size, 
    shuffle=False
)

# Load the first batch from the dataloader
data_iter = iter(test_loader)
# Here, we cheat since we are using the same discretization for simulation and reconstruction.
# But for demonstration purposes, this is fine.
images, _ = next(data_iter)

theta = torch.linspace(0., 180., 181)[:-1]
# Computing sinograms as ground truth for reconstruction
sinograms = skradon(images, theta=theta, circle=False)

print(f"Batch shape: {images.shape}")
print(f"Image data type: {images.dtype}")

class iRadonModel(torch.nn.Module):
    def __init__(self, theta):
        super(iRadonModel, self).__init__()
        self.theta = theta

    def forward(self, x):
        return skiradon(x, theta=self.theta, circle=False)

# Move data to device
images = images.to(device)
sinograms = sinograms.to(device)

# Initialize reconstruction with zeros (or random initialization)
reco = torch.zeros_like(sinograms, requires_grad=True, device=device)

# Set up optimizer
optimizer = torch.optim.Adam([reco], lr=0.1)
loss_fn = torch.nn.MSELoss()
model = iRadonModel(theta=theta).to(device)

print("Starting sinogram reconstruction...")
print(f"Target image shape: {images.shape}")
print(f"Reconstruction shape: {reco.shape}")

# Reconstruction loop
for iter in range(100):
    optimizer.zero_grad()

    # Forward pass: compute image of current reconstruction
    pred_image = model(reco)

    # Compute loss between predicted and target images
    loss_value = loss_fn(pred_image, images)

    # Backward pass
    loss_value.backward()
    
    # Update reconstruction
    optimizer.step()
    
    if iter % 10 == 0:
        print(f"Iteration {iter}, Loss: {loss_value.item():.6f}")

print("Reconstruction completed!")

# Visualize results
with torch.no_grad():
    mse = torch.nn.functional.mse_loss(reco, sinograms)
    print(f"\nReconstruction Quality:")
    print(f"MSE: {mse.item():.6f}")
    fig, axes = plt.subplots(2, batch_size)
    for i in range(batch_size):
        # Original image
        axes[0, i].imshow(sinograms[i, 0].cpu().numpy(), cmap='gray')
        axes[0, i].set_title(f'Original {i}')
        axes[0, i].axis('off')
        
        # Reconstructed image
        axes[1, i].imshow(reco[i, 0].cpu().numpy(), cmap='gray')
        axes[1, i].set_title(f'Reconstructed {i}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()

    # Sanity check: Visualize the reconstructed images from the sinograms
    fig, axes = plt.subplots(2, batch_size)
    for i in range(batch_size):
        # Original image
        axes[0, i].imshow(images[i, 0].cpu().numpy(), cmap='gray')
        axes[0, i].set_title(f'Original {i}')
        axes[0, i].axis('off')
        
        # Reconstructed image
        pred_image = model(reco)
        axes[1, i].imshow(pred_image[i, 0].cpu().numpy(), cmap='gray')
        axes[1, i].set_title(f'Reconstructed {i}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()
