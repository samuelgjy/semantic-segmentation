from sklearn.model_selection import train_test_split
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from models.unet import UNet
from utils.dataset import MoNuSegDataset
from utils.metrics import iou
import os
import json
import matplotlib.pyplot as plt
from torchvision import transforms
import albumentations as augment

# Set device for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LOAD DATA
image_dir = 'data/training_data/tissue_images'
annotation_dir = 'data/training_data/annotation'

image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.tif')]
annotation_paths = [os.path.join(annotation_dir, f.replace('.tif', '.xml')) for f in os.listdir(image_dir) if f.endswith('.tif')]

test_data_dir = 'data/test_data/'
test_image_paths = [os.path.join(test_data_dir, f) for f in os.listdir(test_data_dir) if f.endswith('.tif')]
test_annotation_paths = [os.path.join(test_data_dir, f.replace('.tif', '.xml')) for f in os.listdir(test_data_dir) if f.endswith('.tif')]

train_image_paths, val_image_paths, train_annotation_paths, val_annotation_paths = train_test_split(
    image_paths, annotation_paths, test_size=0.2, random_state=42)

augmentation = augment.Compose([
    augment.HorizontalFlip(p=0.5),
    augment.VerticalFlip(p=0.5),
    # Add more transformations as needed
])

train_dataset = MoNuSegDataset(
    image_paths=train_image_paths,
    annotation_paths=train_annotation_paths,
    mask_dir='data/train/mask',
    augmentation=augmentation
)

val_dataset = MoNuSegDataset(
    image_paths=val_image_paths,
    annotation_paths=val_annotation_paths,
    mask_dir='data/val/mask',
    augmentation=augmentation
)

test_dataset = MoNuSegDataset(
    image_paths=test_image_paths,
    annotation_paths=test_annotation_paths,
    mask_dir='data/test/mask',
    augmentation=augmentation
)

with open('train_images.json', 'w') as f:
    json.dump(train_image_paths, f)
with open('val_images.json', 'w') as f:
    json.dump(val_image_paths, f)
with open('test_images.json', 'w') as f:
    json.dump(test_image_paths, f)
    
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)

# Model, Loss function, Optimizer
model = UNet(n_channels=3, n_classes=1).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training and Validation Loops
num_epochs = 25
train_losses, val_losses = [], []
train_ious, val_ious = [], []

for epoch in range(num_epochs):
    # Training Loop
    model.train()
    total_train_loss = 0.0
    total_train_iou = 0.0

    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        loss = criterion(outputs, masks)
        total_train_loss += loss.item()

        # Continue with other calculations (IoU, etc.) without binarization
        total_train_iou += iou(outputs, masks).item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_losses.append(total_train_loss / len(train_loader))
    train_ious.append(total_train_iou / len(train_loader))

    # Validation Loop
    model.eval()
    total_val_loss, total_val_iou = 0.0, 0.0
    with torch.no_grad():
        for val_images, val_masks in val_loader:
            val_images, val_masks = val_images.to(device), val_masks.to(device)
            val_outputs = model(val_images)
            val_loss = criterion(val_outputs, val_masks)
            total_val_loss += val_loss.item()

            # Continue with other calculations (IoU, etc.) without binarization
            total_val_iou += iou(val_outputs, val_masks).item()

    val_losses.append(total_val_loss / len(val_loader))
    val_ious.append(total_val_iou / len(val_loader))

    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Train IoU: {train_ious[-1]:.4f}, '
          f'Val Loss: {val_losses[-1]:.4f}, Val IoU: {val_ious[-1]:.4f}')

# Save the trained model
torch.save(model.state_dict(), 'trained_unet.pth')

# Plot the training and validation loss and IoU
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), train_ious, label='Train IoU')
plt.plot(range(1, num_epochs + 1), val_ious, label='Validation IoU')
plt.title('IoU over Epochs')
plt.xlabel('Epochs')
plt.ylabel('IoU')
plt.legend()

plt.tight_layout()
plt.show()