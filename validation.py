import torch
import json
from torch.utils.data import DataLoader
from models.unet import UNet
from utils.dataset import MoNuSegDataset
from utils.metrics import iou, calculate_additional_metrics
import os

def load_paths(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
model = UNet(n_channels=3, n_classes=1)
model.load_state_dict(torch.load('trained_unet.pth'))
model = model.to(device)
model.eval()

# Setup validation dataset and DataLoader
val_image_paths = load_paths('val_images.json')
val_mask_dir = 'data/val/mask'
val_dataset = MoNuSegDataset(val_image_paths, val_mask_dir)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# Validation
total_val_iou = 0.0
total_precision = 0.0
total_recall = 0.0
total_f1 = 0.0

with torch.no_grad():
    for images, masks in val_loader:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        preds = torch.sigmoid(outputs) > 0.5
        total_val_iou += iou(preds, masks).item()

        precision, recall, f1 = calculate_additional_metrics(preds, masks)
        total_precision += precision
        total_recall += recall
        total_f1 += f1

# Calculate averages
avg_val_iou = total_val_iou / len(val_loader)
avg_precision = total_precision / len(val_loader)
avg_recall = total_recall / len(val_loader)
avg_f1 = total_f1 / len(val_loader)

# Print results
print(f'Average IoU on Validation Set: {avg_val_iou:.4f}')
print(f'Average Precision: {avg_precision:.4f}')
print(f'Average Recall: {avg_recall:.4f}')
print(f'Average F1 Score: {avg_f1:.4f}')
