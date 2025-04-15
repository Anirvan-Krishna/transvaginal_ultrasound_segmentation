import os
import random
import numpy as np
from tqdm import tqdm
import math
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image
import matplotlib.pyplot as plt

import timm
from timm.models.vision_transformer import VisionTransformer

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# Configuration
class Config:
    # General
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 4
    
    # Data
    unlabeled_data_dir = "./train/unlabeled_data/images"
    labeled_data_dir = "./train/labeled_data"
    crop_size = 224  # Size of random crops
    num_crops = 4  # Number of crops per image for both training and inference
    
    # SimMIM pretraining
    mask_ratio = 0.6  # Percentage of patches to mask
    patch_size = 16
    
    # Pretraining
    pretrain_batch_size = 16
    pretrain_epochs = 50
    pretrain_lr = 3e-4
    
    # Segmentation
    seg_batch_size = 8
    seg_epochs = 20
    seg_lr = 3e-4
    num_classes = 3  # Multi-class segmentation (adjust as needed for your dataset)
    
    # Loss weights
    dice_weight = 1.0
    ce_weight = 1.0
    hausdorff_weight = 1.0

# Helper functions for random crops
def get_random_crop_params(img_size, crop_size):
    h, w = img_size
    th, tw = crop_size, crop_size
    if h <= th or w <= tw:
        # If image is smaller than crop size, don't crop
        return 0, 0, h, w
    
    i = random.randint(0, h - th)
    j = random.randint(0, w - tw)
    return i, j, th, tw

def get_center_crop_params(img_size, crop_size):
    h, w = img_size
    th, tw = crop_size, crop_size
    if h <= th or w <= tw:
        # If image is smaller than crop size, don't crop
        return 0, 0, h, w
    
    i = (h - th) // 2
    j = (w - tw) // 2
    return i, j, th, tw

def random_crop_tensor(img, crop_size):
    i, j, h, w = get_random_crop_params(img.shape[-2:], crop_size)
    return img[..., i:i+h, j:j+w], (i, j, h, w)

def get_multiple_random_crops(img, mask=None, crop_size=224, num_crops=4):
    """Get multiple random crops from image and corresponding mask if available"""
    crops = []
    crop_params = []
    mask_crops = [] if mask is not None else None
    
    for _ in range(num_crops):
        if isinstance(img, torch.Tensor):
            img_crop, params = random_crop_tensor(img, crop_size)
            crops.append(img_crop)
            crop_params.append(params)
            if mask is not None:
                mask_crops.append(mask[..., params[0]:params[0]+params[2], params[1]:params[1]+params[3]])
        else:  # PIL Image
            i, j, h, w = get_random_crop_params(img.size[::-1], crop_size)
            crops.append(TF.crop(img, i, j, h, w))
            crop_params.append((i, j, h, w))
            if mask is not None:
                mask_crops.append(TF.crop(mask, i, j, h, w))
    
    return crops, mask_crops, crop_params

# Dataset classes
class UnlabeledDataset(Dataset):
    def __init__(self, data_dir, crop_size=224, num_crops=4, transform=None):
        self.data_dir = data_dir
        self.image_files = [f for f in os.listdir(data_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform
        self.crop_size = crop_size
        self.num_crops = num_crops
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Get multiple random crops
        crops, _, _ = get_multiple_random_crops(image, None, self.crop_size, self.num_crops)
        
        # Apply additional transforms to crops if needed and convert to tensors
        if not isinstance(crops[0], torch.Tensor):
            crops = [TF.to_tensor(crop) for crop in crops]
        
        # Stack crops along batch dimension
        crops = torch.stack(crops)
        
        return crops

class LabeledDataset(Dataset):
    def __init__(self, data_dir, crop_size=224, num_crops=4, transform=None):
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, "images")
        self.mask_dir = os.path.join(data_dir, "masks")
        self.image_files = [f for f in os.listdir(self.image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform
        self.crop_size = crop_size
        self.num_crops = num_crops
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name.split('.')[0] + '.png')
        
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # Load as grayscale
        
        if self.transform:
            image = self.transform(image)
            mask = TF.to_tensor(mask)
            
            # Convert mask to class indices (assuming values from 0 to num_classes-1)
            mask = mask * (Config.num_classes - 1)
            mask = mask.long().squeeze(0)
        
        # Get multiple random crops with same coordinates for both image and mask
        image_crops, mask_crops, _ = get_multiple_random_crops(image, mask, self.crop_size, self.num_crops)
        
        # Convert to tensors if needed
        if not isinstance(image_crops[0], torch.Tensor):
            image_crops = [TF.to_tensor(crop) for crop in image_crops]
            
        if not isinstance(mask_crops[0], torch.Tensor):
            mask_crops = [TF.to_tensor(crop) * (Config.num_classes - 1) for crop in mask_crops]
            mask_crops = [m.long().squeeze(0) for m in mask_crops]
        
        # Stack crops along batch dimension
        image_crops = torch.stack(image_crops)
        mask_crops = torch.stack(mask_crops)
        
        return image_crops, mask_crops

# SimMIM implementation
class SimMIM(nn.Module):
    def __init__(self, encoder, patch_size=16):
        super().__init__()
        self.encoder = encoder
        self.patch_size = patch_size
        self.embed_dim = encoder.embed_dim
        
        # Reconstruction decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.embed_dim, patch_size * patch_size * 3),
            nn.GELU(),
            nn.Linear(patch_size * patch_size * 3, patch_size * patch_size * 3)
        )
    
    def forward(self, x, mask=None):
        batch_size = x.shape[0]
        img_size = x.shape[-1]
        num_patches = (img_size // self.patch_size) ** 2
        
        # Generate random mask if not provided
        if mask is None:
            mask = self.generate_mask(batch_size, num_patches, Config.mask_ratio)
        
        # Save original patches for reconstruction loss
        patches = self.patchify(x)
        
        # Apply mask to input image
        x_masked = self.mask_image(x, mask)
        
        # Get encoder features
        features = self.encoder.forward_features(x_masked)
        
        # Skip the class token for reconstruction
        patch_features = features[:, 1:]  # Remove CLS token
        
        # Decode features to patches
        pred_patches = self.decoder(patch_features)
        
        # Calculate reconstruction loss only for masked patches
        loss = self.calculate_loss(pred_patches, patches, mask)
        
        return loss
    
    def patchify(self, imgs):
        """
        Convert images to patches
        imgs: (B, 3, H, W)
        x: (B, L, patch_size**2 * 3)
        """
        p = self.patch_size
        h = w = imgs.shape[-1] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x
    
    def unpatchify(self, x, img_size):
        """
        Reverse patchify operation
        x: (B, L, patch_size**2 * 3)
        imgs: (B, 3, H, W)
        """
        p = self.patch_size
        h = w = img_size // p
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, w * p))
        return imgs
    
    def generate_mask(self, batch_size, num_patches, mask_ratio):
        """Generate random mask with specified ratio"""
        mask = torch.zeros(batch_size, num_patches, device=next(self.parameters()).device)
        mask_indices = torch.rand(batch_size, num_patches, device=mask.device).argsort(dim=1)
        mask_len = int(num_patches * mask_ratio)
        
        for i in range(batch_size):
            mask[i, mask_indices[i, :mask_len]] = 1
            
        return mask.bool()
    
    def mask_image(self, imgs, mask):
        """Apply mask to image"""
        p = self.patch_size
        img_size = imgs.shape[-1]
        h = w = img_size // p
        
        imgs_patches = self.patchify(imgs)
        batch_range = torch.arange(imgs.shape[0], device=imgs.device)[:, None]
        mask_flatten = mask.reshape(mask.shape[0], -1)
        
        # Replace masked patches with zeros
        masked_patches = imgs_patches.clone()
        masked_patches[batch_range, mask_flatten] = 0
        
        # Reconstruct masked image
        masked_imgs = self.unpatchify(masked_patches, img_size)
        
        return masked_imgs
    
    def calculate_loss(self, pred_patches, target_patches, mask):
        """Calculate MSE loss only on masked patches"""
        mask_flatten = mask.reshape(mask.shape[0], -1)
        
        # Only compute loss on masked patches
        # Extract values using mask
        batch_range = torch.arange(pred_patches.shape[0], device=pred_patches.device)[:, None]
        pred_values = pred_patches[batch_range, mask_flatten]
        target_values = target_patches[batch_range, mask_flatten]
        
        # Compute MSE loss
        loss = F.mse_loss(pred_values, target_values)
        
        return loss

# Double Convolution block for UNet
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

# UNet segmentation head
class UNetSegHead(nn.Module):
    def __init__(self, backbone_dim, num_classes):
        super().__init__()
        
        self.backbone_dim = backbone_dim
        factor = 2  # factor for decoder path
        
        # Upsampling path (decoder)
        self.up1 = nn.ConvTranspose2d(backbone_dim, 512 // factor, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(512, 512 // factor)
        
        self.up2 = nn.ConvTranspose2d(512 // factor, 256 // factor, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(256 // factor, 256 // factor)
        
        self.up3 = nn.ConvTranspose2d(256 // factor, 128 // factor, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(128 // factor, 128 // factor)
        
        self.up4 = nn.ConvTranspose2d(128 // factor, 64, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(64, 64)
        
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
    
    def forward(self, x):
        # x comes from the backbone and has shape [B, embed_dim, H/16, W/16]
        
        # Decoder path
        x = self.up1(x)
        x = self.conv1(x)
        
        x = self.up2(x)
        x = self.conv2(x)
        
        x = self.up3(x)
        x = self.conv3(x)
        
        x = self.up4(x)
        x = self.conv4(x)
        
        # Final 1x1 convolution to get classes
        x = self.final_conv(x)
        
        return x

# ViT with segmentation head
class ViTSegmentation(nn.Module):
    def __init__(self, vit_model, num_classes, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.vit = vit_model
        self.embed_dim = vit_model.embed_dim
        
        # Take features from ViT and reshape to spatial features
        self.reshape_features = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU()
        )
        
        # Segmentation head
        self.seg_head = UNetSegHead(self.embed_dim, num_classes)
        
    def forward(self, x):
        batch_size, _, h, w = x.shape
        
        # Get patch tokens from ViT
        features = self.vit.forward_features(x)  # [B, num_patches+1, embed_dim]
        
        # Remove CLS token and reshape to spatial form
        patch_tokens = features[:, 1:]  # [B, num_patches, embed_dim]
        
        # Calculate spatial dimensions
        h_patches = w_patches = int(math.sqrt(patch_tokens.shape[1]))
        
        # Reshape patch tokens to spatial form [B, embed_dim, h_patches, w_patches]
        spatial_tokens = patch_tokens.reshape(batch_size, h_patches, w_patches, self.embed_dim)
        spatial_tokens = spatial_tokens.permute(0, 3, 1, 2)
        
        # Apply segmentation head
        logits = self.seg_head(spatial_tokens)
        
        return logits

# Loss functions
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        batch_size = logits.shape[0]
        
        # Convert targets to one-hot encoding
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
        
        # Apply softmax to logits
        probs = F.softmax(logits, dim=1)
        
        # Calculate Dice coefficient for each class
        dice_score = 0
        for cls in range(num_classes):
            pred_cls = probs[:, cls]
            target_cls = targets_one_hot[:, cls]
            
            intersection = (pred_cls * target_cls).sum(dim=(1, 2))
            union = pred_cls.sum(dim=(1, 2)) + target_cls.sum(dim=(1, 2))
            
            dice_cls = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_score += dice_cls.mean()
            
        # Average Dice score across all classes
        dice_score /= num_classes
        
        return 1 - dice_score

class HausdorffDTLoss(nn.Module):
    """
    Hausdorff Distance Loss based on distance transform
    """
    def __init__(self, alpha=2.0):
        super(HausdorffDTLoss, self).__init__()
        self.alpha = alpha
        
    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        batch_size = logits.shape[0]
        
        # Convert targets to one-hot encoding
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
        
        # Apply softmax to logits
        probs = F.softmax(logits, dim=1)
        
        # Initialize total loss
        total_loss = 0
        
        # Process each class
        for cls in range(num_classes):
            # Skip background class if needed
            if cls == 0 and num_classes > 2:  # For multi-class, often class 0 is background
                continue
                
            pred_cls = probs[:, cls]
            target_cls = targets_one_hot[:, cls]
            
            # Compute distance transforms for target
            # This is a simplified version - in a real implementation you'd use scipy or custom CUDA code
            # for actual distance transform. Here we'll use a simple approximation.
            
            # For targets: high values where target=0, low values where target=1
            neg_target = 1 - target_cls
            pos_target = target_cls
            
            # For predictions: high values where pred is low, low values where pred is high
            neg_pred = 1 - pred_cls
            pos_pred = pred_cls
            
            # Compute approximate Hausdorff losses
            loss_gt = (neg_target * pos_pred).sum(dim=(1, 2)) / (pos_pred.sum(dim=(1, 2)) + 1e-6)
            loss_pred = (neg_pred * pos_target).sum(dim=(1, 2)) / (pos_target.sum(dim=(1, 2)) + 1e-6)
            
            # Apply alpha exponent to emphasize boundaries
            class_loss = (loss_gt ** self.alpha + loss_pred ** self.alpha).mean()
            total_loss += class_loss
            
        # Average loss across applicable classes
        num_actual_classes = num_classes if num_classes <= 2 else num_classes - 1
        total_loss /= num_actual_classes
        
        return total_loss

class CombinedLoss(nn.Module):
    def __init__(self, num_classes, dice_weight=1.0, ce_weight=1.0, hausdorff_weight=1.0):
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.hausdorff_loss = HausdorffDTLoss()
        
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.hausdorff_weight = hausdorff_weight
        
    def forward(self, logits, targets):
        # Calculate individual losses
        dice = self.dice_loss(logits, targets)
        ce = self.ce_loss(logits, targets)
        hausdorff = self.hausdorff_loss(logits, targets)
        
        # Combine losses
        total_loss = (
            self.dice_weight * dice + 
            self.ce_weight * ce + 
            self.hausdorff_weight * hausdorff
        )
        
        return total_loss, {'dice': dice.item(), 'ce': ce.item(), 'hausdorff': hausdorff.item()}

# Training functions
def pretraining_step(model, optimizer, images, device):
    images = images.to(device)
    batch_size = images.shape[0]
    
    # Generate random mask
    patch_size = Config.patch_size
    num_patches = (images.shape[-1] // patch_size) ** 2
    mask = model.generate_mask(batch_size, num_patches, Config.mask_ratio)
    
    # Forward pass
    loss = model(images, mask)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

def train_simmim(model, data_loader, device, epochs=50, lr=1e-4):
    """Train SimMIM model on unlabeled data"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    model.train()
    model.to(device)
    
    for epoch in range(epochs):
        epoch_loss = 0
        with tqdm(data_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False) as pbar:
            for images in pbar:
                # Handle batch of crops
                b, num_crops, c, h, w = images.shape
                images = images.view(-1, c, h, w)  # [b*num_crops, c, h, w]
                
                loss = pretraining_step(model, optimizer, images, device)
                epoch_loss += loss
                pbar.set_postfix({"loss": loss})
        
        scheduler.step()
        avg_loss = epoch_loss / len(data_loader)
        print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}")
    
    return model

def segmentation_step(model, criterion, optimizer, images, masks, device):
    images = images.to(device)
    masks = masks.to(device)
    
    # Forward pass
    logits = model(images)
    loss, loss_components = criterion(logits, masks)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Calculate metrics
    preds = torch.argmax(logits, dim=1)
    accuracy = (preds == masks).float().mean().item()
    
    return loss.item(), loss_components, accuracy

def train_segmentation(model, data_loader, device, epochs=20, lr=1e-4):
    """Train segmentation model on labeled data"""
    criterion = CombinedLoss(
        Config.num_classes, 
        dice_weight=Config.dice_weight,
        ce_weight=Config.ce_weight,
        hausdorff_weight=Config.hausdorff_weight
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    model.train()
    model.to(device)
    
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_acc = 0
        dice_losses = 0
        ce_losses = 0
        hausdorff_losses = 0
        
        with tqdm(data_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False) as pbar:
            for images, masks in pbar:
                # Handle batch of crops
                b, num_crops, c, h, w = images.shape
                images = images.view(-1, c, h, w)  # [b*num_crops, c, h, w]
                masks = masks.view(-1, h, w)  # [b*num_crops, h, w]
                
                loss, loss_components, acc = segmentation_step(model, criterion, optimizer, images, masks, device)
                
                epoch_loss += loss
                epoch_acc += acc
                dice_losses += loss_components['dice']
                ce_losses += loss_components['ce']
                hausdorff_losses += loss_components['hausdorff']
                
                pbar.set_postfix({
                    "loss": loss, 
                    "acc": acc,
                    "dice": loss_components['dice'],
                    "ce": loss_components['ce'],
                    "hausdorff": loss_components['hausdorff']
                })
        
        scheduler.step()
        avg_loss = epoch_loss / len(data_loader)
        avg_acc = epoch_acc / len(data_loader)
        avg_dice = dice_losses / len(data_loader)
        avg_ce = ce_losses / len(data_loader)
        avg_hausdorff = hausdorff_losses / len(data_loader)
        
        print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}")
        print(f"Component Losses - Dice: {avg_dice:.4f}, CE: {avg_ce:.4f}, Hausdorff: {avg_hausdorff:.4f}")
    
    return model

def validate_segmentation(model, data_loader, device):
    """Validate segmentation model on labeled data"""
    criterion = CombinedLoss(
        Config.num_classes, 
        dice_weight=Config.dice_weight,
        ce_weight=Config.ce_weight,
        hausdorff_weight=Config.hausdorff_weight
    )
    
    model.eval()
    model.to(device)
    
    val_loss = 0
    val_acc = 0
    val_dice = 0
    val_ce = 0
    val_hausdorff = 0
    
    with torch.no_grad():
        with tqdm(data_loader, desc="Validation", leave=False) as pbar:
            for images, masks in pbar:
                # Handle batch of crops
                b, num_crops, c, h, w = images.shape
                images = images.view(-1, c, h, w)  # [b*num_crops, c, h, w]
                masks = masks.view(-1, h, w)  # [b*num_crops, h, w]
                
                images = images.to(device)
                masks = masks.to(device)
                
                # Forward pass
                logits = model(images)
                loss, loss_components = criterion(logits, masks)
                
                # Calculate metrics
                preds = torch.argmax(logits, dim=1)
                accuracy = (preds == masks).float().mean().item()
                
                val_loss += loss.item()
                val_acc += accuracy
                val_dice += loss_components['dice']
                val_ce += loss_components['ce']
                val_hausdorff += loss_components['hausdorff']
                
                pbar.set_postfix({"val_loss": loss.item(), "val_acc": accuracy})
    
    # Calculate averages
    val_loss /= len(data_loader)
    val_acc /= len(data_loader)
    val_dice /= len(data_loader)
    val_ce /= len(data_loader)
    val_hausdorff /= len(data_loader)
    
    print(f"Validation - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
    print(f"Component Losses - Dice: {val_dice:.4f}, CE: {val_ce:.4f}, Hausdorff: {val_hausdorff:.4f}")
    
    return val_loss, val_acc

def inference_with_crops(model, image, crop_size=224, overlap=0.5, batch_size=4):
    """
    Run inference on a large image by taking overlapping crops and 
    stitching predictions together
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Convert image to tensor if it's not already
    if not isinstance(image, torch.Tensor):
        transform = transforms.ToTensor()
        image = transform(image)
    
    if len(image.shape) == 3:  # Single image
        image = image.unsqueeze(0)  # Add batch dimension
    
    # Get image dimensions
    _, c, h, w = image.shape
    
    # Calculate stride (with overlap)
    stride = int(crop_size * (1 - overlap))
    
    # Create empty prediction mask
    pred_mask = torch.zeros((image.shape[0], Config.num_classes, h, w), device='cpu')
    count_mask = torch.zeros((image.shape[0], 1, h, w), device='cpu')
    
    # Generate crops
    crops = []
    crop_positions = []
    
    for i in range(0, h - crop_size + 1, stride):
        for j in range(0, w - crop_size + 1, stride):
            
            crop = image[:, :, i:i+crop_size, j:j+crop_size]
            crops.append(crop)
            crop_positions.append((i, j))
            
            # Process in batches when we have enough crops
            if len(crops) == batch_size or (i == h - crop_size and j == w - crop_size):
                # Stack crops
                batch_crops = torch.cat(crops, dim=0)
                
                # Forward pass
                with torch.no_grad():
                    batch_crops = batch_crops.to(device)
                    logits = model(batch_crops)
                    probs = F.softmax(logits, dim=1)
                
                # Add predictions to output mask
                for k, (crop_i, crop_j) in enumerate(crop_positions):
                    idx = k % image.shape[0]  # Get original image index
                    pred_mask[idx, :, crop_i:crop_i+crop_size, crop_j:crop_j+crop_size] += probs[k].cpu()
                    count_mask[idx, :, crop_i:crop_i+crop_size, crop_j:crop_j+crop_size] += 1
                
                # Reset crop lists
                crops = []
                crop_positions = []
    
    # Handle any remaining crops
    if len(crops) > 0:
        # Stack crops
        batch_crops = torch.cat(crops, dim=0)
        
        # Forward pass
        with torch.no_grad():
            batch_crops = batch_crops.to(device)
            logits = model(batch_crops)
            probs = F.softmax(logits, dim=1)
        
        # Add predictions to output mask
        for k, (crop_i, crop_j) in enumerate(crop_positions):
            idx = k % image.shape[0]  # Get original image index
            pred_mask[idx, :, crop_i:crop_i+crop_size, crop_j:crop_j+crop_size] += probs[k].cpu()
            count_mask[idx, :, crop_i:crop_i+crop_size, crop_j:crop_j+crop_size] += 1
    
    # Average predictions (handle areas with multiple overlapping crops)
    count_mask[count_mask == 0] = 1  # Avoid division by zero
    pred_mask = pred_mask / count_mask
    
    # Get class predictions
    final_preds = torch.argmax(pred_mask, dim=1)
    
    return final_preds

# Main execution
def main():
    print(f"Using device: {Config.device}")
    
    # Setup data transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    unlabeled_dataset = UnlabeledDataset(
        Config.unlabeled_data_dir, 
        crop_size=Config.crop_size, 
        num_crops=Config.num_crops,
        transform=transform
    )
    
    labeled_dataset = LabeledDataset(
        Config.labeled_data_dir, 
        crop_size=Config.crop_size, 
        num_crops=Config.num_crops,
        transform=transform
    )
    
    # Create data loaders
    unlabeled_loader = DataLoader(
        unlabeled_dataset, 
        batch_size=Config.pretrain_batch_size,
        shuffle=True, 
        num_workers=Config.num_workers
    )
    
    # Split labeled data into train and validation
    train_size = int(0.8 * len(labeled_dataset))
    val_size = len(labeled_dataset) - train_size
    labeled_train_dataset, labeled_val_dataset = torch.utils.data.random_split(
        labeled_dataset, [train_size, val_size]
    )
    
    labeled_train_loader = DataLoader(
        labeled_train_dataset, 
        batch_size=Config.seg_batch_size,
        shuffle=True, 
        num_workers=Config.num_workers
    )
    
    labeled_val_loader = DataLoader(
        labeled_val_dataset, 
        batch_size=Config.seg_batch_size,
        shuffle=False, 
        num_workers=Config.num_workers
    )
    
    # Step 1: Load pretrained ViT model
    print("Loading pretrained ViT model...")
    vit_model = timm.create_model('vit_base_patch16_224', pretrained=True)
    
    # Step 2: Create SimMIM model and pretraining on unlabeled data
    print("Creating SimMIM model for pretraining...")
    simmim_model = SimMIM(vit_model, patch_size=Config.patch_size)
    
    print("Pretraining with SimMIM on unlabeled data...")
    simmim_model = train_simmim(
        simmim_model, 
        unlabeled_loader, 
        Config.device, 
        epochs=Config.pretrain_epochs, 
        lr=Config.pretrain_lr
    )
    
    # Save pretrained model
    torch.save(simmim_model.encoder.state_dict(), 'pretrained_vit_simmim.pth')
    print("Pretrained model saved to 'pretrained_vit_simmim.pth'")
    
    # Step 3: Create ViT with segmentation head
    print("Creating segmentation model...")
    seg_model = ViTSegmentation(vit_model, Config.num_classes, patch_size=Config.patch_size)
    
    # Step 4: Train segmentation head on labeled data
    print("Training segmentation model on labeled data...")
    seg_model = train_segmentation(
        seg_model, 
        labeled_train_loader, 
        Config.device, 
        epochs=Config.seg_epochs, 
        lr=Config.seg_lr
    )
    
    # Step 5: Validate segmentation model
    print("Validating segmentation model...")
    val_loss, val_acc = validate_segmentation(seg_model, labeled_val_loader, Config.device)
    
    # Save final segmentation model
    torch.save(seg_model.state_dict(), 'vit_segmentation_model.pth')
    print("Segmentation model saved to 'vit_segmentation_model.pth'")
    
    print(f"Final validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
    
    return seg_model

# Sample inference function
def predict_on_single_image(model, image_path, device=None):
    """Run prediction on a single image file"""
    if device is None:
        device = next(model.parameters()).device
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0)
    
    # Run inference with multiple crops
    prediction = inference_with_crops(
        model, 
        image_tensor, 
        crop_size=Config.crop_size, 
        overlap=0.5, 
        batch_size=4
    )
    
    return prediction.squeeze(0).numpy()

# Visualization functions
def visualize_prediction(image_path, prediction, num_classes, save_path=None):
    """Visualize segmentation prediction"""
    # Load original image
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)
    
    # Create color map for segmentation classes
    colors = plt.cm.rainbow(np.linspace(0, 1, num_classes))
    
    # Create segmentation overlay
    seg_image = np.zeros((prediction.shape[0], prediction.shape[1], 4))
    for i in range(num_classes):
        seg_image[prediction == i] = colors[i]
    
    # Plot
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image_np)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(image_np)
    plt.imshow(seg_image, alpha=0.5)
    plt.title('Segmentation')
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    seg_model = main()
    