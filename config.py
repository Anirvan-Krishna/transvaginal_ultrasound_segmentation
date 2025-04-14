import torch
import torchvision
import os
import torchvision.transforms as T

class Dataset:

    def __init__(self,
                  image_path, 
                  label_path=None,
                  transform=None):
        
        self.image_files = sorted(
            [f for f in os.listdir(image_path) if f.endswith('.png')])

        self.label_files = sorted(
            [f for f in os.listdir(label_path) if f.endswith('.png')]
        ) if label_path else None

        self.image_path = image_path
        self.label_path = label_path
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_path, self.image_files[idx])

        try:
            image = torchvision.io.read_image(
                image_path).float() / 255.0  # Convert to float [0,1]
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Instead of returning (None, None), return None to skip this sample
            return None

        label = None
        if self.label_path:
            label_path = os.path.join(self.label_path, self.label_files[idx])
            try:
                label = torchvision.io.read_image(label_path)
            except Exception as e:
                print(f"Error loading label {label_path}: {e}")

        if self.transform:
            image = self.transform(image)
            
            if label is not None:
                label = self.transform(label)

        return image if label is None else (image, label)
