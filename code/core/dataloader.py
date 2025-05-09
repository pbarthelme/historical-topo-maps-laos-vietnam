import lightning as L
import numpy as np
import torch
import torchvision.transforms.v2 as transforms

from torch.utils.data import DataLoader, Dataset
from torchvision.tv_tensors import Mask


class TopoMapsDataset(Dataset):
    """A PyTorch Dataset for handling the labelled topographic maps data."""
    def __init__(self, images, masks, transform=None):
        self.images = images
        self.masks = torch.LongTensor(masks)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]

        if self.transform:
            # set to type Mask to ensure the transform functions know
            # to treat it as a label
            image, mask = self.transform(image, Mask(mask))

        # return transformed image and mask
        return image, mask
        
class TopoMapsPredDataset(Dataset):
    """A PyTorch Dataset for handling the topographic maps data during prediction."""
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image

class TopoMapsDataModule(L.LightningDataModule):
    """
    A PyTorch Lightning DataModule for handling the labelled topographic maps data.
    
    This module handles data preparation, augmentation, and loading during training, validation, and prediction.

    Parameters:
        data_path (str, optional): Path to the `.npz` file containing training and validation data.
        pred_data (numpy.ndarray, optional): Array of images for prediction.
        batch_size (int, optional): Batch size for the DataLoader. Defaults to 8.
        brightness (float, optional): Maximum brightness adjustment for data augmentation. Defaults to 0.
        contrast (float, optional): Maximum contrast adjustment for data augmentation. Defaults to 0.
        saturation (float, optional): Maximum saturation adjustment for data augmentation. Defaults to 0.
        hue (float, optional): Maximum hue adjustment for data augmentation. Defaults to 0.
        rot_deg (float, optional): Maximum rotation angle (in degrees) for data augmentation. Defaults to 0.
        rot_fill (int, optional): Fill value for areas outside the image after rotation. Defaults to 0.
    """
    def __init__(self, data_path=None, pred_data=None, batch_size=8, brightness=0, contrast=0, saturation=0, hue=0, rot_deg=0, rot_fill=0):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.pred_data = pred_data

        # Define augmentations for training and validation
        self.train_transform = transforms.Compose([
            transforms.ToImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue),
            transforms.RandomRotation(degrees=rot_deg, fill=rot_fill),
            transforms.ToDtype(torch.float32, scale=True),
        ])
        self.val_transform = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
        ])
        self.pred_transform = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True)
        ])

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if stage == "predict":
            self.pred_dataset = TopoMapsPredDataset(self.pred_data, transform=self.pred_transform)
                
        if stage != "predict":
            data = np.load(self.data_path)
            x_train, y_train = data["x_train"], data["y_train"]
            x_val, y_val = data["x_val"], data["y_val"]

            self.train_dataset = TopoMapsDataset(x_train, y_train, transform=self.train_transform)
            self.val_dataset = TopoMapsDataset(x_val, y_val, transform=self.val_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
    
    def predict_dataloader(self):
        return DataLoader(self.pred_dataset, batch_size=self.batch_size, shuffle=False)