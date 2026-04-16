import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import os
from datasets import load_dataset
import logging

logger = logging.getLogger(__name__)

class DummyImageNet(torch.utils.data.Dataset):
    def __init__(self, size=256, num_samples=1000, num_classes=1000):
        self.size = size
        self.num_samples = num_samples
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Random image in [-1, 1]
        img = torch.randn(3, self.size, self.size)
        img = torch.clamp(img, -1, 1)

        # Random label
        label = torch.randint(0, self.num_classes, (1,)).item()
        return img, label

class HuggingFaceImageNet(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        try:
            item = self.dataset[idx]
            image = item['image']
            label = item['label']

            if image.mode != 'RGB':
                image = image.convert('RGB')

            if self.transform:
                image = self.transform(image)

            return image, label
        except Exception as e:
            logger.error(f"Error loading image at index {idx}: {e}")
            # Robustness: try next image or return zero
            # Returning zero might affect training stability if frequent.
            # Ideally, we retry another index.
            # Simple retry logic:
            if idx + 1 < len(self):
                return self.__getitem__(idx + 1)
            return torch.zeros(3, 256, 256), 0

def get_loader(config, train=True):
    if train:
        # Standard ImageNet training transforms
        transform = transforms.Compose([
            transforms.RandomResizedCrop(config.image_size),
            transforms.RandomHorizontalFlip(),
            # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4), # Uncomment if paper uses it
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    else:
        # Validation transforms
        transform = transforms.Compose([
            transforms.Resize(config.image_size), # Usually resize to 256 then crop 224 for standard ResNet, but for DiT it might be just Resize or Resize(256)+CenterCrop(256)
            transforms.CenterCrop(config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    # Check if path contains parquet files or looks like a HF dataset directory
    is_hf_dataset = False
    if os.path.exists(config.data_path):
        # Heuristic: check for parquet files or dataset_dict.json
        if any(f.endswith('.parquet') for f in os.listdir(config.data_path)) or \
            os.path.exists(os.path.join(config.data_path, "dataset_dict.json")) or \
            os.path.exists(os.path.join(config.data_path, "data")):
            is_hf_dataset = True

    if is_hf_dataset:
        try:
            split = 'train' if train else 'validation'
            # Check if validation split exists, otherwise use test or whatever is available
            # Or assume standard names

            # Load dataset from disk
            # If the path is a folder with parquet files, load_dataset might need 'data_dir' or just path
            # The structure in LS shows 'data' folder inside.

            # If loading from local directory with parquet files
            # datasets.load_dataset(path, split=split)
            print(f"Loading HF dataset from {config.data_path}...")
            hf_ds = load_dataset(config.data_path, split=split) # streaming=False for map-style

            dataset = HuggingFaceImageNet(hf_ds, transform=transform)
            print(f"Loaded {split} split with {len(dataset)} samples.")

        except Exception as e:
            print(f"Failed to load HF dataset: {e}. Falling back to ImageFolder/Dummy.")
            is_hf_dataset = False

    if not is_hf_dataset:
        if os.path.exists(config.data_path):
            # Assuming ImageNet structure: train/val folders with class subfolders
            split = 'train' if train else 'val'
            path = os.path.join(config.data_path, split)
            if os.path.exists(path):
                # Robustness check included in ImageFolder by default for loading, but we can wrap it if needed.
                # Standard ImageFolder is usually robust enough for standard files.
                dataset = datasets.ImageFolder(path, transform=transform)
            else:
                logger.warning(f"Path {path} does not exist (ImageFolder). Using Dummy Dataset.")
                dataset = DummyImageNet(size=config.image_size, num_classes=config.num_classes-1)
        else:
            logger.warning(f"Data path {config.data_path} does not exist. Using Dummy Dataset.")
            dataset = DummyImageNet(size=config.image_size, num_classes=config.num_classes-1)

    loader = DataLoader(
        dataset,
        batch_size=config.micro_batch_size,
        shuffle=train,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    return loader
