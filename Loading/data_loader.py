import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from .data_reader import read_image, read_mask


def get_train_transform():
    return A.Compose(
        [
            A.Resize(256, 256),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            A.HorizontalFlip(p=0.25),
            A.VerticalFlip(p=0.25),
            ToTensorV2()
        ])


# Dataset Loader
class LoadDataSet(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transforms = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        mask_folder = self.df.iloc[idx]["mask_dir"]
        image_path = self.df.iloc[idx]["image_path"]

        img = read_image(image_path)
        mask = read_mask(mask_folder)

        augmented = self.transforms(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask']
        mask = mask.permute(2, 0, 1)
        mask = torch.div(mask, 255)
        return img, mask
