import numpy as np
import os
import pickle

from torch.utils.data import Dataset
from PIL import Image



class ImageNet32(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, transform=None, target_transform=None):
        data = []
        targets = []
        for idx in range(1, 11):
            with open(os.path.join(root_dir, 'Imagenet32_train/train_data_batch_' + str(idx)), "rb") as fo:
                batch = pickle.load(fo)

            data.extend(batch['data'].reshape(-1, 3, 32, 32))
            targets.extend([y - 1 for y in batch['labels']])

        self.data = np.asarray(data).transpose((0, 2, 3, 1))

        self.targets = np.asarray(targets)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return 1281167

    def __getitem__(self, idx):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[idx], self.targets[idx]
        print(img.shape)

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
