import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
import torchvision.transforms as T


# Based on https://github.com/Tramac/awesome-semantic-segmentation-pytorch
class CitySegmentation(Dataset):
    """Cityscapes Semantic Segmentation Dataset.
    Parameters
    ----------
    root : string
        Path to Cityscapes folder. Default is './datasets/citys'
    split: string
        'train', 'val' or 'test'
    transform : callable, optional
        A function that transforms the image
    Examples
    --------
    """
    def __init__(self, root='./data/cityscapes', split='train', img_transform=None, mask_transform=None):

        self.root = root
        self.split = split
        assert os.path.exists(self.root)
        self.images, self.mask_paths = _get_city_pairs(self.root, self.split)
        assert (len(self.images) == len(self.mask_paths))
        if len(self.images) == 0:
            raise RuntimeError("Found 0 images in subfolders of:" + root + "\n")
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22,
                              23, 24, 25, 26, 27, 28, 31, 32, 33]
        self._key = np.array([-1, -1, -1, -1, -1, -1,
                              -1, -1, 0, 1, -1, -1,
                              2, 3, 4, -1, -1, -1,
                              5, -1, 6, 7, 8, 9,
                              10, 11, 12, 13, 14, 15,
                              -1, -1, 16, 17, 18])
        self._mapping = np.array(range(-1, len(self._key) - 1)).astype('int32')

        self._img_transform = img_transform
        self._mask_transform = mask_transform


    def _class_to_index(self, mask):
        # assert the value
        values = np.unique(mask)
        for value in values:
            assert (value in self._mapping)
        index = np.digitize(mask.ravel(), self._mapping, right=True)
        return self._key[index].reshape(mask.shape)

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        state = torch.get_rng_state()
        if self._img_transform:
            img = self._img_transform(img)
        mask = Image.open(self.mask_paths[index])
        mask = self._class_to_index(np.array(mask).astype('int32'))
        mask = Image.fromarray(mask.astype('int32'))
        torch.set_rng_state(state)
        if self._mask_transform:
            mask = self._mask_transform(mask).squeeze()
        return img, mask, os.path.basename(self.images[index])

    def __len__(self):
        return len(self.images)

    @property
    def pred_offset(self):
        return 0


def _get_city_pairs(folder, split='train'):
    def get_path_pairs(img_folder, mask_folder):
        img_paths = []
        mask_paths = []
        for root, _, files in os.walk(img_folder):
            for filename in files:
                if filename.endswith('.png'):
                    imgpath = os.path.join(root, filename)
                    foldername = os.path.basename(os.path.dirname(imgpath))
                    maskname = filename.replace('leftImg8bit', 'gtFine_labelIds')
                    maskpath = os.path.join(mask_folder, foldername, maskname)
                    if os.path.isfile(imgpath) and os.path.isfile(maskpath):
                        img_paths.append(imgpath)
                        mask_paths.append(maskpath)
                    else:
                        print('cannot find the mask or image:', imgpath, maskpath)
        print('Found {} images in the folder {}'.format(len(img_paths), img_folder))
        return img_paths, mask_paths

    if split in ('train', 'val'):
        img_folder = os.path.join(folder, 'leftImg8bit/' + split)
        mask_folder = os.path.join(folder, 'gtFine/' + split)
        img_paths, mask_paths = get_path_pairs(img_folder, mask_folder)
        return img_paths, mask_paths
    else:
        assert split == 'trainval'
        print('trainval set')
        train_img_folder = os.path.join(folder, 'leftImg8bit/train')
        train_mask_folder = os.path.join(folder, 'gtFine/train')
        val_img_folder = os.path.join(folder, 'leftImg8bit/val')
        val_mask_folder = os.path.join(folder, 'gtFine/val')
        train_img_paths, train_mask_paths = get_path_pairs(train_img_folder, train_mask_folder)
        val_img_paths, val_mask_paths = get_path_pairs(val_img_folder, val_mask_folder)
        img_paths = train_img_paths + val_img_paths
        mask_paths = train_mask_paths + val_mask_paths
    return img_paths, mask_paths


def get_dataloaders(cfg):
    if cfg.TRAIN_CROP:
        train_img_transform = T.Compose([
            T.Resize(cfg.SHAPE),
            T.RandomCrop(cfg.CROP_SIZE),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        train_label_transform = T.Compose([
            T.Resize(cfg.SHAPE),
            T.RandomCrop(cfg.CROP_SIZE),
            T.ToTensor(),
        ])
    else:
        train_img_transform = T.Compose([
            T.Resize(cfg.SHAPE),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        train_label_transform = T.Compose([
            T.Resize(cfg.SHAPE),
            T.ToTensor(),
        ])

    if cfg.VAL_CROP:
        val_img_transform = T.Compose([
            T.Resize(cfg.SHAPE),
            T.RandomCrop(cfg.CROP_SIZE),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        val_label_transform = T.Compose([
            T.Resize(cfg.SHAPE),
            T.RandomCrop(cfg.CROP_SIZE),
            T.ToTensor(),
        ])
    else:
        val_img_transform = T.Compose([
            T.Resize(cfg.SHAPE),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        val_label_transform = T.Compose([
            T.Resize(cfg.SHAPE),
            T.ToTensor(),
        ])

    train_set = CitySegmentation(root=cfg.DATASET_ROOT, split='train', img_transform=train_img_transform, mask_transform=train_label_transform)
    val_set = CitySegmentation(root=cfg.DATASET_ROOT, split='val', img_transform=val_img_transform, mask_transform=val_label_transform)
    train_set = ConcatDataset([train_set]*cfg.TRAIN_REPEAT)

    if cfg.SUBSET:
        train_set = Subset(train_set, np.arange(cfg.SUBSET_SIZE))
        val_set = Subset(val_set, np.arange(cfg.SUBSET_SIZE))

    train_loader = DataLoader(train_set, cfg.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, 1, shuffle=False)
    return train_loader, val_loader


if __name__ == '__main__':
    train_loader, _ = get_dataloaders()
    img, target, _ = train_loader.dataset[0]
    print(img.shape, target.shape)
    from matplotlib import pyplot as plt
    plt.imshow(img[0,:,:])
    plt.show()
    plt.imshow(target)
    plt.show()