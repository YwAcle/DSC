from __future__ import absolute_import
import os
import os.path as osp
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import copy
import math
from PIL import Image


class Preprocessor(Dataset):
    def __init__(self, dataset, root=None, transform=None):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        img = Image.open(fpath).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, fname, pid, camid, index

class Preprocessor_color(Dataset):
    def __init__(self, dataset, root=None, transform=None,transform1=None):
        super(Preprocessor_color, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform
        self.transform1 = transform1
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        img_ori = Image.open(fpath).convert('RGB')

        if self.transform is not None:
            img = self.transform(img_ori)
            img1 = self.transform1(img_ori)
        return img, img1,fname, pid, camid, index


class VideoPreprocessor(Dataset):
    def __init__(self, dataset, root=None, transform=None, mutual=False):
        super(VideoPreprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform
        self.mutual = mutual
        self.sample_method = 'random'
        self.seq_len = 6

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        return self._get_seq_item(indices)

    def _get_seq_item(self, index):
        img_paths, pid, camid = self.dataset[index]
        num_imgs = len(img_paths)

        if self.sample_method == 'random':
            # Randomly samples seq_len images from a tracklet of length num_imgs,
            # if num_imgs is smaller than seq_len, then replicates images
            indices = np.arange(num_imgs)
            replace = False if num_imgs >= self.seq_len else True
            indices = np.random.choice(
                indices, size=self.seq_len, replace=replace
            )
            # sort indices to keep temporal order (comment it to be order-agnostic)
            indices = np.sort(indices)

        elif self.sample_method == 'evenly':
            # Evenly samples seq_len images from a tracklet
            if num_imgs >= self.seq_len:
                num_imgs -= num_imgs % self.seq_len
                indices = np.arange(0, num_imgs, num_imgs / self.seq_len)
            else:
                # if num_imgs is smaller than seq_len, simply replicate the last image
                # until the seq_len requirement is satisfied
                indices = np.arange(0, num_imgs)
                num_pads = self.seq_len - num_imgs
                indices = np.concatenate(
                    [
                        indices,
                        np.ones(num_pads).astype(np.int32) * (num_imgs - 1)
                    ]
                )
            assert len(indices) == self.seq_len

        elif self.sample_method == 'all':
            # Samples all images in a tracklet. batch_size must be set to 1
            indices = np.arange(num_imgs)
        else:
            raise ValueError(
                'Unknown sample method: {}'.format(self.sample_method)
            )

        imgs = []
        for index in indices:
            img_path = img_paths[int(index)]
            img = Image.open(img_path).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            img = img.unsqueeze(0)  # img must be torch.Tensor
            imgs.append(img)
        imgs = torch.cat(imgs, dim=0)
        imgs = imgs.cpu().numpy()

        return imgs, img_paths[0], pid, camid, index


class VideoPreprocessor_color(Dataset):
    def __init__(self, dataset, root=None, transform=None, transform1=None):
        super(VideoPreprocessor_color, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform
        self.transform1 = transform1
        self.sample_method = 'random'
        self.seq_len = 6

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        return self._get_seq_item(indices)

    def _get_seq_item(self, index):
        img_paths, pid, camid = self.dataset[index]
        num_imgs = len(img_paths)

        if self.sample_method == 'random':
            # Randomly samples seq_len images from a tracklet of length num_imgs,
            # if num_imgs is smaller than seq_len, then replicates images
            indices = np.arange(num_imgs)
            replace = False if num_imgs >= self.seq_len else True
            indices = np.random.choice(
                indices, size=self.seq_len, replace=replace
            )
            # sort indices to keep temporal order (comment it to be order-agnostic)
            indices = np.sort(indices)

        elif self.sample_method == 'evenly':
            # Evenly samples seq_len images from a tracklet
            if num_imgs >= self.seq_len:
                num_imgs -= num_imgs % self.seq_len
                indices = np.arange(0, num_imgs, num_imgs / self.seq_len)
            else:
                # if num_imgs is smaller than seq_len, simply replicate the last image
                # until the seq_len requirement is satisfied
                indices = np.arange(0, num_imgs)
                num_pads = self.seq_len - num_imgs
                indices = np.concatenate(
                    [
                        indices,
                        np.ones(num_pads).astype(np.int32) * (num_imgs - 1)
                    ]
                )
            assert len(indices) == self.seq_len

        elif self.sample_method == 'all':
            # Samples all images in a tracklet. batch_size must be set to 1
            indices = np.arange(num_imgs)

        else:
            raise ValueError(
                'Unknown sample method: {}'.format(self.sample_method)
            )

        imgs = []
        imgs_ca = []
        for index in indices:
            img_path = img_paths[int(index)]
            img = Image.open(img_path).convert('RGB')
            img_copy = copy.deepcopy(img)

            if self.transform is not None:
                img = self.transform(img)
            img = img.unsqueeze(0)  # img must be torch.Tensor
            imgs.append(img)

            if self.transform1 is not None:
                img_ca = self.transform1(img_copy)
            img_ca = img_ca.unsqueeze(0)
            imgs_ca.append(img_ca)

        imgs = torch.cat(imgs, dim=0)
        imgs_ca = torch.cat(imgs_ca, dim=0)

        return imgs, imgs_ca, img_paths[0], pid, camid, index
