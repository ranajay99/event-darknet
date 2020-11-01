##### done editing

import glob
import random
import os
import numpy as np

import torch

from torch.utils.data import Dataset
import torchvision.transforms as transforms

import sys

# event-image is NOT cropped & resized here
# w, h = 640, 480
# make it 640 640 with padding

class ImageFolder(Dataset):
    def __init__(self, folder_path, img_res=(640,480)):
        self.files = sorted(glob.glob('%s/*.*' % folder_path))
        self.res = img_res

    def __getitem__(self, index):
        event_path = self.files[index % len(self.files)]
        events = np.load(event_path).astype(np.float32)

        w, h = res
        dim_diff = np.abs(h - w)
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        events[:,2]+=pad1

        return event_path, events

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(self, list_path, img_res=(640,480), img_size=412):
        with open(list_path, 'r') as file:
            self.event_files = file.readlines()
        self.label_files = [path.replace('events', 'labels').replace('.npz', '.txt') for path in self.event_files]
        self.max_objects = 1 ###############
        self.img_shape = (img_size, img_size)
        self.res = img_res

    def __getitem__(self, index):
        event_path = self.event_files[index % len(self.event_files)].rstrip()
        events = np.load(event_path).astype(np.float32)

        w, h = res
        dim_diff = np.abs(h - w)
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        events[:,2]+=pad1

        label_path = self.label_files[index % len(self.event_files)].rstrip()

        labels = None
        if os.path.exists(label_path):
            labels = np.loadtxt(label_path).reshape(-1, 5)
            # Extract coordinates for unpadded + unscaled image
            y1 = h * (labels[:, 2] - labels[:, 4]/2)
            y2 = h * (labels[:, 2] + labels[:, 4]/2)
            # Adjust for added padding
            y1 += pad1
            y2 += pad1
            # Calculate ratios from coordinates
            labels[:, 2] = ((y1 + y2) / 2) / w
            labels[:, 4] *= h / w

        filled_labels = np.zeros((self.max_objects, 5))
        if labels is not None:
            filled_labels[range(len(labels))[:self.max_objects]] = labels[:self.max_objects]
        filled_labels = torch.from_numpy(filled_labels)

        return event_path, events, filled_labels

    def __len__(self):
        return len(self.event_files)
