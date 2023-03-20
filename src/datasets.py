import glob
import torch
import random

import torch.nn.functional as F
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset, DataLoader

import random

class PairedImgDataset(Dataset):
    def __init__(self, data_source, mode, crop=256, random_resize=None, fixed_resize=None):
        if not mode in ['train', 'val', 'test']:
            raise Exception('The mode should be "train", "val" or "test".')
        
        self.random_resize = random_resize
        self.fixed_resize = fixed_resize
        self.crop = crop
        self.mode = mode
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
            
        self.img_paths, self.gt_paths = self.get_path(data_source, mode, val_len=100)

    def __getitem__(self, index):
        img = Image.open(self.img_paths[index % len(self.img_paths)]).convert('RGB')
        gt = Image.open(self.gt_paths[index % len(self.gt_paths)]).convert('RGB')
        
        img = self.transform(img)
        gt = self.transform(gt)
        
        # semantic alignment for input
        h, w = img.size(1), img.size(2)
        shift_up = 4
        shift_left = 5
        img[:, 0:img.size(1)-shift_up, 0:img.size(2)-shift_left] = img[:, shift_up:, shift_left:]
        
        if self.mode == 'train':
            if self.random_resize is not None:
                # random resize
                scale_factor = random.uniform(self.crop/self.random_resize, 1.)
                img = F.interpolate(img.unsqueeze(0), scale_factor=scale_factor, align_corners=False, mode='bilinear', recompute_scale_factor=False).squeeze(0)
                gt = F.interpolate(gt.unsqueeze(0), scale_factor=scale_factor, align_corners=False, mode='bilinear', recompute_scale_factor=False).squeeze(0)
                
            elif self.fixed_resize is not None:
                # fixed resize
                scale_size = (self.crop + self.fixed_resize, int((self.crop + self.fixed_resize)*1.33))
                img = F.interpolate(img.unsqueeze(0), size=scale_size, align_corners=False, mode='bilinear', recompute_scale_factor=False).squeeze(0)
                gt = F.interpolate(gt.unsqueeze(0), size=scale_size, align_corners=False, mode='bilinear', recompute_scale_factor=False).squeeze(0)
                
            # crop
            h, w = img.size(1), img.size(2)
            offset_h = random.randint(0, max(0, h - self.crop - 1))
            offset_w = random.randint(0, max(0, w - self.crop - 1))

            img = img[:, offset_h:offset_h + self.crop, offset_w:offset_w + self.crop]
            gt = gt[:, offset_h:offset_h + self.crop, offset_w:offset_w + self.crop]
        
            # flip
            # vertical flip
#             if random.random() < 0.5:
#                 idx = [i for i in range(img.size(1) - 1, -1, -1)]
#                 idx = torch.LongTensor(idx)
#                 img = img.index_select(1, idx)
#                 gt = gt.index_select(1, idx)
            # horizontal flip
            if random.random() < 0.5:
                idx = [i for i in range(img.size(2) - 1, -1, -1)]
                idx = torch.LongTensor(idx)
                img = img.index_select(2, idx)
                gt = gt.index_select(2, idx)
        
        return img, gt

    def __len__(self):
        return max(len(self.img_paths), len(self.gt_paths))
    
    def get_path(self, data_source, mode, val_len):
        img_paths_train = sorted(glob.glob(data_source + '/train' + '/input' + '/*.*'))
        gt_paths_train = sorted(glob.glob(data_source + '/train' + '/gt' + '/*.*'))
        all_len = len(img_paths_train)
        
        # get random list
        index = []
        while len(index) < val_len:
            rand = random.randint(0,all_len-1)
            if rand not in index:
                index.append(rand)
        
        index = sorted(index)
        
        if mode == 'train':
            for i in range(val_len):
                del img_paths_train[index[val_len-(i+1)]]
                del gt_paths_train[index[val_len-(i+1)]]
                
            return img_paths_train, gt_paths_train
        
        else:
            img_paths_val = []
            gt_paths_val = []
            for i in range(val_len):
                img_paths_val.append(img_paths_train[index[i]])
                gt_paths_val.append(gt_paths_train[index[i]])
            
            return img_paths_val, gt_paths_val
        
class SingleImgDataset(Dataset):
    def __init__(self, data_source):
        
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        self.img_paths = sorted(glob.glob(data_source + '/test' + '/input' + '/*.*'))

    def __getitem__(self, index):
        
        path = self.img_paths[index % len(self.img_paths)]
        
        img = Image.open(path).convert('RGB')
        
        img = self.transform(img)
        
        # semantic alignment for input
        h, w = img.size(1), img.size(2)
        shift_up = 4
        shift_left = 5
        img[:, 0:img.size(1)-shift_up, 0:img.size(2)-shift_left] = img[:, shift_up:, shift_left:]
        
        return img, path

    def __len__(self):
        return len(self.img_paths)