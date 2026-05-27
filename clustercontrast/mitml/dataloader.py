from __future__ import print_function, absolute_import

import random
import math
import numpy as np
from PIL import Image
from sklearn.preprocessing import normalize

import torch
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.data import Dataset


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class VideoDataset_test(Dataset):
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """
    sample_methods = ['evenly', 'random', 'all']

    def __init__(self, dataset, seq_len=12, sample='evenly', transform=None):
        self.dataset = dataset
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_paths, pid, camid = self.dataset[index]
        num = len(img_paths)

        S = self.seq_len
        sample_clip_ir = []
        frame_indices_ir = list(range(num))
        if num < S:  
            strip_ir = list(range(num)) + [frame_indices_ir[-1]] * (S - num)
            for s in range(S):
                pool_ir = strip_ir[s * 1:(s + 1) * 1]
                sample_clip_ir.append(list(pool_ir))
        else:
            inter_val_ir = math.ceil(num / S)
            strip_ir = list(range(num)) + [frame_indices_ir[-1]] * (inter_val_ir * S - num)
            for s in range(S):
                pool_ir = strip_ir[inter_val_ir * s:inter_val_ir * (s + 1)]
                sample_clip_ir.append(list(pool_ir))

        sample_clip_ir = np.array(sample_clip_ir)

        if self.sample == 'dense':
            """
            Sample all frames in a video into a list of clips, each clip contains seq_len frames, batch_size needs to be set to 1.
            This sampling strategy is used in test phase.
            """
            cur_index=0
            frame_indices = range(num)
            indices_list=[]
            while num-cur_index > self.seq_len:
                indices_list.append(frame_indices[cur_index:cur_index+self.seq_len])
                cur_index+=self.seq_len
            last_seq=frame_indices[cur_index:]
            last_seq = list(last_seq)
            for index in last_seq:
                if len(last_seq) >= self.seq_len:
                    break
                last_seq.append(index)
            indices_list.append(last_seq)
            imgs_list=[]
            for indices in indices_list:
                imgs = []
                for index in indices:
                    index=int(index)
                    img_path = img_paths[index]
                    img = read_image(img_path)
 
                    img = np.array(img)
                    if self.transform is not None:
                        img = self.transform(img)
                    img = img.unsqueeze(0)
                    imgs.append(img)
                imgs = torch.cat(imgs, dim=0)
              
                imgs_list.append(imgs)
            imgs_array = torch.stack(imgs_list) 
            return imgs_array, pid, camid

        if self.sample == 'random':
            """
            Randomly sample seq_len consecutive frames from num frames,
            if num is smaller than seq_len, then replicate items.
            This sampling strategy is used in training phase.
            """
            num_ir = len(img_paths)
            frame_indices = range(num_ir)
            rand_end = max(0, len(frame_indices) - self.seq_len - 1)
            begin_index = random.randint(0, rand_end)
            end_index = min(begin_index + self.seq_len, len(frame_indices))

            indices = frame_indices[begin_index:end_index]
            indices = list(indices)
            for index in indices:
                if len(indices) >= self.seq_len:
                    break
                indices.append(index)
            indices = np.array(indices)
            imgs_ir = []
            for index in indices:
                index = int(index)
                img_path = img_paths[index]
                img = read_image(img_path)

                img = np.array(img)
                if self.transform is not None:
                    img = self.transform(img)

                imgs_ir.append(img)
            imgs_ir = torch.cat(imgs_ir, dim=0)
            return imgs_ir, pid, camid

        if self.sample == 'video_test':
            number = sample_clip_ir[:, 0]
            imgs_ir = []
            for index in number:
                index = int(index)
                img_path = img_paths[index]
                img = read_image(img_path)

                img = np.array(img)
                if self.transform is not None: 
                    img = self.transform(img)

                imgs_ir.append(img)
            imgs_ir = torch.cat(imgs_ir, dim=0)
            return imgs_ir, pid, camid
        else:
            raise KeyError("Unknown sample method: {}. Expected one of {}".format(self.sample, self.sample_methods))


class VCMTestLoader():
    def __init__(self, dataset_vcm, height=256, width=128, test_batch=32, seq_length=6) -> None:

        # self.n_class = dataset_vcm.num_train_pids
        self.n_testset_rgb = dataset_vcm.testset_rgb_tracklets_num
        self.n_testset_ir = dataset_vcm.testset_ir_tracklets_num

        self.seq_length = seq_length

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform_test = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((height, width)),
            transforms.ToTensor(),
            normalize,
        ])

        self.testloader_rgb = DataLoader(
            VideoDataset_test(dataset_vcm.testset_rgb, seq_len=seq_length, sample='video_test', transform=transform_test),
            batch_size=test_batch, shuffle=False, num_workers=4)
        self.testloader_ir = DataLoader(
            VideoDataset_test(dataset_vcm.testset_ir, seq_len=seq_length, sample='video_test', transform=transform_test),
            batch_size=test_batch, shuffle=False, num_workers=4)
        

        
class BUPTTestLoader():
    def __init__(self, dataset_bupt, height=256, width=128, test_batch=32, seq_length=6) -> None:
        self.seq_length = seq_length

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform_test = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((height, width)),
            transforms.ToTensor(),
            normalize,
        ])

        self.testloader_rgb = DataLoader(
            VideoDataset_test(dataset_bupt.testset_rgb, seq_len=seq_length, sample='video_test', transform=transform_test),
            batch_size=test_batch, shuffle=False, num_workers=4)
        
        self.testloader_ir = DataLoader(
            VideoDataset_test(dataset_bupt.testset_ir, seq_len=seq_length, sample='video_test', transform=transform_test),
            batch_size=test_batch, shuffle=False, num_workers=4)
            
            
    
