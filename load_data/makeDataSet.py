import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import glob


def read_split(txt_path, stackSize, label_in_frout=False, appointed_labels=list(range(15))):
    Dataset = []
    Labels = []
    NumFrames = []
    with open(txt_path, 'r') as fr:
        for line in fr:
            if label_in_frout:
                path, label, num_frame = line.strip().split(' ')
            else:
                path, num_frame, label = line.strip().split(' ')
            if int(label) in appointed_labels:
                num_frame = int(num_frame)
                if num_frame >= stackSize:
                    Dataset.append(path)
                    Labels.append(int(label))
                    NumFrames.append(num_frame)
    return Dataset, Labels, NumFrames

class makeDataset(Dataset):
    def __init__(self, txt_path, spatial_transform=None,seqLen=20,
                 train=True, mulSeg=False, numSeg=1, fmt='.jpg', get_VidName=False):

        self.images, self.labels, self.numFrames = read_split(txt_path, 5)
        self.spatial_transform = spatial_transform
        self.train = train
        self.mulSeg = mulSeg
        self.numSeg = numSeg
        self.seqLen = seqLen
        self.fmt = fmt
        self.get_vidName = get_VidName

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        vid_name = self.images[idx]
        label = self.labels[idx]
        numFrame = self.numFrames[idx]
        inpSeq = []
        for trans in self.spatial_transform.values():
            trans.randomize_parameters()
        for i in np.linspace(1, numFrame, self.seqLen, endpoint=False):
            fl_name = vid_name + '/' + 'img_' + str(int(np.floor(i))).zfill(5) + self.fmt
            img = Image.open(fl_name)
            inpSeq.append(img.convert('RGB'))
        inpSeq = self.spatial_transform['vid'](inpSeq)
        inpSeq = [self.spatial_transform['pic'](img) for img in inpSeq]
        inpSeq = torch.stack(inpSeq, 0).permute(1, 0, 2, 3)
        if self.get_vidName:
            return inpSeq, label, vid_name
        return inpSeq, label