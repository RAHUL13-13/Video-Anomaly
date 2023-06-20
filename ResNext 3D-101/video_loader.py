import torch as th
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
import ffmpeg
import json


class VideoLoader(Dataset):
    """Pytorch video loader."""

    def __init__(
            self,
            jsn,
            framerate=1,
            size=112,
            centercrop=False,
    ):
        """
        Args:
        """
        with open(jsn,"r") as f:
            self.vid_list=json.load(f) #list of 20 dictionaries
        self.centercrop = centercrop
        self.size = size
        self.framerate = framerate

    def __len__(self):
        return len(self.vid_list)

    def _get_video_dim(self, vid_dict):

        vid_t=vid_dict['video'].shape
        height=vid_t[2]
        width=vid_t[3]
        self.size=(height,width)
        return height, width

    def _get_output_dim(self, h, w):
        if isinstance(self.size, tuple) and len(self.size) == 2:
            return self.size
        elif h >= w:
            return int(h * self.size / w), self.size
        else:
            return self.size, int(w * self.size / h)

    def __getitem__(self, idx):
        vid_dict=self.vid_list[idx]
        vid_dict['video']=th.from_numpy(np.array(vid_dict['video']))
        h,w=self._get_video_dim(vid_dict)
        (height, width) = self._get_output_dim(h, w)
        return vid_dict
