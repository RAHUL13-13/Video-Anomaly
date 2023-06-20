import torch as th
import math
import numpy as np
from video_loader import VideoLoader
from torch.utils.data import DataLoader
import argparse
from model import get_model
from preprocessing import Preprocessing
from random_sequence_shuffler import RandomSequenceSampler
import torch.nn.functional as F
import os
import gc

device = th.device('cuda:0')

parser = argparse.ArgumentParser(description='Easy video feature extractor')

parser.add_argument(
    '--jsn',
    type=str,
    help='input json with video input path')
parser.add_argument('--batch_size', type=int, default=64,
                            help='batch size')
parser.add_argument('--type', type=str, default='2d',
                            help='CNN type')
parser.add_argument('--half_precision', type=int, default=1,
                            help='output half precision float')
parser.add_argument('--num_decoding_thread', type=int, default=4,
                            help='Num parallel thread for video decoding')
parser.add_argument('--l2_normalize', type=int, default=1,
                            help='l2 normalize feature')
parser.add_argument('--resnext101_model_path', type=str, default='model/resnext101.pth',
                            help='Resnext model path')
parser.add_argument('--vid_num', type=str, help='video number')
args = parser.parse_args()

dataset = VideoLoader(
    args.jsn,
    framerate=1 if args.type == '2d' else 24,
    size=224 if args.type == '2d' else 112,
    centercrop=(args.type == '3d'),
)
n_dataset = len(dataset)
# sampler = RandomSequenceSampler(n_dataset, 10)
loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=args.num_decoding_thread,
    #sampler=sampler if n_dataset > 10 else None,
    sampler=None
)
preprocess = Preprocessing(args.type)
model = get_model(args)
vid_num = args.vid_num

address = '/shared/home/v_rahul_pratap_singh/local_scratch/UnsupervisedVAD'
if not os.path.exists(address+'/Dance/tempoResNext/vid_'+str(vid_num)):
    os.mkdir(address+'/Dance/tempoResNext/vid_'+str(vid_num))

path=address+'/Dance/tempoResNext/vid_'+str(vid_num)

segment_num = len(os.listdir(path))

with th.no_grad():
    for k, data in enumerate(loader):

        input_file = ''
        output_file = ''
        if len(data['video'].shape) > 3:
            # print('Computing features of video {}/{}: {}'.format(
            #     k + 1, n_dataset, input_file))
            video = data['video'].squeeze()
            if len(video.shape) == 4:
                
                if (os.path.isdir(path+'/output'+str(segment_num)+'.npy')):
                    segment_num+=1
                    continue
                
                video = preprocess(video)
                n_chunk = len(video)
                features = th.FloatTensor(n_chunk, 2048).fill_(0)
                features = features.to(device)
                n_iter = int(math.ceil(n_chunk / float(args.batch_size)))
                for i in range(n_iter):
                    min_ind = i * args.batch_size
                    max_ind = (i + 1) * args.batch_size
                    video_batch = video[min_ind:max_ind].to(device)
                  
                    gc.collect()


                    batch_features = model(video_batch)
                    if args.l2_normalize:
                        batch_features = F.normalize(batch_features, dim=1)
                    features[min_ind:max_ind] = batch_features
                  
                    gc.collect()

                features = features.cpu().numpy()
                if args.half_precision:
                    features = features.astype('float16')
                np.save(path+'/output'+str(segment_num)+'.npy', features)

                del features
                gc.collect()

                segment_num+=1
        else:
            print('Video {} already processed.'.format(input_file))

# print(vid_num, end=" ")
