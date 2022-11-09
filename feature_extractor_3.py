import sys
import os
import json
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from numba import cuda
from tqdm import tqdm
import gc
import psutil

p = psutil.Process()

device = torch.device('cpu')

sys.path.insert(1, '/shared/home/v_rahul_pratap_singh/local_scratch/UnsupervisedVAD')

try:
  import google.colab
  IN_COLAB = True
except:
  IN_COLAB = False

if IN_COLAB:
    from UnsupervisedVAD.video_dataset import VideoFrameDataset, ImglistToTensor
    from google.colab.patches import cv2_imshow as cv2_imshow
    path = './foo/'
else:
    from video_dataset import VideoFrameDataset, ImglistToTensor
    from cv2 import imshow as cv2_imshow
    path = './'

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# device = cuda.get_current_device()
# device.reset()

transfrom = transforms.Compose([
            ImglistToTensor(),
            transforms.CenterCrop((256,256)),
            transforms.Resize((224, 224)),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

skipped = [1499]#[931, 1259]

# Number of segments to be processed at one time
chunk = 1875

@profile
def func():
    for vid_num in skipped:

        vid_address = '/shared/home/v_rahul_pratap_singh/local_scratch/UnsupervisedVAD/Dance/all_text/train_'+str(vid_num)+'.txt'

        if IN_COLAB:
            file = open('./UnsupervisedVAD/train.txt', 'r')
        else:
            file = open(vid_address, 'r')

        dataset = VideoFrameDataset('/shared/home/v_rahul_pratap_singh/local_scratch/UnsupervisedVAD/Dataset/Frames/', vid_address, num_segments=1, frames_per_segment=16, 
                                    imagefile_template='frame_{:05d}.jpg', transform=None, test_mode=False)
        
        # [[[][][]n frames][][]...1900]
        # print(dataset[0])
        # print(len(dataset), len(dataset[0][0]), len(dataset[0]))
        # frames = len(dataset[0][0]

        # len dataset[0,0] =8000
        # num_iterations = 8000/16 
        # start = 0
        # segments has 1 len

        num_iterations = len(dataset[0][0])//16
        start = 0
        while(start<=num_iterations):
            segments = [[]] * 1    
            # print(sample)



            # sample = dataset[0]
            
            
            
            if ( start+chunk < num_iterations):
                sample = dataset[0][0][16*start:16*(start+chunk)]

            else:
                sample = dataset[0][0][16*start:]
            start += chunk

            frames = sample
            
            
            
            del dataset
            del sample
            gc.collect()



            # print(len(frames))
            # print(frames)
            segment = []

            for cluster in range(len(frames)//16):
                # 16 segment, 0-255 frames, 0 15 ok 16 31....240 255....0:16,  .....[240:]
                if (cluster == (len(frames)//16)-1):
                    segment_i = frames[16 * cluster:]
                
                else:
                    segment_i = frames[16 * cluster: 16 * (cluster + 1)]
                segment.append(segment_i)

            # print(segment)
            segments[0] = segment



            del frames
            del segment
            gc.collect()



            segments = np.array(segments, dtype = object)
            # print(super_segments)

            frameSize = (224, 224)
            vid_tensors=[[]] * 1

            for segment in range(len(segments)): #get_video
                lt=[]
                #print('video no.',i)
                vid=segments[segment]
                frameset=np.array(vid, dtype = object) #array of 10 segments
                #print('no. of segments in video ',i,': ',len(frameset))

                for seg_index in range(0,len(frameset)):
                    #print('seg_index',seg_index)
                    seg=frameset[seg_index]
                    l=[]
                    for k in range(0,len(seg)): #16 frames per segment
                        #print('frame no.',k)
                        pil_img=seg[k]
                        cv_img=np.array(pil_img)
                        cv_img=cv2.resize(cv_img,(112,112))
                        #cv2_imshow(cv_img)
                        #print(cv_img.shape)
                        l.append(cv_img)
                
                    t=tuple(l)
                    
                    x = np.stack(t, axis = -1)
                    y= np.transpose(x, (3,2,1,0)) #tensor for one segment
                    lt.append(y)
                    #print('current shape',np.array(vid_tensors[0]).shape,np.array(vid_tensors[1]).shape)
                vid_tensors[segment]=lt
                #print(i,len(vid_tensors[i]))


            del frameset
            gc.collect()



            # print("tensor", vid_num)
            #cv2.destroyAllWindows()
            vid_tensors=np.array(vid_tensors, dtype = object)

            data_segments=[]

            for j in range(0,len(vid_tensors)):
                for k in range(0,len(vid_tensors[j])):
                    data_segments.append(vid_tensors[j][k])
            # print(len(data_segments))


            
            del vid_tensors
            del segments
            gc.collect()



            # Create json file for feature extractor
            count=0
            ls=[]
            for s in range(0, len(data_segments)):
                count+=1
                tens=data_segments[s]
                op_d={'video': tens}
                op_d_new=op_d
                op_d_new['video']=np.array(op_d['video']).tolist()
                ls.append(op_d_new)



                del tens
                del op_d
                del op_d_new
                gc.collect()
                
                
                
                # print('writing line ',count)
            # print("json", vid_num)
            json_object=json.dumps(ls)


            del data_segments
            del ls
            gc.collect()



            json_filename = 'sample_'+str(vid_num)+'.json'

            os.path.join('/shared/home/v_rahul_pratap_singh/local_scratch/UnsupervisedVAD/Dance/sample_jsons', json_filename)
            with open('/shared/home/v_rahul_pratap_singh/local_scratch/UnsupervisedVAD/Dance/sample_jsons/'+json_filename, "w") as outfile:
                outfile.write(json_object)
            
            
            outfile.close()
            del json_object
            gc.collect()



            path = '/shared/home/v_rahul_pratap_singh/local_scratch/UnsupervisedVAD/'
            json_path = '/shared/home/v_rahul_pratap_singh/local_scratch/UnsupervisedVAD/Dance/sample_jsons/'+json_filename
            
            command =  'python '+path+'video_feature_extractor/extract.py --jsn='+json_path+' --type=3d --batch_size=1 --resnext101_model_path='+path+'resnext101.pth --vid_num='+str(vid_num)
            os.system(command)
        
        print(vid_num)

func()

print(p.memory_info())