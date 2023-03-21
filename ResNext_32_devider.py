import os
import numpy as np
from natsort import natsorted

path='/shared/home/v_rahul_pratap_singh/local_scratch/UnsupervisedVAD/Dance/ResNext_Extractions'

#go to all videos in the path
for video in natsorted(os.listdir(path)):
    vid_path=os.path.join(path,video)
    ex_seg=len(os.listdir(vid_path)) #no.of existing segments for the video
    Feature_vect= np.zeros((ex_seg,2048)) #2048 dim features for resnext
    i=0
    
    for segment in os.listdir(vid_path):        
        s=np.load(os.path.join(vid_path,segment))
        Feature_vect[i]=s
        i+=1
    #Feature vect will be of the form [[--s1--],[--s2---],]

    # print("shape of Feature_vect: ", Feature_vect.shape)

    #32 Segments
    
    Segments_Features=np.zeros((32,2048))
    temp_vect=np.array([])

    #generates list of 32 numbers with appropriate gaps so as to cover all segments
    thirty2_shots= np.linspace(1,ex_seg,33, dtype=int)
    count=0
    for ishots in range(0,len(thirty2_shots)-1):
        ss=int(thirty2_shots[ishots])
        ee=int(thirty2_shots[ishots+1]-1)

        if ishots==len(thirty2_shots):
            ee=thirty2_shots[ishots+1]

        if ss==ee:
            temp_vect=Feature_vect[ss:ee+1,:]
            
        elif ee<ss:
                temp_vect=Feature_vect[ss,:]
            
        else:
            temp_vect=np.mean(Feature_vect[ss:ee+1,:], axis=0)
       
        temp_vect=temp_vect/np.linalg.norm(temp_vect)
        
       # if norm(temp_vect)==0
           #error('??')
        Segments_Features[count,:]=temp_vect
        count+=1
    #save segments for each video
    # print("shape of Segments_Features: ", Segments_Features.shape)

    np.save('/shared/home/v_rahul_pratap_singh/local_scratch/UnsupervisedVAD/Dance/ResNext_32_interpole/'+str(video)+'.npy', Segments_Features)
