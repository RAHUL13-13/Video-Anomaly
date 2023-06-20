import os
import numpy as np

vid_names = [...]

for vid_name in vid_names:
    for segment in os.listdir('/shared/home/v_rahul_pratap_singh/local_scratch/UnsupervisedVAD/Dance/temp_ResNext/'+vid_name):
        new = np.load('/shared/home/v_rahul_pratap_singh/local_scratch/UnsupervisedVAD/Dance/temp_ResNext/'+vid_name+'/'+segment)
        old = np.load('/shared/home/v_rahul_pratap_singh/local_scratch/UnsupervisedVAD/Dance/ResNext_Extractions/'+vid_name+'/'+segment)
        print(new, old)
        if np.array_equal(np.array(new), np.array(old)):
            continue

        print("No Use", segment)
        exit()
print("all set")

# # old
# dataset all
# segments based on fill size
# fill segments all

# tensor..

# # new
# loop i (0, 1900)
#     dataset i
#     segments based on fill size
#     fill segments i

#     tensor..

#     full dataset------1 video -----some segment at a time
#     calculate frames
#     caluculate segments
#     if bottleneck 200 segment
#     loop with increment of 200*16


