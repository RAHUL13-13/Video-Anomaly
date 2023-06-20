import os

combined = open('/shared/home/v_rahul_pratap_singh/local_scratch/UnsupervisedVAD/train.txt', 'r')
lines = combined.readlines()
combined.close()

for i in range(len(lines)):
    os.path.join('/shared/home/v_rahul_pratap_singh/local_scratch/UnsupervisedVAD/Dance/all_train/', 'train_'+str(i)+'.txt')
    new = open('train_'+str(i)+'.txt', 'w')
    new.writeline(lines[i])
    new.close()
