import numpy as np
from sklearn.decomposition import PCA

X_train = np.load('/shared/home/v_rahul_pratap_singh/local_scratch/UnsupervisedVAD/Dance/different_dataset_splits/all_train_set_video_features.npy')
# X_test = np.load('/content/gdrive/MyDrive/ResNext Features/different_dataset_splits/all_test_set_video_features.npy')
print('shape of train set:', X_train.shape)

pca = PCA(n_components=0)

pca.fit(X_train)

X_test = np.load('/shared/home/v_rahul_pratap_singh/local_scratch/UnsupervisedVAD/Dance/different_dataset_splits/all_test_set_video_features.npy')
pca_test = pca.transform(X_test)
print('shape test:', pca_test.shape)

temp = np.load('/shared/home/v_rahul_pratap_singh/local_scratch/UnsupervisedVAD/Dance/different_dataset_splits/all_test_set_labels_segment_level.npy', allow_pickle = True)
a = []
for i in range(290):
  for j in range(len(temp[i])):
    a.append(temp[i][j])

# print(len(a))
labels = np.array(a).reshape(-1)
print(labels.shape)

import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

pca_data = np.vstack((pca_test.T, labels)).T
pca_df = pd.DataFrame(data=pca_data, columns=("1st_principal", "2nd_principal", "label"))
sn.FacetGrid(pca_df, hue="label", height =6).map(plt.scatter, '1st_principal', '2nd_principal').add_legend()
plt.show()
plt.savefig("pca.png")
