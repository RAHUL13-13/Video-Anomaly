from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import  os

X = []
Y = []

path = '/shared/home/v_rahul_pratap_singh/local_scratch/UnsupervisedVAD/UCF-Crime/all_rgbs/'

all_category = os.listdir(path)

for i in all_category:
    all_vid_categ = os.listdir(path+i)
    for j in all_vid_categ:
        vid_feat = np.load(path+i+'/'+j, allow_pickle=True)
        for k in vid_feat:
            X.append(np.array(k))
            if i == 'Normal_Videos_event':
                Y.append(0)
            else:
                Y.append(1)

X = np.array(X)
y = np.array(Y)

def fun(n_comp, perplex, iter, lr):
    
    # first reduce dimensionality before feeding to t-sne
    pca = PCA(n_components=n_comp)
    X_pca = pca.fit_transform(X) 
    
    # randomly sample data to run quickly
    rows = np.arange(len(X))
    np.random.shuffle(rows)
    n_select = len(X)

    # reduce dimensionality with t-sne
    tsne = TSNE(n_components=2, verbose=1, perplexity=perplex, n_iter=iter, learning_rate=lr)
    tsne_results = tsne.fit_transform(X_pca[rows[:n_select],:])
    
    # visualize
    df_tsne = pd.DataFrame(tsne_results, columns=['comp1', 'comp2'])
    df_tsne['label'] = y[rows[:n_select]]
    plot_ = sns.lmplot(x='comp1', y='comp2', data=df_tsne, hue='label', fit_reg=False)

    name = "tSNE,I3D,"+str(n_comp)+','+str(perplex)+','+str(iter)+','+str(lr)
    
    plot_.savefig(name)

fun(5, 2, 250, 500)
