import os, os.path
import string
import scipy

import PIL

import plotly.offline as py

import plotly.graph_objs as go

import matplotlib.pyplot as plt

import itertools
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix

from data_loader import DataCarrierObject
from data_loader import DataLoader

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

path = "../chars74k-lite_split"

dto = DataLoader.loadData(path)

X_std = StandardScaler().fit_transform(dto.X_train)
pca = PCA(n_components=20)
pca.fit(X_std)
X_5d = pca.transform(X_std)

Y_train_transform = [ ord(x) - ord('a') + 1 for x in dto.Y_train ]


kmeansClassifier = KMeans(n_clusters=26)
# Compute cluster centers and predict cluster indices
X_clustered = kmeansClassifier.fit(X_5d,dto.Y_train)


X_std = StandardScaler().fit_transform(dto.X_test)
pca = PCA(n_components=5)
pca.fit(X_std)
X_5d = pca.transform(X_std)

Y_test_transform = [ ord(x) - ord('a') + 1 for x in dto.Y_test ]

#cnf_matrix = confusion_matrix(Y_test_transform, kmeansClassifier.predict(X_5d))

#df_cm = pd.DataFrame(cnf_matrix, index = range(27),
#                  columns = range(27))
#plt.figure()
#sn.heatmap(df_cm, annot=True)
#plt.savefig("confusion.png")