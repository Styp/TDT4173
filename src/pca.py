import time
import os, os.path
import pandas as pd
import string
import scipy

import numpy
import PIL

import scipy.misc as scipymisc

import plotly.graph_objs as go

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from data_loader import DataLoader

import plotly.offline as py

path = "../chars74k-lite_split"
dto = DataLoader.loadData(path)

X_std = StandardScaler().fit_transform(dto.X_train)

pca = PCA(n_components=5)
pca.fit(X_std)
X_pca = pca.transform(X_std)

groups = [ ord(x) - ord('a') + 1 for x in dto.Y_train ]

print(X_pca)

trace0 = go.Scatter(
    x = X_pca[:,0],
    y = X_pca[:,1],
    #z = X_pca[:,2],
    name = dto.Y_train,
    #hoveron = dto.Y_train,
    mode = 'markers',
    text = dto.Y_train,
    showlegend = False,
    marker = dict(
        size = 8,
        color = groups,
        colorscale ='Jet',
        showscale = False,
        line = dict(
            width = 2,
            color = 'rgb(255, 255, 255)'
        ),
        opacity = 0.8
    )
)
data = [trace0]

layout = go.Layout(
    title= 'Principal Component Analysis (PCA)',
    hovermode= 'closest',
    xaxis= dict(
         title= 'First Principal Component',
        ticklen= 5,
        zeroline= False,
        gridwidth= 2,
    ),
    yaxis=dict(
        title= 'Second Principal Component',
        ticklen= 5,
        gridwidth= 2,
    ),
    showlegend= True
)


fig = dict(data=data, layout=layout)
py.plot(fig, filename='pca')



from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans # KMeans clustering

kmeans = KMeans(n_clusters=26)
# Compute cluster centers and predict cluster indices

Y_pred = kmeans.fit_predict(X_pca, y=groups)

print(kmeans.score(X_pca))

trace_Kmeans = go.Scatter(x=X_pca[:, 0], y= X_pca[:, 1], mode="markers",
                    showlegend=False,
                    marker=dict(
                            size=8,
                            color = Y_pred,
                            colorscale = 'Portland',
                            showscale=False,
                            line = dict(
            width = 2,
            color = 'rgb(255, 255, 255)'
        )
                   ))



layout = go.Layout(
    title= 'KMeans Clustering',
    hovermode= 'closest',
    xaxis= dict(
         title= 'First Principal Component',
        ticklen= 5,
        zeroline= False,
        gridwidth= 2,
    ),
    yaxis=dict(
        title= 'Second Principal Component',
        ticklen= 5,
        gridwidth= 2,
    ),
    showlegend= True
)

data = [trace_Kmeans]
fig1 = dict(data=data, layout= layout)
# fig1.append_trace(contour_list)
py.plot(fig1, filename="cluster")
