import numpy as np
import pandas as pd
import csv
import json
from pandas import DataFrame
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from mlxtend.preprocessing import standardize
import matplotlib.pyplot as plt
import seaborn as sns



with open('tokenized_data.csv', 'r') as read_obj: 
  
    # Return a reader object which will 
    # iterate over lines in the given csvfile 
    csv_reader = csv.reader(read_obj) 
  
    # convert string to list 
    list_of_csv = list(csv_reader) 

column_names = list_of_csv.pop(0)

actual_lists = [json.loads(s) for l_of_l in list_of_csv for s in l_of_l]

list1 = []

for list_of_list in list_of_csv:
    list2 = []
    for l in list_of_list:
        list2.append(json.loads(l))
    list1.append(list2[1:8])

dimension0 = [list[0] for list in list1]
dimension1 = [list[1] for list in list1]
dimension2 = [list[2] for list in list1]
dimension3 = [list[3] for list in list1]
dimension4 = [list[4] for list in list1]
dimension5 = [list[5] for list in list1]
dimension6 = [list[6] for list in list1]

# Function to pad sublists with a placeholder value
def pad_sublist(sublist, length, placeholder):
    return sublist + [0] * (length - len(sublist))

def kmeans_clustering(dimension):

    # Determine the maximum length of sublists
    max_len = max(len(sublist) for sublist in dimension)

    # Pad sublists to make them the same length
    padded_data = [pad_sublist(sublist, max_len, 0) for sublist in dimension]

    # Create a K-Means model
    kmeans = KMeans(n_clusters=30, random_state=0)

    kmeans.fit(padded_data)

    cluster_centers = kmeans.cluster_centers_
    cluster_labels = kmeans.labels_

    return cluster_centers, cluster_labels

cluster0 = kmeans_clustering(dimension0)
cluster1 = kmeans_clustering(dimension1)
cluster2 = kmeans_clustering(dimension2)
cluster3 = kmeans_clustering(dimension3)
cluster4 = kmeans_clustering(dimension4)
cluster5 = kmeans_clustering(dimension5)
cluster6 = kmeans_clustering(dimension6)

# Example cluster assignments from 7 dimensions
matrix = []

for i in range(156):
    line = [cluster0[1][i].tolist(), cluster1[1][i].tolist(), cluster2[1][i].tolist(), cluster3[1][i].tolist(), cluster4[1][i].tolist(), cluster5[1][i].tolist(), cluster6[1][i].tolist()]
    matrix.append(line)

# cluster_assignments = [cluster0[1].tolist(), cluster1[1].tolist(), cluster2[1].tolist(), cluster3[1].tolist(), cluster4[1].tolist(), cluster5[1].tolist(), cluster6[1].tolist()]

df = pd.DataFrame(matrix, columns=[0, 1, 2, 3, 4, 5, 6])

X = standardize(df, columns=[0, 1, 2, 3, 4, 5, 6])

pca = PCA(n_components=2)

reduced_data = pca.fit_transform(X)

x = []
y = []

for i in range(156):

    x.append(reduced_data[i][0])
    y.append(reduced_data[i][1])    

# plt.scatter(x, y)
# plt.xlabel("PC1")
# plt.ylabel("PC2")
# plt.show()

kmeans_pca = KMeans(n_clusters=30)
cluster_labels = kmeans_pca.fit_predict(reduced_data)
print(cluster_labels)