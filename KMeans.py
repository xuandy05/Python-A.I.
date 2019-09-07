import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/00292/Wholesale%20customers%20data.csv")
#seperate the Channel and Region column from the data frame
#Channel should be its own series, as well as Region
channel = df['Channel']
region = df['Region']

df = df.drop('Channel', 1)
df = df.drop('Region', 1)

df.describe()

df.hist()

describe = df.describe()
features = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']
for each in features:
  IQR = describe[each]['75%'] - describe[each]['25%']
  channel = channel[df[each] < describe[each]['75%'] + 1.5 * IQR]
  region = region[df[each] < describe[each]['75%'] + 1.5 * IQR]
  df = df[df[each] < describe[each]['75%'] + 1.5 * IQR]
  channel = channel[df[each] > describe[each]['25%'] - 1.5 * IQR]
  region = region[df[each] > describe[each]['25%'] - 1.5 * IQR]
  df = df[df[each] > describe[each]['25%'] - 1.5 * IQR]

df = df.reset_index(drop = True)
channel = channel.reset_index(drop = True)
region = region.reset_index(drop = True)

df = np.log(df)
df.hist()

from sklearn.decomposition import PCA
#Create a PCA with 2 components
#train the PCA model
#use pca to summarize the data
#convert output back into a dataframe, the output should have only 2 features, named them Dimension 1 and Dimension 2
pca = PCA(n_components = 2)
pca.fit(df)
p = pca.transform(df)
df2 = pd.DataFrame(p, columns = ['Dimension 1', 'Dimension 2'])

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

#Create the KMeans model with any number of clusters. The only input is number of cluster centers
#train the model with .fit
#create predictions with .predict
kmeans = KMeans(n_clusters = 2)
kmeans.fit(df2)
pred = kmeans.predict(df2)
#Use silhouette_score to evaluate the performance of the model.
#silhouette_score is ranges from -1 to 1
#1 is the best, 0 indicates overlapping clusters, -1 indicates that points have been assigned to the wrong cluster
score = silhouette_score(df2, pred, metric='euclidean')

# Find the optimal number of clusters. Use silhouette_score to evaluate the performance of the model
# Use a loop to train the KMeans model with different number of clusters
best_score = -1
optimal = 0
for i in range(2, 10):
  kmeans = KMeans(n_clusters = i)
  kmeans.fit(df2)
  pred = kmeans.predict(df2)
  score = silhouette_score(df2, pred, metric='euclidean')
  if score > best_score:
    best_score = score
    optimal = i
print("The optimal number of clusters is", optimal, "with a score of", best_score)
kmeans = KMeans(n_clusters = optimal)
kmeans.fit(df2)
pred = kmeans.predict(df2)

# plot the data points with the predicted labels
p1 = df2[pred == 0]
p2 = df2[pred == 1]
plt.scatter(p1['Dimension 1'], p1['Dimension 2'], color = 'b', marker = 'o')
plt.scatter(p2['Dimension 1'], p2['Dimension 2'], color = 'r', marker = '*')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')

# plot the data points with the actual labels for Channel
plt.scatter(df2[channel == 1]['Dimension 1'], df2[channel == 1]['Dimension 2'])
plt.scatter(df2[channel == 2]['Dimension 1'], df2[channel == 2]['Dimension 2'])
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')

# plot the data points with the actual labels for Region
plt.scatter(df2[region == 1]['Dimension 1'], df2[region == 1]['Dimension 2'])
plt.scatter(df2[region == 2]['Dimension 1'], df2[region == 2]['Dimension 2'])
plt.scatter(df2[region == 3]['Dimension 1'], df2[region == 3]['Dimension 2'])
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
