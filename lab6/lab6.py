import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cluster, datasets, metrics
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/'
                      'machine-learning-databases/wine/wine.data',
                      header=None)

# if the Wine dataset is temporarily unavailable from the
# UCI machine learning repository, un-comment the following line
# of code to load the dataset from a local path:

# df_wine = pd.read_csv('wine.data', header=None)

df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines', 'Proline']

df_wine.head()
#df_wine.to_csv('WindeData.csv')

# Task 1:
# Check the counts of each wine class
print(df_wine["Class label"].value_counts())

# Fetch data into x and y variables
x=df_wine.ix[:,1:14] # Covariates or input
y=df_wine.ix[:,:1] # Labels or classes or output
print(x.columns)
print(y.columns)

# Task 2
# Clustering using K Means Algorithm
km_wine=cluster.KMeans(n_clusters=3)
km_wine.fit(x)

#print("Completeness: %0.3f" % metrics.completeness_score(y.values().tolist(), km.labels_))
#print("Silhouette Coefficient: %0.3f"
      #% metrics.silhouette_score(y., km.labels_

# Task 3
# Scatter plot data into 3 classes based on True Labels and plot with legends. Hint: use any two variables
plt.figure(figsize=(8,5))
plt.title("Wine data", fontsize=18)
plt.grid(True)
plt.scatter(x["Alcohol"],x["Proline"],c=y["Class label"])
plt.savefig('Wine_plot.png', dpi=300)
plt.legend([1,2,3])
plt.show()