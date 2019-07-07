## Introductions

- Unsupervised ml is learning without labeled data
- k-means clustering is an unsup ml algo avail in sklearn.cluster.KMeans

### Getting started with KMeans

We can first look at the data using a scatterplot (a good dataset is the flower
dataset `sklearn.dataaset.load_iris()`)

```python
xs = points[:,0]
ys = points[:,1]
plt.scatter(xs,ys)
plt.show()
```

The following snippet demonstrates the use

```python
from sklearn.cluster import KMeans
model = KMeans(n_clusters=3)
model.fit(points)
labels = model.predict(new_points)
print(labels)
```

Now lets scatterplot again with colors and cluster centers

```python
centroids_x = model.cluster_centers_[:,0]
centroids_y = model.cluster_centers_[:,1]
plt.scatter(xs,ys,c=labels)
plt.scatter(centroids_x,centroids_y,marker='D',s=100)
```


### Measuring quality

So how do we know that our model is effective?
- If we have some labels we can check if they match their clusters
- We can measure intertia, i.e. how spread out the clusters are
- We can choose the amount of clusters using an elbowing method, plot the
  intertia and choose the amount of clusters when we see diminishing returns

```python
inertias = []
ks = range(1,20)
for k in ks:
    model = KMeans(n_clusters=k)
    model.fit(train)
    inertias.append(model.inertia_)
plt.plot(ks, inertias, '-o')
```

- Crosstabbing using labeled data.  It takes to pd columns, say targets and
  predictions and creates a matrix of targets in columns and predictions in
  rows

```python
import pandas as pd
df = pd.DataFrame({'labels': labels,  # Kmeans clusters
                   'targets': iris.target_names[iris.target]})
ct = pd.crosstab(df['labels'], df['targets'])
ct
```

### Preprocessing the data

If our data is very spread out (or this is really a often a good thing to do
anyway), we probably want to normalize the data.  Scikit has the standardscaler
ready for us to use in sklearn.preprocessing:

```python
scaler = StandardScaler()
scaler.fit(samples)
sankes_scaled = scaler.transform(samples)
```

Note that the standardscaler is here to change/normalize our data, thus it uses
the transform method. We can use the wine dataset to create a pipeline:

```python
scaler = StandardScaler()
kmeans = KMeans(n_clusters=3)
pipeline = make_pipeline(scaler,  kmeans)
pipeline.fit(samples)
labels = pipeline.predict(samples)
```

Then do the crosstab and it fly!

In addition to the standardscaler, sklearn has among avail the normalizer and
maxabsscaler aswell.
