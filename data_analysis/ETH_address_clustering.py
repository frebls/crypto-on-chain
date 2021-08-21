# %%
import os
from google.cloud.bigquery.client import Client
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import typing
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from collections import Counter

# %%
# set Google BigQuery client
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'C:\\Users\\Francesco\\fiery-rarity-322109-6ba6fa8a811c.json'
bq_client = Client()

# set seaborn plotting theme
sns.set_theme()

# set colour palette
COLORS_PALETTE = ['#00388F', '#FFB400', '#FF4B00', '#65B800', '#00B1EA']
sns.set_palette(sns.color_palette(COLORS_PALETTE))

# %%
# run query
address_clustering = bq_client.query('''
    select *
    from `fiery-rarity-322109.ethereum.address_classification`
    where rank <= 1000 and date_month >= '2017-01-01'
''').to_dataframe()

# %%
class SklearnWrapper:
    def __init__(self, transformation: typing.Callable):
        self.transformation = transformation
        self._group_transforms = []
        # Start with -1 and for each group up the pointer by one
        self._pointer = -1

    def _call_with_function(self, df: pd.DataFrame, function: str):
        # If pointer >= len we are making a new apply, reset _pointer
        if self._pointer >= len(self._group_transforms):
            self._pointer = -1
        self._pointer += 1
        return pd.DataFrame(
            getattr(self._group_transforms[self._pointer], function)(df.values),
            columns=df.columns,
            index=df.index,
        )

    def fit(self, df):
        self._group_transforms.append(self.transformation.fit(df.values))
        return self

    def transform(self, df):
        return self._call_with_function(df, "transform")

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)

    def inverse_transform(self, df):
        return self._call_with_function(df, "inverse_transform")

# %%
# Create scaler outside the class
scaler = SklearnWrapper(StandardScaler())

# Fit and transform data (holding state)
df_scale = address_clustering.loc[: , ~address_clustering.columns.isin(['address', 'rank'])].groupby("date_month").apply(scaler.fit_transform)
df_scale = df_scale.loc[: , ~df_scale.columns.isin(['date_month'])]

# %%
# Run t-SNE
tsne = TSNE(n_components=3, verbose=1, random_state=42) #, perplexity=80, n_iter=5000, learning_rate=200
tsne_scale_results = tsne.fit_transform(df_scale)
tsne_df_scale = pd.DataFrame(tsne_scale_results, columns=['t-SNE 1', 't-SNE 2', 't-SNE 3'])

# %%
# run K-means for a range of k values
sse = []
k_list = range(1, 15)

for k in k_list:
    km = KMeans(n_clusters=k)
    km.fit(tsne_df_scale)
    sse.append([k, km.inertia_])
    
tsne_results_scale = pd.DataFrame({'Cluster': k_list, 'SSE': sse})

plt.figure(figsize=(10,5))
plt.plot(pd.DataFrame(sse)[0], pd.DataFrame(sse)[1], marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')

# plt.savefig("charts/tsne_elbow_method.pdf")

# %%
kmeans_tsne_scale = KMeans(n_clusters=6, n_init=100, max_iter=400, init='k-means++', random_state=42).fit(tsne_df_scale)
print('K-Means t-SNE Scaled Silhouette Score: {}'.format(silhouette_score(tsne_df_scale, kmeans_tsne_scale.labels_, metric='euclidean')))
labels_tsne_scale = kmeans_tsne_scale.labels_
clusters_tsne_scale = pd.concat([tsne_df_scale, pd.DataFrame({'tsne_clusters':labels_tsne_scale})], axis=1)

# %%
# from mpl_toolkits.mplot3d import Axes3D

# fig = plt.figure(figsize=(8,8))
# ax3 = Axes3D(fig) #, auto_add_to_figure=False#, palette=sns.color_palette(pal[:6])
# fig.add_axes(ax3)

# sc = ax.scatter(clusters_tsne_scale.tsne1, clusters_tsne_scale.tsne2, clusters_tsne_scale.tsne3, cmap='tsne_clusters', s=10, marker='o')
# ax3.set_xlabel('t-SNE1')
# ax3.set_ylabel('t-SNE2')
# ax3.set_zlabel('t-SNE3')

# plt.savefig("charts/kmeans_from_tsne_3d.pdf", bbox_inches='tight')

# %%
plt.figure(figsize = (8,8))
sns.scatterplot(x='t-SNE1', y='t-SNE2', data=clusters_tsne_scale, hue='tsne_clusters', s=10, linewidth=0, palette=sns.color_palette(pal[:6]))
plt.legend('', frameon=False)

# plt.savefig('charts/kmeans_from_tsne.pdf')

# %%
df_ = address_clustering.copy()
df_['tsne_clusters'] = clusters_tsne_scale['tsne_clusters'].values
df_.loc[:, ~df_.columns.isin(['date_month', 'address', 'rank'])].groupby('tsne_clusters').mean()

# %%
pd.Series([y for x,y in sorted(Counter(df_[df_['tsne_clusters'] == 2]['date_month']).items())]).plot()

# %%
df = df_.loc[df_['tsne_clusters'] == 2, df_.columns.isin(['address', 'date_month'])].copy()
# df.to_csv('eth_exchanges_monthly.csv', index=False, header = False)
