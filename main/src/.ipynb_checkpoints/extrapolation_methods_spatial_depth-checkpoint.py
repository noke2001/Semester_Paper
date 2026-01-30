import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis
from sklearn.metrics.pairwise import euclidean_distances
from umap import UMAP
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import gower
from sklearn_extra.cluster import KMedoids

# rpy2 imports for spatial depth
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

def mahalanobis_split(X, quantile=0.8):
    mean = np.mean(X, axis=0)
    cov = np.cov(X.T)
    inv_cov = np.linalg.inv(cov)
    distances = X.apply(lambda row: mahalanobis(row, mean, inv_cov), axis=1)
    threshold = distances.quantile(quantile)
    far = distances[distances >= threshold].index
    close = distances[distances < threshold].index
    return close, far

def umap_split(X, quantile=0.8, n_components=2, random_state=42):
    umap = UMAP(n_components=n_components, random_state=random_state, init='random')
    X_umap = umap.fit_transform(X)
    euclidean_dist_matrix = np.mean(euclidean_distances(X_umap), axis=1)
    distances = pd.Series(euclidean_dist_matrix, index=X.index)
    threshold = distances.quantile(quantile)
    far = distances[distances >= threshold].index
    close = distances[distances < threshold].index
    return close, far

def kmeans_split(X: pd.DataFrame, n_clusters: int = 20, ideal_fraction: float = 0.2, random_state: int = 42) -> tuple:

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    kmeans.fit(X)
    
    global_centroid = X.mean(axis=0)
    
    centroid_distances = np.linalg.norm(kmeans.cluster_centers_ - global_centroid.values, axis=1)
    
    cluster_df = pd.DataFrame({
        'cluster': np.arange(n_clusters),
        'centroid_distance': centroid_distances
    })
    cluster_df.sort_values('centroid_distance', ascending=False, inplace=True)
    
    ideal_count = len(X) * ideal_fraction
    cumulative_count = 0
    selected_clusters = []
    
    for _, row in cluster_df.iterrows():
        cluster_label = int(row['cluster'])
        count = np.sum(kmeans.labels_ == cluster_label)
        cumulative_count += count
        selected_clusters.append(cluster_label)
        if cumulative_count >= ideal_count:
            break

    mask_far = np.isin(kmeans.labels_, selected_clusters)
    far_idx = X.index[mask_far]
    close_idx = X.index[~mask_far]
    
    return close_idx, far_idx


def random_split(X, y, test_size=0.2, val_size=0.2, random_state=42):
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=val_size, random_state=random_state
    )
    
    return X_train, X_val, y_train, y_val, X_test, y_test

def gower_split(X: pd.DataFrame, quantile: float = 0.8) -> tuple:
    X_float = X.copy()
    num_cols = X_float.select_dtypes(include=[np.number]).columns
    X_float[num_cols] = X_float[num_cols].astype(float)
    gower_matrix = gower.gower_matrix(X_float)

    
    avg_distances = np.mean(gower_matrix, axis=1)
    distances = pd.Series(avg_distances, index=X.index)
    
    # Determine the threshold based on the provided quantile
    threshold = distances.quantile(quantile)
    
    far_idx = distances[distances >= threshold].index
    close_idx = distances[distances < threshold].index
    
    return close_idx, far_idx



def kmedoids_split(
    X: pd.DataFrame, 
    n_clusters: int = 20, 
    ideal_fraction: float = 0.2, 
    random_state: int = 0
) -> tuple:

    for col in X.select_dtypes(include=['category']).columns:
        X[col] = X[col].astype('object')
    X_float = X.copy()
    num_cols = X_float.select_dtypes(include=[np.number]).columns
    X_float[num_cols] = X_float[num_cols].astype(float)
    gower_dist_matrix = gower.gower_matrix(X_float)

    kmedoids = KMedoids(
        n_clusters=n_clusters, 
        random_state=random_state, 
        metric='precomputed', 
        init='k-medoids++'
    ).fit(gower_dist_matrix)
    
    ideal_count = len(X) * ideal_fraction
    clusters = np.unique(kmedoids.labels_)
    avg_dist_list = []
    counts = []
    for i in clusters:
        cluster_data = X_float.loc[kmedoids.labels_ == i]
        dist_matrix = gower.gower_matrix(cluster_data, X_float)
        avg_dist = np.mean(dist_matrix)
        avg_dist_list.append(avg_dist)
        counts.append(cluster_data.shape[0])

    cluster_df = pd.DataFrame({
        'cluster': clusters,
        'gower_dist': avg_dist_list,
        'count': counts
    })

    cluster_df = cluster_df.sort_values('gower_dist', ascending=False)

    cluster_df['cumulative_count'] = cluster_df['count'].cumsum()
    cluster_df['abs_diff'] = np.abs(cluster_df['cumulative_count'] - ideal_count)

    idx_to_select = cluster_df['abs_diff'].idxmin()
    selected_clusters = cluster_df.loc[:idx_to_select, 'cluster'].tolist()
    
    labels = pd.Series(kmedoids.labels_, index=X.index)
    far_idx = labels[labels.isin(selected_clusters)].index
    close_idx = labels[~labels.isin(selected_clusters)].index

    return close_idx, far_idx

def spatial_depth_split(
     X: pd.DataFrame,
     quantile: float = 0.2
 ) -> tuple:
     """
     Split DataFrame X into close and far indices based on spatial depth using R's ddalpha package.
     Far indices correspond to lowest spatial depth up to the specified quantile.
     """
     # activate pandas conversion for rpy2
     pandas2ri.activate()
     # import the ddalpha package and spatial depth function
     ddalpha = importr('ddalpha')
     spatialDepth = robjects.r['depth.spatial']

     # calculate spatial depth for each point
     depth_vals = spatialDepth(X, X)
     depth_series = pd.Series(depth_vals, index=X.index)

     # determine threshold for far points
     threshold = depth_series.quantile(quantile)
     far_idx = depth_series[depth_series <= threshold].index
     close_idx = depth_series[depth_series > threshold].index

     return close_idx, far_idx






