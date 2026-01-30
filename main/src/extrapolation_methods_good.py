import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis
from sklearn.metrics.pairwise import euclidean_distances
from umap import UMAP
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import gower
from sklearn_extra.cluster import KMedoids
from sklearn.preprocessing import StandardScaler

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

def umap_split(X_clean, quantile=0.8, n_components=2, random_state=10):
    X_tmp = X_clean.copy()
    for c in X_tmp.select_dtypes(include=['category','object','string','bool']).columns:
        X_tmp[c] = X_tmp[c].astype('category').cat.codes
    X_tmp = X_tmp.fillna(0)
    umap = UMAP(n_components=2, random_state=10)
    X_umap = umap.fit_transform(X_tmp)
    euclidean_dist_matrix = np.mean(euclidean_distances(X_umap), axis=1)
    distances = pd.Series(euclidean_dist_matrix, index=X_clean.index)
    threshold = np.quantile(distances, quantile)
    far = distances.index[np.where(distances >= threshold)[0]]
    close = distances.index[np.where(distances < threshold)[0]]
    return close, far
    

def kmeans_split(
    X: pd.DataFrame,
    n_clusters: int = 20,
    random_state: int = 0
) -> tuple[pd.Index, pd.Index]:
    mean = np.mean(X.values, axis=0)
    cov = np.cov(X.values.T)
    inv_cov = np.linalg.inv(cov)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    kmeans.fit(X_scaled)

    mahalanobis_dist=[]
    counts=[]
    ideal_len=len(kmeans.labels_)/5
    for i in np.arange(n_clusters):
        counts.append(np.sum(kmeans.labels_== i))
        mean_k= np.mean(X.loc[kmeans.labels_== i,:], axis=0)
        mahalanobis_dist.append(mahalanobis(mean_k, mean, inv_cov))

    dist_df=pd.DataFrame(data={'mahalanobis_dist': mahalanobis_dist, 'count': counts}, index=np.arange(n_clusters))
    dist_df=dist_df.sort_values('mahalanobis_dist', ascending=False)
    
    dist_df['cumulative_count']=dist_df['count'].cumsum()
    dist_df['abs_diff']=np.abs(dist_df['cumulative_count']-ideal_len)

    final=(np.where(dist_df['abs_diff']==np.min(dist_df['abs_diff']))[0])[0]
    labelss=dist_df.index[0:final+1].to_list()

    labels=pd.Series(kmeans.labels_).isin(labelss)
    labels.index=X.index

    close_idx=labels.index[np.where(labels==False)[0]]
    far_idx=labels.index[np.where(labels==True)[0]]

    return close_idx, far_idx



def random_split(X, y, test_size=0.2, val_size=0.2, random_state=10):
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=val_size, random_state=random_state
    )
    
    return X_train, X_val, y_train, y_val, X_test, y_test

def gower_split(X: pd.DataFrame, quantile: float = 0.8) -> tuple:
    X_float = X.copy()
    X_float = X_float.fillna(0)
    num_cols = X_float.select_dtypes(include=[np.number]).columns
    if len(num_cols) == 0:
        raise ValueError("No numeric columns found in the DataFrame for Gower distance calculation.")
    X_float[num_cols] = X_float[num_cols].astype(float)
    gower_matrix = gower.gower_matrix(X_float)

    
    avg_distances = np.mean(gower_matrix, axis=1)
    distances = pd.Series(avg_distances, index=X.index)
    
    # Determine the threshold based on the provided quantile
    threshold = distances.quantile(quantile)
    
    far_idx = distances[distances >= threshold].index
    close_idx = distances[distances < threshold].index
    
    return close_idx, far_idx



def kmedoids_split_old(
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

def kmedoids_split(
    X_clean: pd.DataFrame,
    n_clusters: int = 20,
    ideal_fraction: float = 0.2,
    random_state: int = 0
) -> tuple[pd.Index, pd.Index]:
    
    X0 = X_clean.copy()
    X0 = X0.fillna(0)
    for col in X0.select_dtypes(include=['category']):
        X0[col] = X0[col].astype(object)

    num_cols = X0.select_dtypes(include=[np.number]).columns
    X0[num_cols] = X0[num_cols].astype(float)

    D = gower.gower_matrix(X0)

    km = KMedoids(
        n_clusters=n_clusters,
        metric='precomputed',
        init='k-medoids++',
        random_state=random_state
    ).fit(D)

    ideal_count = len(X0) * ideal_fraction
    gower_dists = []
    counts      = []
    for c in range(n_clusters):
        mask = (km.labels_ == c)
        subset = X0.loc[mask]
        D_sub = gower.gower_matrix(subset, X0)
        gower_dists.append(np.mean(D_sub))
        counts.append(mask.sum())


    df = pd.DataFrame({
        'cluster':     np.arange(n_clusters),
        'gower_dist':  gower_dists,
        'count':       counts
    }).sort_values('gower_dist', ascending=False).reset_index(drop=True)
    df['cum_count'] = df['count'].cumsum()
    df['abs_diff']  = (df['cum_count'] - ideal_count).abs()

    cutoff_pos = df['abs_diff'].idxmin() 
    far_clusters = df.loc[:cutoff_pos, 'cluster'].tolist()

    labels = pd.Series(km.labels_, index=X0.index)
    far_idx   = labels[labels.isin(far_clusters)].index
    close_idx = labels[~labels.isin(far_clusters)].index

    return close_idx, far_idx


def spatial_depth_handled(
    X: pd.DataFrame,
    quantile: float = 0.2,
    tol: float = 1e-8,
    fallback_test_size: float = 0.2,
    random_state: int = 0,
) -> tuple:
    variances = X.var(axis=0)
    keep_cols = variances[variances > tol].index
    if len(keep_cols) == 0:
        idx = np.arange(len(X))
        train_idx, test_idx = train_test_split(
            idx, test_size=fallback_test_size, random_state=random_state
        )
        return X.index[train_idx], X.index[test_idx]

    X_sub = X[keep_cols]

    pandas2ri.activate()
    ddalpha      = importr("ddalpha")
    spatialDepth = robjects.r["depth.spatial"]
    depth_vals   = spatialDepth(X_sub, X_sub)
    depths       = pd.Series(np.asarray(depth_vals), index=X_sub.index)

    if depths.isna().any():
        min_d = depths[depths.notna()].min()
        depths.fillna(min_d, inplace=True)

    thresh   = depths.quantile(quantile)
    far_idx  = depths[depths <= thresh].index
    close_idx = depths[depths >  thresh].index

    return close_idx, far_idx


def spatial_depth_split(
     X: pd.DataFrame,
     quantile: float = 0.2
 ) -> tuple:

     pandas2ri.activate()
     ddalpha = importr('ddalpha')
     spatialDepth = robjects.r['depth.spatial']

     depth_vals = spatialDepth(X, X)
     depth_series = pd.Series(depth_vals, index=X.index)

     threshold = depth_series.quantile(quantile)
     far_idx   = depth_series.index[np.where(depth_series <= threshold)[0]]
     close_idx = depth_series.index[np.where(depth_series > threshold)[0]]
     return close_idx, far_idx


