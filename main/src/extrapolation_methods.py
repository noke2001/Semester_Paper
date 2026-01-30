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
from rpy2.robjects.conversion import localconverter

# --- Helper for Numeric Conversion ---
def _to_numeric_representation(X: pd.DataFrame) -> pd.DataFrame:
    """
    Converts a mixed-type DataFrame to a purely numeric one for algorithms 
    like UMAP, KMeans, or Mahalanobis that cannot handle strings.
    1. One-Hot Encodes categorical columns.
    2. Fills NaNs (0 for now, robust enough for splitting).
    3. Standardizes the data.
    """
    # 1. One-Hot Encode
    X_enc = pd.get_dummies(X, drop_first=True)
    
    # 2. Fill NaNs
    X_enc = X_enc.fillna(0)
    
    # 3. Standardize (Crucial for UMAP/KMeans distance calculations)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_enc), index=X.index, columns=X_enc.columns)
    
    return X_scaled

# --- Split Methods ---

def random_split(X, y=None, test_size=0.2, val_size=0.2, random_state=10):
    """
    Standard random split. Handles both X only (2 outputs) and X,y (4 or 6 outputs).
    """
    if y is None:
        # Simple train/test split for indices
        train_idx, test_idx = train_test_split(X.index, test_size=test_size, random_state=random_state)
        return train_idx, test_idx
        
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=val_size, random_state=random_state
    )
    
    return X_train, X_val, y_train, y_val, X_test, y_test

def mahalanobis_split(X, quantile=0.8):
    # Ensure numeric input
    X_num = _to_numeric_representation(X)
    
    mean = np.mean(X_num.values, axis=0)
    cov = np.cov(X_num.values.T)
    
    # Add regularization to covariance matrix to prevent singular matrix errors
    # (common with one-hot encoded variables)
    reg = 1e-6 * np.eye(cov.shape[0])
    try:
        inv_cov = np.linalg.inv(cov + reg)
    except np.linalg.LinAlgError:
        # Fallback to pseudoinverse if still singular
        inv_cov = np.linalg.pinv(cov)

    distances = X_num.apply(lambda row: mahalanobis(row, mean, inv_cov), axis=1)
    threshold = distances.quantile(quantile)
    
    far = distances[distances >= threshold].index
    close = distances[distances < threshold].index
    return close, far

def umap_split(X, quantile=0.8, n_components=2, random_state=10):
    """
    Performs UMAP reduction and splits based on Euclidean distance in the embedding space.
    FIXED: Handles categorical variables via One-Hot Encoding before UMAP.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("Input X for umap_split must be a pandas DataFrame.")
    
    # FIX: UMAP cannot handle strings. Convert to numeric first.
    X_num = _to_numeric_representation(X)
    
    umap = UMAP(n_components=n_components, random_state=random_state)
    X_umap = umap.fit_transform(X_num)
    
    # Calculate average distance of each point to all other points (proxy for centrality)
    # Note: For large datasets, mean(euclidean_distances) is O(N^2). 
    # Optimization: Calculate distance to the centroid of the UMAP embedding instead.
    centroid = np.mean(X_umap, axis=0)
    euclidean_dist = np.linalg.norm(X_umap - centroid, axis=1)
    
    distances = pd.Series(euclidean_dist, index=X.index)
    threshold = np.quantile(distances, quantile)
    
    far = distances.index[np.where(distances >= threshold)[0]]
    close = distances.index[np.where(distances < threshold)[0]]
    return close, far

def kmeans_split(X: pd.DataFrame, n_clusters: int = 20, random_state: int = 0) -> tuple[pd.Index, pd.Index]:
    # Ensure numeric input
    X_num = _to_numeric_representation(X)

    mean = np.mean(X_num.values, axis=0)
    cov = np.cov(X_num.values.T)
    reg = 1e-6 * np.eye(cov.shape[0])
    try:
        inv_cov = np.linalg.inv(cov + reg)
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.pinv(cov)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    kmeans.fit(X_num) # X_num is already scaled by _to_numeric_representation

    mahalanobis_dist=[]
    counts=[]
    ideal_len=len(kmeans.labels_)/5
    
    for i in np.arange(n_clusters):
        counts.append(np.sum(kmeans.labels_== i))
        # Get mean of the cluster in the original (numeric) space
        mean_k = np.mean(X_num.loc[kmeans.labels_== i,:], axis=0)
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

def gower_split(X: pd.DataFrame, quantile: float = 0.8) -> tuple:
    # Gower handles mixed types natively, but we ensure correct dtypes
    X_tmp = X.copy()
    num_cols = X_tmp.select_dtypes(include=[np.number]).columns
    X_tmp[num_cols] = X_tmp[num_cols].astype(float)
    
    for c in X_tmp.columns.difference(num_cols):
        X_tmp[c] = X_tmp[c].astype('object')

    # Gower matrix computation (O(N^2))
    D = gower.gower_matrix(X_tmp)
    avg_dist = D.mean(axis=1)
    
    dist_series = pd.Series(avg_dist, index=X.index)
    thresh = dist_series.quantile(quantile)

    close_idx = dist_series[dist_series <  thresh].index
    far_idx   = dist_series[dist_series >= thresh].index
    return close_idx, far_idx

def kmedoids_split(
    X: pd.DataFrame, 
    n_clusters: int = 20, 
    ideal_fraction: float = 0.2, 
    random_state: int = 0
) -> tuple[pd.Index, pd.Index]:
    
    X0 = X.copy()
    X0 = X0.fillna(0) # Simple fill for Gower
    
    # Ensure types for Gower
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
        # Calculate mean gower distance of this cluster to the whole dataset
        # We can extract the sub-block from the precomputed matrix D
        # Indices of this cluster
        cluster_indices = np.where(mask)[0]
        sub_D = D[cluster_indices, :]
        gower_dists.append(np.mean(sub_D))
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

def spatial_depth_split(X: pd.DataFrame, quantile: float = 0.2) -> tuple:
    # 1. Prepare Data
    X_num = _to_numeric_representation(X)

    # --- CRITICAL FIX: Sanitize Column Names for R ---
    # R is very strict about column names. Python's One-Hot encoding often leaves 
    # characters like spaces, '<', '>', or brackets that cause R to crash.
    clean_cols = [
        str(c).replace(' ', '_')
              .replace('(', '.')
              .replace(')', '.')
              .replace('[', '.')
              .replace(']', '.')
              .replace('<', 'lt')
              .replace('>', 'gt') 
        for c in X_num.columns
    ]
    X_num.columns = clean_cols

    # 2. Import R packages (imports are lightweight if already loaded)
    # Note: We REMOVED pandas2ri.activate()
    ddalpha = importr('ddalpha')
    spatialDepth = robjects.r['depth.spatial']

    # 3. Use the Context Manager for Conversion
    # The conversion context must be active specifically when passing data to R
    with localconverter(robjects.default_converter + pandas2ri.converter):
        # Explicitly convert pandas DF to R DF
        r_X = pandas2ri.py2rpy(X_num)
        
        # Call the R function
        depth_vals_r = spatialDepth(r_X, r_X)
        
        # Convert result back to numpy immediately
        depth_vals = np.array(depth_vals_r)

    # 4. Process results
    # Flatten is needed because R sometimes returns a column-vector (N, 1)
    depth_series = pd.Series(depth_vals.flatten(), index=X.index)
    
    # Fill NaNs if any returned from R (robustness)
    if depth_series.isna().any():
        min_d = depth_series[depth_series.notna()].min()
        depth_series.fillna(min_d, inplace=True)

    threshold = depth_series.quantile(quantile)
    
    # "Far" are the outliers (low depth), "Close" are central (high depth)
    # Note: Depth is higher in the center. 
    # If you want 'close' to be the center, it's depth > threshold
    # If you want 'far' to be outliers, it's depth <= threshold
    far_idx   = depth_series.index[np.where(depth_series <= threshold)[0]]
    close_idx = depth_series.index[np.where(depth_series > threshold)[0]]
    
    return close_idx, far_idx