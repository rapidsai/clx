import cudf
import cuml


def dbscan(feature_dataframe, min_samples=3, eps=0.3):
    """
    Pass a feature dataframe to this function to detect anomalies in your feature dataframe. This function uses ``cuML`` DBSCAN to detect anomalies
    and outputs associated labels 0,1,-1.

    Parameters
    ----------
    :param feature_dataframe: Feature dataframe to be used for clustering
    :type feature_dataframe: cudf.DataFrame
    :param min_samples: Minimum samples to use for dbscan
    :type min_samples: int
    :param eps: Max distance to use for dbscan
    :type eps: float

    Examples
    --------
    >>> import cudf
    >>> import clx.features
    >>> import clx.analytics.anomaly_detection
    >>> df = cudf.DataFrame(
    >>>         {
    >>>             "time": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    >>>             "user": ["u1","u1","u1","u1","u1","u1","u1","u1","u1","u1","u5","u4","u2","u3"],
    >>>             "computer": ["c1","c2","c3","c1","c2","c3","c1","c1","c2","c3","c1","c1","c5","c6"],
    >>>         }
    >>>     )
    >>> feature_df = clx.features.frequency(df, entity_id="user", feature_id="computer")
    >>> labels = clx.analytics.anomaly_detection.dbscan(feature_df, min_samples=2, eps=0.5)
    >>> labels
        0   -1
        1   -1
        2   -1
        dtype: int32
    """
    dbscan = cuml.cluster.DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(feature_dataframe)
    # return anomalies only
    labels = cudf.Series(dbscan.labels_)
    anomalies = labels[labels == -1]
    return anomalies
