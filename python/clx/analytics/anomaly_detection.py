import cudf
import cuml


def cluster(feature_dataframe, min_samples=3, eps=0.3):
    """
        Pass a feature dataframe to this function to detect anomalies in your datafeature dataframe and outputs associated labels 0,1,-1.
        Parameters
        ----------
        :param feature_dataframe: Feature dataframe to be used for clustering
        :type feature_dataframe: cudf.DataFrame

        Examples
        --------
        >>> import cudf
        >>> from clx.analytics.cluster import anomaly_detection
        >>> df = cudf.DataFrame(
        >>>     {
        >>>         "time": [1, 2, 3, 4, 5, 6, 7],
        >>>         "user": ["u1", "u2", "u3", "u1", "u1", "u2", "u1"],
        >>>         "computer": ["c1", "c2", "c3", "c1", "c2", "c3", "c1"],
        >>>     }
        >>> )
        >>> feature_df = clx.features.frequency(df, entity_id="user", feature_id="computer")
        >>> labels = anomaly_detection(feature_df)
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
