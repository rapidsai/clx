import cupy as cp


class Loda:
    """
    Anomaly detection using Lightweight Online Detector of Anomalies (LODA). LODA detects anomalies in a dataset
    by computing the likelihood of data points using an ensemble of one-dimensional histograms.

    :param n_bins: Number of bins for each histogram. If None a heuristic is used to compute the number of bins.
    :type n_bins: int
    :param n_random_cuts: Number of projection to use.
    :type n_random_cuts: int
    """

    def __init__(self, n_bins=None, n_random_cuts=100):
        self._n_bins = n_bins
        self._n_random_cuts = n_random_cuts
        self._weights = cp.ones(n_random_cuts) / n_random_cuts
        self._projections = None
        self._histograms = None
        self._limits = None

    def fit(self, train_data):
        """
        Fit training data and construct histograms.

        :param train_data: NxD training sample
        :type train_data: cupy.ndarray

        Examples
        --------
        >>> from clx.analytics.loda import Loda
        >>> import cupy as cp
        >>> x = cp.random.randn(100,5) # 5-D multivariate synthetic dataset
        >>> loda_ad = Loda(n_bins=None, n_random_cuts=100)
        >>> loda_ad.fit(x)
        """
        nrows, n_components = train_data.shape
        if not self._n_bins:
            self._n_bins = int(1 * (nrows ** 1) * (cp.log(nrows) ** -1))
        n_nonzero_components = cp.sqrt(n_components)
        n_zero_components = n_components - cp.int(n_nonzero_components)

        self._projections = cp.random.randn(self._n_random_cuts, n_components)
        self._histograms = cp.zeros([self._n_random_cuts, self._n_bins])
        self._limits = cp.zeros((self._n_random_cuts, self._n_bins + 1))
        for i in range(self._n_random_cuts):
            rands = cp.random.permutation(n_components)[:n_zero_components]
            self._projections[i, rands] = 0.
            projected_data = self._projections[i, :].dot(train_data.T)
            self._histograms[i, :], self._limits[i, :] = cp.histogram(
                projected_data, bins=self._n_bins, density=False)
            self._histograms[i, :] += 1e-12
            self._histograms[i, :] /= cp.sum(self._histograms[i, :])

    def score(self, input_data):
        """
        Calculate anomaly scores using negative likelihood across n_random_cuts histograms.

        :param input_data: NxD training sample
        :type input_data: cupy.ndarray

        Examples
        --------
        >>> from clx.analytics.loda import Loda
        >>> import cupy as cp
        >>> x = cp.random.randn(100,5) # 5-D multivariate synthetic dataset
        >>> loda_ad = Loda(n_bins=None, n_random_cuts=100)
        >>> loda_ad.fit(x)
        >>> loda_ad.score(x)
        array([0.04295848, 0.02853553, 0.04587308, 0.03750692, 0.05050418,
        0.02671958, 0.03538646, 0.05606504, 0.03418612, 0.04040502,
        0.03542846, 0.02801463, 0.04884918, 0.02943411, 0.02741364,
        0.02702433, 0.03064191, 0.02575712, 0.03957355, 0.02729784,
        ...
        0.03943715, 0.02701243, 0.02880341, 0.04086408, 0.04365477])
        """
        if cp.ndim(input_data) < 2:
            input_data = input_data.reshape(1, -1)
        pred_scores = cp.zeros([input_data.shape[0], 1])
        for i in range(self._n_random_cuts):
            projected_data = self._projections[i, :].dot(input_data.T)
            inds = cp.searchsorted(self._limits[i, :self._n_bins - 1],
                                   projected_data, side='left')
            pred_scores[:, 0] += -self._weights[i] * cp.log(
                self._histograms[i, inds])
        pred_scores /= self._n_random_cuts
        return pred_scores.ravel()

    def explain(self, anomaly, scaled=True):
        """
        Explain anomaly based on contributions (t-scores) of each feature across histograms.

        :param anomaly: selected anomaly from input dataset
        :type anomaly: cupy.ndarray
        :param scaled: set to scale output feature importance scores
        :type scaled: boolean

        Examples
        --------
        >>> loda_ad.explain(x[5]) # x[5] is found anomaly
        array([[1.        ],
        [0.        ],
        [0.69850349],
        [0.91081035],
        [0.78774349]])
        """
        if cp.ndim(anomaly) < 2:
            anomaly = anomaly.reshape(1, -1)
        ranked_feature_importance = cp.zeros([anomaly.shape[1], 1])

        for feature in range(anomaly.shape[1]):
            # find all projections without the feature j and with feature j
            index_selected_feature = cp.where(
                self._projections[:, feature] != 0)[0]
            index_not_selected_feature = cp.where(
                self._projections[:, feature] == 0)[0]
            scores_with_feature = self._instance_score(
                anomaly, index_selected_feature)
            scores_without_feature = self._instance_score(
                anomaly, index_not_selected_feature)
            ranked_feature_importance[feature, 0] = self._t_test(
                scores_with_feature, scores_without_feature)

        if scaled:
            assert cp.max(ranked_feature_importance) != cp.min(
                ranked_feature_importance)
            normalized_score = (ranked_feature_importance - cp.min(
                ranked_feature_importance)) / (
                cp.max(ranked_feature_importance) - cp.min(
                    ranked_feature_importance))
            return normalized_score
        else:
            return ranked_feature_importance

    def _instance_score(self, x, projection_index):
        """
            Return scores from selected projection index.
            x (cupy.ndarray) : D x 1 feature instance.
        """
        if cp.ndim(x) < 2:
            x = x.reshape(1, -1)
        pred_scores = cp.zeros([x.shape[0], len(projection_index)])
        for i in projection_index:
            projected_data = self._projections[i, :].dot(x.T)
            inds = cp.searchsorted(self._limits[i, :self._n_bins - 1],
                                   projected_data, side='left')
            pred_scores[:, i] = -self._weights[i] * cp.log(
                self._histograms[i, inds])
        return pred_scores

    def _t_test(self, with_sample, without_sample):
        """
        compute one-tailed two-sample t-test with a test statistics according to
            t_j: \frac{\mu_j - \bar{\mu_j}}{\sqrt{\frac{s^2_j}{\norm{I_j}} +
            \frac{\bar{s^2_j}}{\norm{\bar{I_j}}}}}
        """
        return (cp.mean(with_sample) - cp.mean(without_sample)) /\
            cp.sqrt(cp.var(with_sample)**2 / len(with_sample) + cp.var(without_sample)**2 / len(without_sample))

    def save_model(self, file_path):
        """ This function save model to given location.

        :param file_path: File path to save model.
        :type file_path: string
        """
        cp.savez_compressed(file_path, histograms=self._histograms,
                            limits=self._limits, projections=self._projections)

    @classmethod
    def load_model(cls, file_path):
        """ This function load already saved model and sets cuda parameters.
        :param file_path: File path of a model to load.
        :type filel_path: string
        """

        model = cp.load(file_path)
        histograms = model['histograms']
        projections = model['projections']
        limits = model['limits']
        n_random_cuts = histograms.shape[0]
        n_bins = histograms.shape[1]
        loda = Loda(n_random_cuts=n_random_cuts, n_bins=n_bins)
        loda._histograms = histograms
        loda._limits = limits
        loda._projections = projections
        return loda
