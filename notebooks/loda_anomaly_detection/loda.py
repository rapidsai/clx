import cupy as cp


class Loda(object):

    def __init__(self, n_bins=None, n_random_cuts=100):
        """
        n_bins (int): Number of bins for each histogram. If None a heurstic is
                    used to compute the number of bins. 
        n_random_cuts (int): Number of projection to use. 
        """
        self.n_bins = n_bins
        self.n_random_cuts = n_random_cuts
        self.weights = cp.ones(n_random_cuts) / n_random_cuts
        self.projections = None
        self.histograms = None
        self.limits = None

    def fit(self, X, y=None):
        """Fit training data and  construct histogram.
        The type of histogram is 'regular', and right-open
        Note: If n_bins=None, the number of breaks is being computed as in:
        L. Birge, Y. Rozenholc, How many bins should be put in a regular
        histogram? 2006.
        X (cupy.ndarray) : NxD training sample. 
             
        """
        nrows, n_components = X.shape
        if not self.n_bins:
            self.n_bins = int(1 * (nrows ** 1) * (cp.log(nrows) ** -1))
        n_nonzero_components = cp.sqrt(n_components)
        n_zero_components = n_components - cp.int(n_nonzero_components)

        self.projections = cp.random.randn(self.n_random_cuts, n_components)
        self.histograms = cp.zeros([self.n_random_cuts, self.n_bins])
        self.limits = cp.zeros((self.n_random_cuts, self.n_bins + 1))
        for i in range(self.n_random_cuts):
            rands = cp.random.permutation(n_components)[:n_zero_components]
            self.projections[i, rands] = 0.
            projected_data = self.projections[i, :].dot(X.T)
            self.histograms[i, :], self.limits[i, :] = cp.histogram(
                projected_data, bins=self.n_bins, density=False)
            self.histograms[i, :] += 1e-12
            self.histograms[i, :] /= cp.sum(self.histograms[i, :])
        return self

    def score(self, X):
        if cp.ndim(X) < 2:
            X = X.reshape(1, -1)
        pred_scores = cp.zeros([X.shape[0], 1])
        for i in range(self.n_random_cuts):
            projected_data = self.projections[i, :].dot(X.T)
            inds = cp.searchsorted(self.limits[i, :self.n_bins - 1],
                                   projected_data, side='left')
            pred_scores[:, 0] += -self.weights[i] * cp.log(
                self.histograms[i, inds])
        pred_scores /= self.n_random_cuts
        return pred_scores.ravel()

    def instance_score(self, x, projection_index):
        """
            Return scores from selected projection index.
            x (cupy.ndarray) : D x 1 feature instance.
        """
        if cp.ndim(x) < 2:
            x = x.reshape(1, -1)
        pred_scores = cp.zeros([x.shape[0], len(projection_index)])
        for i in projection_index:
            projected_data = self.projections[i, :].dot(x.T)
            inds = cp.searchsorted(self.limits[i, :self.n_bins - 1],
                                   projected_data, side='left')
            pred_scores[:, i] = -self.weights[i] * cp.log(
                self.histograms[i, inds])
        return pred_scores

    def t_test(self, with_sample, without_sample):
        """
        compute one-tailed two-sample t-test with a test statistics according to
            t_j: \frac{\mu_j - \bar{\mu_j}}{\sqrt{\frac{s^2_j}{\norm{I_j}} +
            \frac{\bar{s^2_j}}{\norm{\bar{I_j}}}}}
        """
        return (cp.mean(with_sample) - cp.mean(without_sample)) /\
            cp.sqrt(cp.var(with_sample)**2 / len(with_sample) +
                    cp.var(without_sample)**2 / len(without_sample))

    def explain(self, x, scaled=True):
        """
        Return explanation of the anomalies based on t-scores.
        """
        if cp.ndim(x) < 2:
            x = x.reshape(1, -1)
        ranked_feature_importance = cp.zeros([x.shape[1], 1])

        for feature in range(x.shape[1]):
            # find all projections without the feature j and with feature j
            index_selected_feature = cp.where(
                self.projections[:, feature] != 0)[0]
            index_not_selected_feature = cp.where(
                self.projections[:, feature] == 0)[0]
            scores_with_feature = self.instance_score(x,
                                                      index_selected_feature)
            scores_without_feature = self.instance_score(
                x, index_not_selected_feature)
            ranked_feature_importance[feature, 0] = self.t_test(
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
