from gmr import GMM, kmeansplusplus_initialization, covariance_initialization
import numpy as np

# You can also create custom classes or extend existing ones
class IterativeGMM(GMM):
    def from_samples_iterative(self, X, R_diff=1e-4, n_iter=100, init_params="random",
                 oracle_approximating_shrinkage=False):

        """MLE of the mean and covariance.

        MODIFY WITH THE MATH FROM
        Incremental Learning of Multivariate Gaussian Mixture Models
        by Paulo Martins Engel and Milton Roberto Heinen

        thsi modify code should run iterative GMM. X are the sampels to process one at a time

        Expectation-maximization is used to infer the model parameters. The
        objective function is non-convex. Hence, multiple runs can have
        different results.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Samples from the true distribution.

        R_diff : float
            Minimum allowed difference of responsibilities between successive
            EM iterations.

        n_iter : int
            Maximum number of iterations.

        init_params : str, optional (default: 'random')
            Parameter initialization strategy. If means and covariances are
            given in the constructor, this parameter will have no effect.
            'random' will sample initial means randomly from the dataset
            and set covariances to identity matrices. This is the
            computationally cheap solution.
            'kmeans++' will use k-means++ initialization for means and
            initialize covariances to diagonal matrices with variances
            set based on the average distances of samples in each dimensions.
            This is computationally more expensive but often gives much
            better results.

        oracle_approximating_shrinkage : bool, optional (default: False)
            Use Oracle Approximating Shrinkage (OAS) estimator for covariances
            to ensure positive semi-definiteness.

        Returns
        -------
        self : GMM
            This object.
        """
        n_samples, n_features = X.shape

        if self.priors is None:
            self.priors = np.ones(self.n_components,
                                dtype=float) / self.n_components

        if init_params not in ["random", "kmeans++"]:
            raise ValueError("'init_params' must be 'random' or 'kmeans++' "
                            "but is '%s'" % init_params)

        if self.means is None:
            if init_params == "random":
                indices = self.random_state.choice(
                    np.arange(n_samples), self.n_components)
                self.means = X[indices]
            else:
                self.means = kmeansplusplus_initialization(
                    X, self.n_components, self.random_state)

        if self.covariances is None:
            if init_params == "random":
                self.covariances = np.empty(
                    (self.n_components, n_features, n_features))
                self.covariances[:] = np.eye(n_features)
            else:
                self.covariances = covariance_initialization(
                    X, self.n_components)
                
        # R = np.zeros((n_samples, self.n_components))
        R = np.zeros((1, self.n_components)) ## modified line
        for sample in X:
            R_prev = R
            
            means_prev = self.means ## new line
            # Expectation -> take single sample
            # R = self.to_responsibilities(X)
            R = self.to_responsibilities(sample)  ## modified line

            if np.linalg.norm(R - R_prev) < R_diff:
                if self.verbose:
                    print("EM converged.")
                break

            # Maximization
            w = R.sum(axis=0) + 10.0 * np.finfo(R.dtype).eps
            R_n = R / w
            self.priors = w / w.sum()
            # self.means = R_n.T.dot(X)
            
            for k in range(self.n_components):
                
                sample_m = sample - self.means[k] ## new line
                self.means[k] = self.means[k] +  R_n[:, k].dot(sample_m) ## new line

                mean_diff = self.means[k] - means_prev[k]  ## new line
                self.covariances[k] = self.covariances[k] - mean_diff.dot(mean_diff.T) + R_n[:, k]*(sample_m.dot(sample_m.T) - self.covariances[k])  ## new line
                
                # Xm = X - self.means[k]
                # self.covariances[k] = (R_n[:, k, np.newaxis] * Xm).T.dot(Xm)


            if oracle_approximating_shrinkage:
                n_samples_eff = (np.sum(R_n, axis=0) ** 2 /
                                np.sum(R_n ** 2, axis=0))
                self.apply_oracle_approximating_shrinkage(n_samples_eff)

        return self