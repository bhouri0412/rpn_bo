from jax import numpy as np
from jax import jit, vmap, random
from jax.scipy.stats import multivariate_normal, uniform
import numpy as onp
from scipy.stats import gaussian_kde
from sklearn import mixture
from pyDOE import lhs
from KDEpy import FFTKDE

def fit_kde(predict_fn, prior_pdf, bounds, num_samples=10000, bw=None):
    onp.random.seed(1)
    lb, ub = bounds
    dim = lb.shape[0]
    X = lb + (ub-lb)*lhs(dim, num_samples)
    y = predict_fn(X)
    weights = prior_pdf(X)
    y, weights = onp.array(y), onp.array(weights)
    y = y.flatten()
    if bw is None:
        try:
            sc = gaussian_kde(y, weights=weights)
            bw = onp.sqrt(sc.covariance).flatten()[0]
        except:
            bw = 1.0
        if bw < 1e-8:
            bw = 1.0
    kde_pdf_x, kde_pdf_y = FFTKDE(bw=bw).fit(y, weights).evaluate()
    return kde_pdf_x, kde_pdf_y

def fit_gmm(predict_fn, prior_pdf, bounds, num_samples, num_comp):
    onp.random.seed(0)
    lb, ub = bounds
    dim = lb.shape[0]
    # Evaluate input prior
    X = lb + (ub-lb)*lhs(dim, num_samples)
    p_x = prior_pdf(X)[:,None]
    # Interpolate output KDE
    y = predict_fn(X)
    kde_pdf_x, kde_pdf_y = fit_kde(predict_fn, prior_pdf, bounds)
    p_y = np.clip(np.interp(y, kde_pdf_x, kde_pdf_y), a_min=0.0) + 1e-8
    # Weights
    weights = p_x/p_y
    # Rescale weights as probability distribution
    weights = onp.array(weights, dtype = onp.float64)
    weights = weights / onp.sum(weights)
    # Scale inputs to [0, 1]^D
    lb, ub = bounds
    X = (X - lb) / (ub - lb)
    # Sample from analytical w
    indices = np.arange(num_samples)
    idx = onp.random.choice(indices, num_samples, p=weights.flatten())
    X_train = X[idx] 
    # fit GMM
    clf = mixture.GaussianMixture(n_components=num_comp,
                                  covariance_type='full')
    clf.fit(onp.array(X_train, dtype=np.float64))
    out = (np.array(clf.weights_), 
            np.array(clf.means_), 
            np.array(clf.covariances_))
    return out

def output_weights(predict_fn, prior_pdf, bounds, method='exact', num_samples=10000, num_comp=2):
    # Compute exact likelihood ratio
    if method == 'exact':
        onp.random.seed(0)
        lb, ub = bounds
        dim = lb.shape[0]
        X = lb + (ub-lb)*lhs(dim, num_samples)
        kde_pdf_x, kde_pdf_y = fit_kde(predict_fn, prior_pdf, bounds)
        p_x = lambda x: prior_pdf(x)[:,None]
        p_y = lambda x: np.clip(np.interp(predict_fn(x), kde_pdf_x, kde_pdf_y), a_min=0.0) + 1e-8
        ratio = lambda x: p_x(x)/p_y(x)
        volume = np.prod(ub-lb)
        norm_const = np.mean(ratio(X))*volume
        def compute_w(x):
            w = ratio(x)/norm_const
            return w.flatten()
    # GMM approximation
    elif method == 'gmm':
        gmm_vars = fit_gmm(predict_fn, prior_pdf, bounds, num_samples, num_comp)
        def compute_w(x):
            # expects normalized inputs
            weights, means, covs = gmm_vars
            lb, ub = bounds
            x = (x - lb) / (ub - lb)
            gmm_mode = lambda w, mu, cov:  w*multivariate_normal.pdf(x, mu, cov)
            w = np.sum(vmap(gmm_mode)(weights, means, covs), axis = 0)
            return w/np.prod(ub-lb)
    elif method == 'None':
        compute_w = lambda x: np.ones(x.shape[0])
    else:
            raise NotImplementedError

    return jit(compute_w)

# Helper functions for computing output-weighted acquisitions
class uniform_prior:
    def __init__(self, lb, ub):
        self.lb = lb
        self.ub = ub
        self.dim = lb.shape[0]
    def sample(self, rng_key, N):
        return self.lb + (self.ub-self.lb)*random.uniform(rng_key, (N, self.dim))
    def pdf(self, x):
        return np.sum(uniform.pdf(x, self.lb, self.ub-self.lb), axis=-1)

class gaussian_prior:
    def __init__(self, mu, cov):
        self.mu = mu
        self.cov = cov
        self.dim = mu.shape[0]
    def sample(self, rng_key, N):
        return random.multivariate_normal(rng_key, self.mu, self.cov, (N,))
    def pdf(self, x):
        return multivariate_normal.pdf(x, self.mu, self.cov)
