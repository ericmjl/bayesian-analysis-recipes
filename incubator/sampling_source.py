import numpy as np
from scipy.stats import norm, expon


class Distribution:
    def __init__(self):
        return NotImplementedError("Please implement init.")

    def pdf(self, x):
        """Evaluate total probability of data x."""
        return np.prod(self.dist.pdf(x))

    def logpdf(self, x):
        """Evaluate total log probability of data x."""
        return np.sum(self.dist.logpdf(x))

    def draws(self, n):
        """Draw n samples from the distribution"""
        return self.dist.rvs(n)


class Normal(Distribution):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        self.dist = norm(mu, sigma)


class Exponential(Distribution):
    def __init__(self, lam):
        self.lam = lam
        self.dist = expon(scale=lam)
