# bayesian-analysis-recipes

## introduction

I've recently been inspired by how flexible and powerful Bayesian statistical analysis can be. Yet, as with many things, flexibility often means a tradeoff with ease-of-use. I think having a cookbook of code that can be used in a number of settings can be extremely helpful for bringing Bayesian methods to a more general setting!

## goals

My goal here is to have one notebook per model. In each notebook, you should end up finding:

- The kind of problem that is being tackled here.
- A description of how the data should be structured.
- An example data table. It generally will end up being **[tidy](http://vita.had.co.nz/papers/tidy-data.pdf)** data.
- PyMC3 code for the model; in some notebooks, there may be two versions of the same model.
- Examples on how to report findings from the MCMC-sampled posterior.

It is my hope that these recipes will be useful for you!

## (hypo)thesis

My hypothesis here follows the Pareto principle: a large fraction of real-world problems can essentially be boiled down to a few categories of problems, which have a Bayesian interpretation.

In particular, I have this hunch that commonly-used methods like ANOVA can be replaced by conceptually simpler and much more interpretable Bayesian alternatives, like John Kruschke's BEST (**B**ayesian **E**stimation **S**upersedes the **T**-test). For example, ANOVA only tests whether means of multiple treatment groups are the same or not... but BEST gives us the estimated posterior distribution over each of the treatment groups, assuming each treatment group is identical. Hence, richer information can be gleaned: we can, given the data at hand, make statements about how any particular pair of groups are different, without requiring additional steps such as multiple hypothesis corrections etc.

## further reading/watching/listening

Books:

- [Bayesian Methods for Hackers](https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers)
- [Think Bayes](http://greenteapress.com/wp/think-bayes/)

Papers:

- [Bayesian Estimation Supersedes the t-Test](http://www.indiana.edu/~kruschke/BEST/BEST.pdf)

Videos:
- [Computational Statistics I @ SciPy 2015](https://www.youtube.com/watch?v=fMycLa1bsno)
- [Computational Statistics II @ SciPy 2015](https://www.youtube.com/watch?v=heFaYLKVZY4)
- [Bayesian Statistical Analysis with Python @ PyCon 2017](https://www.youtube.com/watch?v=p1IB4zWq9C8)
- [Bayesian Estimation Supersedes the t-Test](https://www.youtube.com/watch?v=fhw1j1Ru2i0)

## got feedback?

There's a few ways you can help make this repository an awesome one for Bayesian method learners out there.

1. **If you have a question:** Post a [GitHub issue](https://github.com/ericmjl/bayesian-analysis-recipes/issues) with your question. I'll try my best to respond.
1. **If you have a suggested change:** Submit a [pull request](https://github.com/ericmjl/bayesian-analysis-recipes/pulls) detailing the change and why you think it's important. Keep it simple, no need to have essay-length justifications.
