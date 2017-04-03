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
