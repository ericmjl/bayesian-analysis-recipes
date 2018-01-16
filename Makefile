execute_params = --execute --ExecutePreprocessor.timeout=600

RAWNBS = $(wildcard notebooks/*.ipynb)
HTMLS =  $(patsubst notebooks/%.ipynb, ../docs/%.html, $(rawnbs))

docs:
	jupyter nbconvert notebooks/bayesian-estimation-multi-sample.ipynb --output ../docs/bayesian-estimation-multi-sample.html
	jupyter nbconvert notebooks/degrees-of-freedom.ipynb --output ../docs/degrees-of-freedom.html
	jupyter nbconvert notebooks/dirichlet-multinomial-bayesian-proportions.ipynb --output ../docs/dirichlet-multinomial-bayesian-proportions.html
	jupyter nbconvert notebooks/mixture-model.ipynb --output ../docs/mixture-model.html
	jupyter nbconvert notebooks/hierarchical-modelling.ipynb --output ../docs/hierarchical-modelling.html
	jupyter nbconvert notebooks/poisson-regression.ipynb --output ../docs/poisson-regression.html

html: $(HTMLS)
	jupyter nbconvert $< --output $@
