from .__init__ import BayesianModel
import theano
import theano.tensor as tt
import numpy as np
import pymc3 as pm


class ForestCoverModel(BayesianModel):

    def __init__(self, n_hidden):
        super(ForestCoverModel, self).__init__()
        self.n_hidden = n_hidden

    def create_model(self, X=None, y=None):
        if X:
            num_samples, self.num_pred = X.shape

        if y:
            num_samples, self.num_out = Y.shape

        model_input = theano.shared(np.zeros(shape=(1, self.num_pred)))
        model_output = theano.shared(np.zeros(shape=(1,self.num_out)))

        self.shared_vars = {
            'model_input': model_input,
            'model_output': model_output
        }

        with pm.Model() as model:
            # Define weights
            weights_1 = pm.Normal('w_1', mu=0, sd=1,
                                  shape=(self.num_pred, self.n_hidden))
            weights_2 = pm.Normal('w_2', mu=0, sd=1,
                                  shape=(self.n_hidden, self.n_hidden))
            weights_out = pm.Normal('w_out', mu=0, sd=1,
                                    shape=(self.n_hidden, self.num_outs))

            # Define activations
            acts_1 = tt.tanh(tt.dot(model_input, weights_1))
            acts_2 = tt.tanh(tt.dot(acts_1, weights_2))
            acts_out = tt.nnet.softmax(tt.dot(acts_2, weights_out))  # noqa

            # Define likelihood
            out = pm.Multinomial('likelihood', n=1, p=acts_out,
                                 observed=model_output)

        return model


    def fit(self, X, y, n=200000, batch_size=10):
        """
        Train the Bayesian NN model.
        """
        num_samples, self.num_pred = X.shape
        _, self.num_out = y.shape

        if self.cached_model is None:
            self.cached_model = self.create_model()

        with self.cached_model:
            minibatches = {
                self.shared_vars['model_input']: pm.Minibatch(X, batch_size=batch_size),
                self.shared_vars['model_output']: pm.Minibatch(y, batch_size=batch_size),
            }
            self._inference(minibatches, n)

        return self
