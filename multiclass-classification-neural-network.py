import pymc3 as pm
import numpy as np
import theano.tensor as tt
import theano
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import patsy
from sklearn.preprocessing import scale, normalize


def make_nn(ann_input, ann_output, n_hidden):
    """
    Makes a feed forward neural network with n_hidden layers for doing multi-
    class classification.
    
    Feed-forward networks are easy to define, so I have not relied on any 
    other Deep Learning frameworks to define the neural network here.
    """
    init_1 = np.random.randn(ann_input.shape[1], n_hidden)
    init_2 = np.random.randn(n_hidden, n_hidden)
    init_out = np.random.randn(n_hidden, ann_output.shape[1])
    
    with pm.Model() as nn_model:
        # Define weights
        weights_1 = pm.Normal('w_1', mu=0, sd=1, 
                              shape=(ann_input.shape[1], n_hidden),
                              testval=init_1)
        weights_2 = pm.Normal('w_2', mu=0, sd=1,
                              shape=(n_hidden, n_hidden),
                              testval=init_2)
        weights_out = pm.Normal('w_out', mu=0, sd=1, 
                                shape=(n_hidden, ann_output.shape[1]),
                                testval=init_out)

        # Define activations
        acts_1 = pm.Deterministic('activations_1', 
                                  tt.tanh(tt.dot(ann_input, weights_1)))
        acts_2 = pm.Deterministic('activations_2', 
                                  tt.tanh(tt.dot(acts_1, weights_2)))
        acts_out = pm.Deterministic('activations_out', 
                                    tt.nnet.softmax(tt.dot(acts_2, weights_out)))  # noqa
        
        # Define likelihood
        out = pm.Multinomial('likelihood', n=1, p=acts_out, 
                             observed=ann_output)
        
    return nn_model

print('Reading in data.')
df = pd.read_csv('datasets/covtype_preprocess.csv', index_col=0)
df['Cover_Type'] = df['Cover_Type'].apply(lambda x: str(x))

output_col = 'Cover_Type'
input_cols = [c for c in df.columns if c != output_col]
input_formula = ''.join(c + ' + ' for c in input_cols)
input_formula = input_formula + '-1'

X = patsy.dmatrix(formula_like=input_formula, 
                  data=df, 
                  return_type='dataframe')
Y = patsy.dmatrix(formula_like='Cover_Type -1',
                  data=df,
                  return_type='dataframe')
print(X.shape, Y.shape)

downsampled_targets = []

for i in range(1, 7+1):
    # print(f'target[{i}]')
    target = Y[Y['Cover_Type[{i}]'.format(i=i)] == 1]
    # print(len(target))
    downsampled_targets.append(target.sample(2747))
    
mms = MinMaxScaler()
X_tfm = pm.floatX(mms.fit_transform(X[input_cols]))

Y_downsamp = pd.concat(downsampled_targets)
Y_downsamp = pm.floatX(Y_downsamp)
X_downsamp = X_tfm[Y_downsamp.index]
X_downsamp = pm.floatX(X_downsamp)

print('Making neural network...')
n_hidden = 20  # define the number of hidden units
model = make_nn(X_downsamp, Y_downsamp, n_hidden=n_hidden)

print('Fitting with ADVI.')
with model:
    s = theano.shared(pm.floatX(1.1))
    inference = pm.ADVI(cost_part_grad_scale=s)
    approx = pm.fit(10000, method=inference)

print('MCMC sampling.')
with model:
    trace = approx.sample(5000)

print('Sampling from posterior.')
with model:
    samp_ppc = pm.sample_ppc(trace, samples=100)

print('Predictions.')
preds_proba = samp_ppc['likelihood'].mean(axis=0)
preds = (preds_proba == np.max(preds_proba, axis=1, keepdims=True)) * 1
# plt.pcolor(preds)

from sklearn.metrics import classification_report

print('Classification report.')
print(classification_report(Y_downsamp, preds))
