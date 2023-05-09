import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'

from jax import vmap, random, jit
from jax import numpy as np
from jax.scipy.special import logsumexp
from jax.nn import relu
import numpy as onp

from rpn_bo_models import EnsembleRegression
from rpn_bo_dataloaders import BootstrapLoader
from rpn_bo_acquisitions import MCAcquisition
from rpn_bo_utilities import uniform_prior

onp.random.seed(1234)

# Helper functions
normalize = vmap(lambda x, mu, std: (x-mu)/std, in_axes=(0,0,0))
denormalize = vmap(lambda x, mu, std: x*std + mu, in_axes=(0,0,0))

# vectorial input space dimension and its search space
dim = 4
lb = -np.ones((dim,))
ub = np.ones((dim,))
p_x = uniform_prior(lb, ub)
bounds = (lb, ub)

# vectorial output space dimension
N_y = 64
xx, yy = np.meshgrid( np.arange(N_y) / N_y, np.arange(N_y) / N_y )
dim_y = 16*N_y**2

# initial training space
case = 'results/optical_interferometer_MLP'
prev = 0
X = np.load(case+'/X_loc_'+str(prev)+'.npy')
y = np.load(case+'/y_loc_'+str(prev)+'.npy')
opt = np.load(case+'/opt_'+str(prev)+'.npy')
N = X.shape[0]

#### RPN-BO hyperparameters ####
num_restarts_acq = 500
q1 = 1
ensemble_size = 32
batch_size = N
fraction = 1
layers = [dim, 64, 64, dim_y]
nIter_RPN = 5000
options = {'criterion': 'LCB', # 'TS' 'LCB', EI
           'kappa': 2.0,
           'weights': None, # exact gmm None
           }

train_key = random.PRNGKey(0)
key_TS = random.PRNGKey(123)
    
print('Run %s, Nx %s, Ny %s, init %s, best %s' % (str(prev), X.shape[0], y.shape[0], opt[0], np.min(opt)))

# Change random seed for different optimization iterations and different random independent runs
train_key = random.split(train_key, 2)[0]
key_TS = random.split(key_TS, 2)[0]
for i in range(prev):
    for i in range(85):
        train_key = random.split(train_key, 2)[0]
        key_TS = random.split(key_TS, 2)[0]
for i in range(N-15):
    train_key = random.split(train_key, 2)[0]
    key_TS = random.split(key_TS, 2)[0]

# Create data set
dataset = BootstrapLoader(X, y, batch_size, ensemble_size, fraction, 0, rng_key=train_key)
(mu_X, sigma_X), (mu_y, sigma_y) = dataset.norm_const

# Initialize model
train_key = random.split(train_key, 2)[0]
model = EnsembleRegression(layers, ensemble_size, train_key, relu)

# Train model
model.train(dataset, nIter=nIter_RPN)

# prediction function using trained RPN
# mapping vectorial input to scalar obective value
@jit
def predict(x):
    # accepts and returns un-normalized data
    x = np.tile(x[np.newaxis,:,:], (ensemble_size, 1, 1))
    x = normalize(x, mu_X, sigma_X)
    params = vmap(model.get_params)(model.opt_state)
    params_prior = vmap(model.get_params)(model.prior_opt_state)
    opt_params = (params, params_prior)
    samples = model.posterior(opt_params, x)
    samples = denormalize(samples, mu_y, sigma_y)
    
    samples = samples.reshape((samples.shape[0],samples.shape[1],16,N_y,N_y))
    intens = np.exp(-((xx - 0.5) ** 2 + (yy - 0.5) ** 2) / (0.95) ** 2) * samples
    ivec = np.sum(intens, axis = (-1, -2))
    smax = logsumexp(ivec, -1)
    smin = -logsumexp(-ivec, -1)
    v = np.exp( (smax - smin) / (smax + smin) )
    return -v[:,:,None]

# Fit GMM if needed for weighted acquisition functions
weights_fn = lambda x: np.ones(x.shape[0],)

if options['criterion']=='TS':
    args = (key_TS,)
else:
    kappa = options['kappa']
    args = (kappa,)
acq_model = MCAcquisition(predict,
                          bounds,
                          *args, 
                          acq_fn = options['criterion'],
                          output_weights=weights_fn)

# Optimize acquisition with L-BFGS to inquire new point(s)
new_X = acq_model.next_best_point(q = q1, num_restarts = num_restarts_acq, seed_id = 85*prev + (N-15))
new_X = new_X.reshape(q1,dim)
X = np.concatenate([X, new_X], axis = 0) # augment the vectorial input dataset during the BO process
np.save(case+'/X_loc_'+str(prev)+'.npy',X) # save the constructed vectorial input dataset by RPN-BO
