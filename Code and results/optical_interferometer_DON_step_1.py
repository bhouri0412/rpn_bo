import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'

from jax import vmap, random, jit
from jax import numpy as np
from jax.scipy.special import logsumexp

from rpn_bo_models import ParallelDeepOnet
from rpn_bo_dataloaders import DataGenerator_batch
from rpn_bo_acquisitions import MCAcquisition
from rpn_bo_utilities import uniform_prior

# vectorial input space dimension and its search space
dim = 4
lb = -np.ones((dim,))
ub = np.ones((dim,))
p_x = uniform_prior(lb, ub)

# vectorial output space dimension
output_dim = (64, 64, 16)
soln_dim = output_dim[2]
P1 = output_dim[0]
P2 = output_dim[1]
xx, yy = np.meshgrid( np.arange(P1) / P1, np.arange(P2) / P2 )

# initial training space
case = 'results/optical_interferometer_DON'
prev = 0
X = np.load(case+'/X_loc_'+str(prev)+'.npy')
y = np.load(case+'/y_loc_'+str(prev)+'.npy')
opt = np.load(case+'/opt_'+str(prev)+'.npy')
N = X.shape[0]

#### RPN-BO hyperparameters ####
num_restarts_acq = 500
q1 = 1
N_ensemble = 16
fraction = 1
branch_layers = [dim, 32, 32]
trunk_layers =  [2, 32, 32]
nIter_RPN = 5000
acq_fct = 'EI'
batch_size = P1 * P2
batch_size_all = int(fraction * P1 * P2 * N) 

#### DeepONet functional evaluation points ####
arr_s = np.linspace(0, 1, P1)
arr_t = np.linspace(0, 1, P2)
s_grid, t_grid = np.meshgrid(arr_s, arr_t)
y_grid = np.concatenate([s_grid[:, :, None], t_grid[:, :, None]], axis=-1).reshape((-1, 2))
mu_grid = y_grid.mean(0)
sigma_grid = y_grid.std(0)
y_grid = (y_grid - mu_grid) / sigma_grid

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
mu_X = np.zeros(X.shape[1],)
sigma_X = np.ones(X.shape[1],)
mu_y = np.zeros((y.shape[1],y.shape[2],y.shape[3]))
sigma_y = np.max(np.abs(y)) * np.ones((y.shape[1],y.shape[2],y.shape[3]))
u0_train = (X - mu_X) / sigma_X
usol_train = (y - mu_y) / sigma_y
dataset = DataGenerator_batch(usol_train, u0_train, arr_s, arr_t, P1=P1, P2=P2, batch_size=batch_size, batch_size_all=batch_size_all, N_ensemble=N_ensemble, rng_key=train_key, y=y_grid)

# Initialize model
train_key = random.split(train_key, 2)[0]
model = ParallelDeepOnet(branch_layers, trunk_layers, N_ensemble, soln_dim)

# Train model
model.train(dataset, nIter=nIter_RPN)

@jit
def predict(x):
    x = (x - mu_X) / sigma_X
    u_test_sample = vmap(lambda x: np.tile(x, (P1 * P2, 1)))(x)
    samples = model.predict_s(u_test_sample.reshape((-1, dim)), np.tile(y_grid, (x.shape[0], 1)))
    samples = samples.reshape((-1, P1, P2, samples.shape[-1]))
    samples = vmap(lambda s: s * sigma_y + mu_y)(samples)
    samples = samples.reshape((N_ensemble, x.shape[0], P1, P2, samples.shape[-1]))
    samples = np.transpose(samples, (0, 1, 4, 2, 3))
    
    intens = np.exp(-((xx - 0.5) ** 2 + (yy - 0.5) ** 2) / (0.95) ** 2) * samples
    ivec = np.sum(intens, axis = (-1, -2))
    smax = logsumexp(ivec, -1)
    smin = -logsumexp(-ivec, -1)
    v = np.exp( (smax - smin) / (smax + smin) )
    return -v[:, :, None]

kappa = 2
weights_fn = lambda x: np.ones(x.shape[0])
if acq_fct == 'EI':
    args = (opt[-1], )
elif acq_fct == 'TS':
    key_TS = random.split(key_TS, 2)[0]
    args = (key_TS, ) 
elif acq_fct == 'LCB':
    weights_fn = lambda x: np.ones(x.shape[0],)
    args = (kappa,)
    
acq_model = MCAcquisition(predict, 
                          (lb, ub), 
                          *args, 
                          acq_fn=acq_fct, 
                          output_weights=weights_fn)

# Optimize acquisition with L-BFGS to inquire new point(s)
new_X = acq_model.next_best_point(q=q1, num_restarts=num_restarts_acq, seed_id=85*prev + (N-15))
X = np.concatenate([X, new_X[None, :]]) # augment the vectorial input dataset during the BO process
np.save(case+'/X_loc_'+str(prev)+'.npy',X) # save the constructed vectorial input dataset by RPN-BO
    
