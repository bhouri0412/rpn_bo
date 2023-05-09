import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'

from jax import vmap, random, jit
from jax import numpy as np
from jax.scipy.special import logsumexp
from jax.nn import relu
from gym_interf import InterfEnv
import numpy as onp

from rpn_bo_utilities import uniform_prior
from rpn_bo_models import EnsembleRegression
from rpn_bo_dataloaders import BootstrapLoader
from rpn_bo_acquisitions import MCAcquisition

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

# function mapping the vectorial input x to the vectorial output consisting of the 16 images
def f(x):
    gym = InterfEnv()
    gym.reset(actions=(1e-4, 1e-4, 1e-4, 1e-4))
    action = x[:4]
    state = gym.step(action)
    return state[0].flatten()

#### General simulation params ####
N = 15 
prev = 0 # previous independent random runs
nTrSet = 10-prev # total independent random runs to perform

#### RPN-BO hyperparameters ####
num_restarts_acq = 500 
nIter = 100 - N
q1 = 1  
nIter_q1 = nIter//q1
ensemble_size = 32
batch_size = N
fraction = 0.8
layers = [dim, 64, 64, 16*N_y**2]
nIter_RPN = 5000
options = {'criterion': 'LCB', # 'TS' 'LCB', 
           'kappa': 2.0,
           'weights': None, # exact gmm None
           }

train_key = random.PRNGKey(0)
key_TS = random.PRNGKey(123)

case = 'results/optical_interferometer_MLP'

# prediction function mapping vectorial output to scalar obective value
def output(y):
    y = y.reshape((16,N_y,N_y))
    intens = np.exp(-((xx - 0.5) ** 2 + (yy - 0.5) ** 2) / (0.95) ** 2) * y
    ivec = np.sum(intens, axis = (-1, -2))
    smax = logsumexp(ivec, -1)
    smin = -logsumexp(-ivec, -1)
    return - (smax - smin) / (smax + smin)

for j in range(nTrSet):
    print('Train Set:',j+1)
    
    # Initial training data
    X = np.load(case+'/X_'+str(j+prev)+'.npy')
    y = np.load(case+'/y_'+str(j+prev)+'.npy')
    
    X_loc = X
    y_loc = y
    batch_size_loc = batch_size
    
    # array to contain BO results
    yo_loc = vmap(output)(y)
    opt = np.array(np.min(yo_loc))[None]
    
    print('Run %s, Nx %s, Ny %s, init %s, best %s' % (str(j+prev), X_loc.shape[0], y_loc.shape[0], opt[0], opt[-1]))
    
    for it in range(nIter_q1):
        
        # Create data set
        train_key = random.split(train_key, 2)[0]
        dataset = BootstrapLoader(X_loc, y_loc, batch_size_loc, ensemble_size, fraction, 0, rng_key=train_key)
        (mu_X, sigma_X), (mu_y, sigma_y) = dataset.norm_const
        
        # Initialize model
        train_key = random.split(train_key, 2)[0]
        model = EnsembleRegression(layers, ensemble_size, train_key, relu)
        
        # Train model
        model.train(dataset, nIter=nIter_RPN)
        
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
            key_TS = random.split(key_TS, 2)[0]
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
        new_X = acq_model.next_best_point(q = q1, num_restarts = num_restarts_acq)
        new_X = new_X.reshape(q1,dim)
        
        # Obtain the new vectorial output
        new_y = []
        for i in range(new_X.shape[0]):
            new_y.append(f(new_X[i,:]))
        new_y = np.array(new_y)
        
        # Augment training data
        X_loc = np.concatenate([X_loc, new_X], axis = 0) # augment the vectorial input dataset during the BO process
        y_loc = np.concatenate([y_loc, new_y], axis = 0) # augment the vectorial output dataset during the BO process
        opt = np.concatenate( ( opt, np.minimum(opt[-1],output(new_y[0,:]))[None] ) , axis=0 ) # augment the objective values of the constructed dataset during the BO process
        
        # Save augmented datasets and obejctive values
        np.save(case+'/X_loc_'+str(j+prev)+'.npy', X_loc) # save the constructed vectorial input dataset by RPN-BO
        np.save(case+'/y_loc_'+str(j+prev)+'.npy', y_loc) # save the constructed vectorial output dataset by RPN-BO
        np.save(case+'/opt_'+str(j+prev)+'.npy',opt) # save the constructed objective tensor by RPN-BO
        
        batch_size_loc += q1
        
        print('Run %s, Nx %s, Ny %s, init %s, best %s' % (str(j+prev), X_loc.shape[0], y_loc.shape[0], opt[0], opt[-1]))
        