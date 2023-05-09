import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'

from jax import vmap, random, jit
from jax import numpy as np
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
lb = np.array([7.0, 0.02, 0.01, 30.01])
ub = np.array([13.0, 0.12, 3.0, 30.295])
true_x = np.array([10.0, 0.07, 1.505, 30.1525])
p_x = uniform_prior(lb, ub)
bounds = (lb, ub)

# vectorial output space dimension
dim_y = 12

# pollutant concentration function
def c(s,t,M,D,L,tau):
    c1 = M/np.sqrt(4*np.pi*D*t)*np.exp(-s**2/4/D/t)
    c2 = M/np.sqrt(4*np.pi*D*(t-tau))*np.exp(-(s-L)**2/4/D/(t-tau))   
    return np.where(t>tau, c1+c2, c1)
s1 = np.array([0.0, 1.0, 2.5])
t1 = np.array([15.0, 30.0, 45.0, 60.0])
ST = np.meshgrid(s1, t1)
ST = np.array(ST).T.reshape(-1,2)

# function mapping the vectorial input x to the vectorial output consisting of the concentration evaluation at 3x4 grid points
def f(x):                    
    res = []
    for i in range(ST.shape[0]):
        res.append( c(ST[i,0],ST[i,1],x[0],x[1],x[2],x[3]) )
    return np.array(res)

#### General simulation params ####
N = 5 
prev = 0 # previous independent random runs
nTrSet = 10-prev # total independent random runs to perform

#### RPN-BO hyperparameters ####
num_restarts_acq = 500
nIter = 30
q1 = 1  
nIter_q1 = nIter//q1
ensemble_size = 128
batch_size = N
fraction = 0.8
layers = [dim, 64, 64, 64, 64, dim_y]
nIter_RPN = 5000
options = {'criterion': 'LCB', # LCB EI TS 
           'kappa': 2.0,
           'weights': None} # exact gmm None

train_key = random.PRNGKey(0)
key_TS = random.PRNGKey(123)

case = 'results/environmental_model_function_MLP'
true_y = f(true_x)

for j in range(nTrSet):
    
    print('Train Set:',j+1)
    
    # Initial training data
    X = np.load(case+'/X_'+str(j+prev)+'.npy')
    y = np.load(case+'/y_'+str(j+prev)+'.npy')
    
    X_loc = X
    y_loc = y
    batch_size_loc = batch_size
    
    # list to contain BO results
    opt = []
    yo_loc = np.sum((y_loc-true_y)**2, axis = 1)
    opt.append( np.min(yo_loc) )
    
    print('Run %s, Nx %s, Ny %s, init %s, best %s' % (str(j+prev), X_loc.shape[0], y_loc.shape[0], opt[0], opt[-1]))
    
    for it in range(nIter_q1):
        
        # Create data set
        train_key = random.split(train_key, 2)[0]
        dataset = BootstrapLoader(X_loc, y_loc, batch_size_loc, ensemble_size, fraction, 1, rng_key=train_key)
        (mu_X, sigma_X), (mu_y, sigma_y) = dataset.norm_const
        
        # Initialize model
        train_key = random.split(train_key, 2)[0]
        model = EnsembleRegression(layers, ensemble_size, train_key)
        
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
            
            samples = np.sum((samples-true_y)**2, axis = 2)[:,:,None]
            return samples
            
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
        new_y = vmap(f)(new_X)
        
        # Augment training data
        X_loc = np.concatenate([X_loc, new_X], axis = 0) # augment the vectorial input dataset during the BO process
        y_loc = np.concatenate([y_loc, new_y], axis = 0) # augment the vectorial output dataset during the BO process
        
        yo_loc = np.sum((y_loc-true_y)**2, axis = 1)
        opt.append( np.min(yo_loc) ) # augment the objective values of the constructed dataset during the BO process
        
        batch_size_loc += q1
        
        print('Run %s, Nx %s, Ny %s, init %s, best %s' % (str(j+prev), X_loc.shape[0], y_loc.shape[0], opt[0], opt[-1]))
        
    np.save(case+'/opt_'+str(j+prev)+'.npy',np.array(opt)) # save the constructed objective tensor by RPN-BO
    