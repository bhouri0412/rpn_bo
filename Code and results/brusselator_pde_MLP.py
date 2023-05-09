import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
#
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
lb = np.array([0.1, 0.1, 0.01, 0.01])
ub = np.array([5.0, 5.0, 5.0, 5.0])
p_x = uniform_prior(lb, ub)
bounds = (lb, ub)

# vectorial output space dimension
N_y = 64
dim_y = 2*N_y**2

# function mapping the vectorial input x to the vectorial output consisting of the solution to the 2D Brusselator PDE evaluated at N_yxN_y grid points
def f(x):
    from pde import PDE, FieldCollection, ScalarField, UnitGrid
    
    a = x[0]
    b = x[1]
    d0 = x[2]
    d1 = x[3]

    eq = PDE(
        {
            "u": f"{d0} * laplace(u) + {a} - ({b} + 1) * u + u**2 * v",
            "v": f"{d1} * laplace(v) + {b} * u - u**2 * v",
        }
    )

    # initialize state
    grid = UnitGrid([N_y, N_y])
    u = ScalarField(grid, a, label="Field $u$")
    v = b / a + 0.1 * ScalarField.random_normal(grid, label="Field $v$", seed=10)
    state = FieldCollection([u, v])

    sol = eq.solve(state, t_range=20, dt=1e-3)
    
    sol_tensor = []
    sol_tensor.append(sol[0].data)
    sol_tensor.append(sol[1].data)
    sol_tensor = onp.array(sol_tensor)
    
    ss = sol_tensor[onp.isnan(sol_tensor)]
    sol_tensor[onp.isnan(sol_tensor)] = 1e5 * onp.random.randn(*ss.shape)
     
    return sol_tensor.flatten()
    
#### General simulation params ####
N = 5
prev = 0 # previous independent random runs
nTrSet = 10-prev # total independent random runs to perform

#### RPN-BO hyperparameters ####
num_restarts_acq = 500
nIter = 20 
q1 = 2
nIter_q1 = nIter//q1
ensemble_size = 128
batch_size = N
fraction = 0.8
layers = [dim, 64, 64, dim_y]
nIter_RPN = 10000
options = {'criterion': 'EI', # LCB EI
           'kappa': 2.0,
           'weights': None} # exact gmm None

train_key = random.PRNGKey(0)

case = 'results/brusselator_pde_MLP'

# prediction function mapping vectorial output to scalar obective value
def output(y):
    y = y.reshape((2,N_y,N_y))
    weighting = onp.ones((2,N_y,N_y))/10
    weighting[:, [0, 1, -2, -1], :] = 1.0
    weighting[:, :, [0, 1, -2, -1]] = 1.0
    weighted_samples = weighting * y
    return np.var(weighted_samples)

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
    yo_loc = vmap(output)(y_loc)
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
                        
            samples = samples.reshape((samples.shape[0],samples.shape[1],2,N_y,N_y))
            weighting = onp.ones((2,N_y,N_y))/10
            weighting[:, [0, 1, -2, -1], :] = 1.0
            weighting[:, :, [0, 1, -2, -1]] = 1.0
            weighted_samples = weighting * samples
            
            return np.var(weighted_samples, axis=(-3,-2,-1))[:,:,None]
        
        # Fit GMM if needed for weighted acquisition functions
        weights_fn = lambda x: np.ones(x.shape[0],)
        
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

        # Obtain the new data
        new_y = []
        for i in range(new_X.shape[0]):
            new_y.append(f(new_X[i,:]))
        new_y = onp.array(new_y)
        
        # Augment training data
        X_loc = np.concatenate([X_loc, new_X], axis = 0) # augment the vectorial input dataset during the BO process
        y_loc = np.concatenate([y_loc, new_y], axis = 0) # augment the vectorial output dataset during the BO process
        
        yo_loc = vmap(output)(y_loc)
        opt.append( np.min(yo_loc) )  # augment the objective values of the constructed dataset during the BO process
        
        batch_size_loc += q1
        
        print('Run %s, Nx %s, Ny %s, init %s, best %s' % (str(j+prev), X_loc.shape[0], y_loc.shape[0], opt[0], opt[-1]))
        
    np.save(case+'/opt_'+str(j+prev)+'.npy',onp.array(opt)) # save the constructed objective tensor by RPN-BO
        