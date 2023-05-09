import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'

from jax import vmap, random, jit
from jax import numpy as np
from pyDOE import lhs
import numpy as onp

from rpn_bo_utilities import uniform_prior, output_weights
from rpn_bo_models import ParallelDeepOnet
from rpn_bo_dataloaders import DataGenerator_batch
from rpn_bo_acquisitions import MCAcquisition

# vectorial input space dimension and its search space
dim = 4
lb = np.array([0.1, 0.1, 0.01, 0.01])
ub = np.array([5.0, 5.0, 5.0, 5.0])
p_x = uniform_prior(lb, ub)

# vectorial output space dimension and DeepONet functional evaluation points
N_y = 64
output_dim = (N_y, N_y, 2)
soln_dim = 2
P1 = output_dim[0]
P2 = output_dim[1]
arr_s = np.linspace(0, 1, P1)
arr_t = np.linspace(0, 1, P2)
s_grid, t_grid = np.meshgrid(arr_s, arr_t)
y_grid = np.concatenate([s_grid[:, :, None], t_grid[:, :, None]], axis=-1).reshape((-1, 2))
mu_grid = y_grid.mean(0)
sigma_grid = y_grid.std(0)
y_grid = (y_grid - mu_grid) / sigma_grid

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
    
    return np.transpose(np.array(sol_tensor),(1,2,0))

#### General simulation params ####
N = 5
prev = 0
nTrSet = 30-prev

#### RPN-BO hyperparameters ####
nIter = 30
N_ensemble = 16
batch_size = P1 * P2
batch_size_all = P1 * P2 * N
branch_layers = [dim, 64, 64]
trunk_layers =  [2, 64, 64]
nIter_RPN = 1000
acq_fct = 'LCB' # 'LCB', 'TS', 'LW_LCB'

case = 'results/brusselator_pde_DON'

# prediction function mapping vectorial output to scalar obective value
def output(new_y):
    weighting = onp.ones((2, 64, 64)) / 10
    weighting[:, [0, 1, -2, -1], :] = 1.0
    weighting[:, :, [0, 1, -2, -1]] = 1.0
    weighted = weighting * np.transpose(new_y, (2, 0, 1))
    return np.var(weighted, axis=(-3, -2, -1))

for j in range(nTrSet):
    
    # Initial training data
    onp.random.seed(j)
    X = lb + (ub - lb) * lhs(dim, N)
    y = np.array([f(x) for x in X])
    opt = []
    opt.append(np.min(np.array([output(yi) for yi in y])))

    keys, keys_ts, keys_trans, keys_noise, keys_loader = random.split(random.PRNGKey(j), nIter * N).reshape((N, nIter, -1))
    
    print('Run %s, Nx %s, Ny %s, init %s, best %s' % (str(j+prev), X.shape[0], y.shape[0], opt[0], opt[-1]))
    
    for it in range(nIter):
        
        # Create data set
        mu_X = X.mean(0)
        sigma_X = X.std(0)
        mu_y = y.mean(0)
        sigma_y = y.std(0)
        u0_train = (X - mu_X) / sigma_X
        usol_train = (y - mu_y) / sigma_y
        dataset = DataGenerator_batch(usol_train, u0_train, arr_s, arr_t, P1=P1, P2=P2, batch_size=batch_size, batch_size_all=batch_size_all, N_ensemble=N_ensemble, y=y_grid, rng_key=keys_loader[it])
        
        # Initialize model
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
    
            weighting = onp.ones((2, 64, 64)) / 10
            weighting[:, [0, 1, -2, -1], :] = 1.0
            weighting[:, :, [0, 1, -2, -1]] = 1.0
            weighted_samples = weighting * samples
            return np.var(weighted_samples, axis=(-3, -2, -1))[:, :, None]
        
        kappa = 2
        weights_fn = lambda x: np.ones(x.shape[0])
        if acq_fct == 'TS':
            args = (keys_ts[it], )
            num_restarts = 100
            acq_fn = 'TS'
        elif acq_fct == 'LCB':
            weights_fn = lambda x: np.ones(x.shape[0],)
            args = (kappa, )
            num_restarts = 100
            acq_fn = 'LCB'
        elif acq_fct == 'LW_LCB':
            predict_fn = lambda x: np.mean(predict(x), axis=0)
            num_samples = 100
            weights_fn = output_weights(predict_fn, uniform_prior(lb, ub).pdf, (lb, ub), method='gmm', num_samples=num_samples, num_comp=5)
            args = (kappa, )
            num_restarts = 100
            acq_fn = 'LCB'
    
        acq_model = MCAcquisition(predict, (lb, ub), *args, acq_fn=acq_fn, output_weights=weights_fn)
        
        # Optimize acquisition with L-BFGS to inquire new point(s)
        new_X = acq_model.next_best_point(q=1, num_restarts=num_restarts, seed_id=100 * j + it)
        
        # Obtain the new data
        new_y = f(new_X)
        
        # Augment training data
        X = np.concatenate([X, new_X[None, :]]) # augment the vectorial input dataset during the BO process
        y = np.concatenate([y, new_y[None, :, :, :]]) # augment the vectorial output dataset during the BO process
        
        opt.append(np.minimum(opt[-1],output(new_y))) # augment the objective values of the constructed dataset during the BO process
        
        print('Run %s, Nx %s, Ny %s, init %s, best %s' % (str(j+prev), X.shape[0], y.shape[0], opt[0], opt[-1]))
        
        del model, dataset
    
    np.save(case+'/opt_'+str(j+prev)+'.npy',onp.array(opt)) # save the constructed objective tensor by RPN-BO
    