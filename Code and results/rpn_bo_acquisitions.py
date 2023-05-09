from jax import numpy as np
from jax import jit, vjp, random
from jax.scipy.special import expit as sigmoid
import numpy as onp
from functools import partial
from pyDOE import lhs
from tqdm import trange

from rpn_bo_optimizers import minimize_lbfgs

class MCAcquisition:
    def __init__(self, posterior, bounds, *args, 
                acq_fn = 'LCB', output_weights=lambda x: np.ones(x.shape[0])):  
        self.posterior = posterior
        self.bounds = bounds            # domain bounds
        self.args = args                # arguments required by different acquisition functions
        self.acq_fn = acq_fn            # a string indicating the chosen acquisition function
        self.weights = output_weights   # a callable function returning the likelihood weighted weights

    def evaluate(self, x):
        # Inputs are (q x d), use vmap to vectorize across a batch
        # samples[:,:,0]  corresponds to the objective function
        # samples[:,:,1:] corresponds to the constraints
        # samples[:,:,i] are (ensemble_size x q)
        q = x.shape[0]
        # Common acquisition functions
        if self.acq_fn == 'EI':
            best = self.args[0]
            samples = self.posterior(x)[:,:,0]
            reparam = np.maximum(best-samples, 0)
            EI = np.mean(np.max(reparam, axis=-1))
            return -EI
        elif self.acq_fn == 'LCB':
            kappa = self.args[0]
            samples = self.posterior(x)[:,:,0]
            mu = np.mean(samples, axis=0, keepdims=True)
            weights = self.weights(x).reshape(1,q)
            reparam = mu - np.sqrt(0.5*np.pi*kappa) * weights * np.abs(samples - mu)
            LCB = np.mean(np.min(reparam, axis=-1))
            return LCB
        elif self.acq_fn == 'TS':
            rng_key = self.args[0]
            samples = self.posterior(x)[:,:,0]
            idx = random.randint(rng_key, (1,), minval=0, maxval=samples.shape[0])
            reparam = samples[idx,:].reshape(1,q)
            TS = np.mean(np.min(reparam, axis=-1))
            return TS
        elif self.acq_fn == 'US':
            samples = self.posterior(x)[:,:,0]
            mu = np.mean(samples, axis=0, keepdims=True)            
            weights = self.weights(x).reshape(1,q)
            reparam = np.sqrt(0.5*np.pi) * weights * np.abs(samples - mu)
            US = np.mean(np.max(reparam, axis=-1))
            return -US
        elif self.acq_fn == 'CLSF':
            kappa = self.args[0]
            samples = self.posterior(x)[:,:,0]
            mu = np.mean(samples, axis=0, keepdims=True)
            weights = self.weights(x).reshape(1,q)
            reparam = np.abs(np.sqrt(0.5*np.pi) / (np.abs(mu)**(1.0/kappa) + 1e-8) * weights * np.abs(samples - mu))
            CLSF = np.mean(np.max(reparam, axis=-1))
            return -np.log(CLSF)

        # Constrained acquisition functions
        elif self.acq_fn == 'EIC':
            best = self.args[0]
            samples = self.posterior(x)
            # Objective
            objective = samples[:,:,0]
            reparam = np.maximum(best-objective, 0)
            EI = np.mean(np.max(reparam, axis=-1))
            # Constraints
            constraints = samples[:,:,1:]
            indicator = sigmoid(constraints/1e-6) # a smooth indicator function
            feasible = np.prod(np.mean(np.max(indicator, axis=1), axis=0))
            return -EI*feasible
        elif self.acq_fn == 'LCBC':
            kappa = self.args[0]
            threshold = self.args[1]
            samples = self.posterior(x)
            # Objective
            objective = samples[:,:,0]
            mu = np.mean(objective, axis=0, keepdims=True)
            weights = self.weights(x).reshape(1,q)
            reparam = mu - threshold - np.sqrt(0.5*np.pi*kappa) * weights * np.abs(objective - mu)
            LCB = np.mean(np.min(reparam, axis=-1))
            # Constraints
            constraints = samples[:,:,1:] # (ensemble_size x q)
            indicator = sigmoid(constraints/1e-6) # a smooth indicator function
            feasible = np.prod(np.mean(np.max(indicator, axis=1), axis=0))
            return LCB*feasible
        
        # That's all for now..
        else:
            raise NotImplementedError

    @partial(jit, static_argnums=(0,))
    def acq_value_and_grad(self, inputs):
        primals, f_vjp = vjp(self.evaluate, inputs)
        grads = f_vjp(np.ones_like(primals))[0]
        return primals, grads

    # optimization is performed in the normalized input space
    def next_best_point(self, q = 1, num_restarts = 10, seed_id=0, maxfun=15000):
        lb, ub = self.bounds   
        dim = lb.shape[0]
        # Define objective that returns float64 NumPy arrays
        def objective(x):
            x = x.reshape(q, dim)
            value, grads = self.acq_value_and_grad(x)
            out = (onp.array(value, dtype=onp.float64), 
                   onp.array(grads.flatten(), dtype=onp.float64))
            return out
        # Optimize with random restarts
        loc, acq = [], []
        onp.random.seed(seed_id)
        init = lb + (ub-lb)*lhs(dim, q*num_restarts)
        x0 = init.reshape(num_restarts, q, dim)
        dom_bounds = tuple(map(tuple, np.tile(np.vstack((lb, ub)).T,(q,1))))
        for i in trange(num_restarts):
            pos, val = minimize_lbfgs(objective, x0[i,:,:].flatten(), bnds = dom_bounds, maxfun=maxfun)
            loc.append(pos)
            acq.append(val)
        loc = np.vstack(loc)
        acq = np.vstack(acq)
        idx_best = np.argmin(acq)
        x_new = loc[idx_best,:]
        return x_new
