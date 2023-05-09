from jax import numpy as np
from jax import grad, vmap, random, jit
from jax.example_libraries import optimizers
from jax.nn import relu, gelu
from functools import partial
from tqdm import trange
import itertools

from rpn_bo_architectures import MLP

class EnsembleRegression:
    def __init__(self, layers, ensemble_size, rng_key = random.PRNGKey(0), activation=np.tanh):  
        # Network initialization and evaluation functions
        self.init, self.apply = MLP(layers, activation)
        self.init_prior, self.apply_prior = MLP(layers, activation)
        
        # Random keys
        k1, k2, k3 = random.split(rng_key, 3)
        keys_1 = random.split(k1, ensemble_size)
        keys_2 = random.split(k2, ensemble_size)
        keys_3 = random.split(k2, ensemble_size)
        
        # Initialize
        params = vmap(self.init)(keys_1)
        params_prior = vmap(self.init_prior)(keys_2)
                
        # Use optimizers to set optimizer initialization and update functions
        lr = optimizers.exponential_decay(1e-3, decay_steps=1000, decay_rate=0.999)
        self.opt_init, \
        self.opt_update, \
        self.get_params = optimizers.adam(lr)

        self.opt_state = vmap(self.opt_init)(params)
        self.prior_opt_state = vmap(self.opt_init)(params_prior)
        self.key_opt_state = vmap(self.opt_init)(keys_3)

        # Logger
        self.itercount = itertools.count()
        self.loss_log = []

    # Define the forward pass
    def net_forward(self, params, params_prior, inputs):
        Y_pred = self.apply(params, inputs) + self.apply_prior(params_prior, inputs) 
        return Y_pred

    def loss(self, params, params_prior, batch):
        inputs, targets = batch
        # Compute forward pass
        outputs = vmap(self.net_forward, (None, None, 0))(params, params_prior, inputs)
        # Compute loss
        loss = np.mean((targets - outputs)**2)
        return loss

    # Define the update step
    def step(self, i, opt_state, prior_opt_state, key_opt_state, batch):
        params = self.get_params(opt_state)
        params_prior = self.get_params(prior_opt_state)
        g = grad(self.loss)(params, params_prior, batch)
        return self.opt_update(i, g, opt_state)
    
    def monitor_loss(self, opt_state, prior_opt_state, batch):
        params = self.get_params(opt_state)
        params_prior = self.get_params(prior_opt_state)
        loss_value = self.loss(params, params_prior, batch)
        return loss_value

    # Optimize parameters in a loop
    def train(self, dataset, nIter = 1000):
        data = iter(dataset)
        pbar = trange(nIter)
        # Define vectorized SGD step across the entire ensemble
        v_step = jit(vmap(self.step, in_axes = (None, 0, 0, 0, 0)))
        v_monitor_loss = jit(vmap(self.monitor_loss, in_axes = (0, 0, 0)))

        # Main training loop
        for it in pbar:
            batch = next(data)
            self.opt_state = v_step(it, self.opt_state, self.prior_opt_state, self.key_opt_state, batch)
            # Logger
            if it % 100 == 0:
                loss_value = v_monitor_loss(self.opt_state, self.prior_opt_state, batch)
                self.loss_log.append(loss_value)
                pbar.set_postfix({'Max loss': loss_value.max()})
           
    # Evaluates predictions at test points  
    @partial(jit, static_argnums=(0,))
    def posterior(self, params, inputs):
        params, params_prior = params
        samples = vmap(self.net_forward, (0, 0, 0))(params, params_prior, inputs)
        return samples  

class ParallelDeepOnet:
    def __init__(self, branch_layers, trunk_layers, N_ensemble, dim):  

        self.dim = dim  
        # Network initialization and evaluation functions
        self.branch_init, self.branch_apply = MLP(branch_layers, activation=relu) # jelu
        self.branch_init_prior, self.branch_apply_prior = MLP(branch_layers, activation=relu)
        self.trunk_init, self.trunk_apply = MLP(trunk_layers, activation=relu)
        self.trunk_init_prior, self.trunk_apply_prior = MLP(trunk_layers, activation=relu)
        
        # Initialize
        v_branch_params = vmap(self.branch_init)(random.split(random.PRNGKey(1234), N_ensemble))
        v_branch_params_prior = vmap(self.branch_init_prior)(random.split(random.PRNGKey(123), N_ensemble))
        v_trunk_params = vmap(self.trunk_init)(random.split(random.PRNGKey(4321), N_ensemble))
        v_trunk_params_prior = vmap(self.trunk_init_prior)(random.split(random.PRNGKey(321), N_ensemble))

        # If you want to initialize the weight W with 0.1 for all elements
        W = 0.1*np.ones((N_ensemble, branch_layers[-1], self.dim))
           
        # If you want to initialize the weight W with Xavier initialization (This is helpful to check if the method work)
        # Because if the value of different output dimension are same, using the above W will result in same predictions.
        # glorot_stddev = 1. / np.sqrt((branch_layers[-1] + self.dim) / 2.)
        # W = glorot_stddev*random.normal(random.PRNGKey(123), (N_ensemble, branch_layers[-1], self.dim))

        v_params = (v_branch_params, v_trunk_params, W)
        v_params_prior = (v_branch_params_prior, v_trunk_params_prior)

        # Use optimizers to set optimizer initialization and update functions
        lr = optimizers.exponential_decay(1e-3,decay_steps=1000,decay_rate=0.999)
#        lr = 1e-4
        self.opt_init, \
        self.opt_update, \
        self.get_params = optimizers.adam(lr)
        self.v_opt_state = vmap(self.opt_init)(v_params)
        self.v_prior_opt_state = vmap(self.opt_init)(v_params_prior)

        # Logger
        self.itercount = itertools.count()
        self.loss_log = []
    
    # Define the operator net
    def operator_net(self, params, params_prior, u, y):
        branch_params, trunk_params, W = params
        branch_params_prior, trunk_params_prior = params_prior
        B = self.branch_apply(branch_params, u) + self.branch_apply_prior(branch_params_prior, u)
        T = self.trunk_apply(trunk_params, y) + self.trunk_apply_prior(trunk_params_prior, y)
        #outputs = np.sum(B * T)
        outputs = np.dot(B * T, W)
        return outputs
               
    @partial(jit, static_argnums=(0,))
    def loss(self, params, params_prior, batch):
        # Fetch data
        # inputs: (u, y), shape = (N, m), (N,1)
        # outputs: s, shape = (N,1)
        inputs, outputs = batch
        u, y = inputs
        s, w = outputs
        # Compute forward pass
        pred = vmap(self.operator_net, (None, None, 0, 0))(params, params_prior, u, y)
        # Compute loss
        loss = np.mean(1./w**2 * (s - pred)**2)
        return loss

    # Define a compiled update step
    # @partial(jit, static_argnums=(0,))
    def step(self, i, opt_state, prior_opt_state, batch):
        params = self.get_params(opt_state)
        params_prior = self.get_params(prior_opt_state)
        g = grad(self.loss, argnums = 0)(params, params_prior, batch)
        return self.opt_update(i, g, opt_state)

    # Optimize parameters in a loop
    def train(self, dataset, nIter = 10000):
        data = iter(dataset)
        pbar = trange(nIter)
        
        # Define v_step that vectorize the step operation
        self.v_step = jit(vmap(self.step, in_axes = [None, 0, 0, 0]))
           
        # Main training loop
        for it in pbar:
            batch = next(data)
            self.v_opt_state = self.v_step(it, self.v_opt_state, self.v_prior_opt_state, batch)
            # Logger
            if it % 200 == 0:
                params = vmap(self.get_params)(self.v_opt_state)
                params_prior = vmap(self.get_params)(self.v_prior_opt_state)
                branch_params_prior, trunk_params_prior = params_prior
                loss_value = vmap(self.loss, (0, 0, 0))(params, params_prior, batch)
                self.loss_log.append(loss_value)
                pbar.set_postfix({'Max loss': loss_value.max()})
            
    def operator_net_pred_single(self, params, params_prior, U_star, Y_star):
        s_pred_single = vmap(self.operator_net, (None, None, 0, 0))(params, params_prior, U_star, Y_star)
        return s_pred_single
       
    # Evaluates predictions at test points  
    @partial(jit, static_argnums=(0,))
    def predict_s(self, U_star, Y_star):
        params = vmap(self.get_params)(self.v_opt_state)
        params_prior = vmap(self.get_params)(self.v_prior_opt_state)
        s_pred = vmap(self.operator_net_pred_single, (0, 0, None,None))(params, params_prior, U_star, Y_star)
        return s_pred
    
    
    
    
    



