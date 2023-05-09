from jax import vmap, random, jit
from jax import numpy as np
from functools import partial
from torch.utils import data

class BootstrapLoader(data.Dataset):
    def __init__(self, X, y, batch_size=128, ensemble_size=32, fraction=0.5, is_Gauss=1, LF_pred=None, rng_key=random.PRNGKey(1234)):
        'Initialization'
        self.N = X.shape[0]
        self.batch_size = batch_size
        self.ensemble_size = ensemble_size
        self.bootstrap_size = int(self.N*fraction)
        self.is_Gauss = is_Gauss
        self.key = rng_key
        # Create the bootstrapped partitions
        keys = random.split(rng_key, ensemble_size)
        if LF_pred is None:
            self.X, self.y = vmap(self.__bootstrap, (None,None,0))(X, y, keys)
        else:
            self.X, self.y = vmap(self.__bootstrapMF, (None,None,0,0))(X, y, LF_pred, keys)
        # Each bootstrapped data-set has its own normalization constants
        self.norm_const = vmap(self.normalization_constants, in_axes=(0,0))(self.X, self.y)
        
    @partial(jit, static_argnums=(0,))
    def normalization_constants(self, X, y):
        if self.is_Gauss == 1:
            mu_X, sigma_X = X.mean(0), X.std(0)
            mu_y, sigma_y = y.mean(0), y.std(0)
        else:
            mu_X, sigma_X = np.zeros(X.shape[1],), np.ones(X.shape[1],)
            mu_y = np.zeros(y.shape[1],)
            sigma_y = np.max(np.abs(y)) * np.ones(y.shape[1],)
        return (mu_X, sigma_X), (mu_y, sigma_y)
    
    @partial(jit, static_argnums=(0,))
    def __bootstrap(self, X, y, key):
        idx = random.choice(key, self.N, (self.bootstrap_size,), replace=False)
        inputs = X[idx,:]
        targets = y[idx,:]
        return inputs, targets
        
    @partial(jit, static_argnums=(0,))
    def __bootstrapMF(self, X, y, yLH, key):
        idx = random.choice(key, self.N, (self.bootstrap_size,), replace=False)
        inputs = np.concatenate([X[idx,:], yLH[idx,:]], axis=1)
        targets = y[idx,:]
        return inputs, targets

    @partial(jit, static_argnums=(0,))
    def __data_generation(self, key, X, y, norm_const):
        'Generates data containing batch_size samples'
        (mu_X, sigma_X), (mu_y, sigma_y) = norm_const
        idx = random.choice(key, self.N, (self.batch_size,), replace=False)
        X = X[idx,:]
        y = y[idx,:]
        X = (X - mu_X)/sigma_X
        y = (y - mu_y)/sigma_y
        return X, y

    def __getitem__(self, index):
        'Generate one batch of data'
        self.key, subkey = random.split(self.key)
        keys = random.split(self.key, self.ensemble_size)
        inputs, targets = vmap(self.__data_generation, (0,0,0,0))(keys, 
                                                                  self.X, 
                                                                  self.y, 
                                                                  self.norm_const)
        return inputs, targets
    
class DataGenerator_batch(data.Dataset):
    def __init__(self, usol, u0_train, s1, t1, P1 = 100, P2 = 100,
                 batch_size=64, batch_size_all=512, N_ensemble = 10, rng_key=random.PRNGKey(1234), y=None):
        'Initialization'
        self.usol = usol
        self.u0_train = u0_train
        self.N_train_realizations = usol.shape[0]
        self.P1 = P1
        self.P2 = P2
        self.dim = usol.shape[-1]
        u_samples_reshape = usol.reshape(self.N_train_realizations, P1*P2, self.dim)  # realizations x (mxp) x dim

        self.norms = vmap(np.linalg.norm, (0, None, None))(u_samples_reshape, np.inf, 0) # realizations x dim

        T, X = np.meshgrid(t1, s1)
        if y == None:
            self.y = np.hstack([T.flatten()[:,None], X.flatten()[:,None]])
        else:
            self.y = y
        self.batch_size = batch_size
        self.batch_size_all = batch_size_all
        self.N_ensemble = N_ensemble
        self.key = rng_key
        
    def __getitem__(self, index):
        'Generate one batch of data'
        self.key, subkey = random.split(self.key)
        v_subkey  = random.split(subkey, self.N_train_realizations)
        u_temp, y_temp, s_temp, w_temp = self.__get_realizations(v_subkey)
        self.key, subkey = random.split(self.key)
        v_subkey = random.split(subkey, self.N_ensemble)
        inputs, outputs = vmap(self.__data_generation, (0, None, None, None, None))(v_subkey, u_temp, y_temp, s_temp, w_temp)
        return inputs, outputs

    @partial(jit, static_argnums=(0,))
    def __data_generation(self, key, u_temp, y_temp, s_temp, w_temp):
        'Generates data containing batch_size samples'            
        idx = random.choice(key, self.N_train_realizations * self.batch_size, (self.batch_size_all,), replace=False)
        u = u_temp[idx,:]
        y = y_temp[idx,:]
        s = s_temp[idx,:]
        w = w_temp[idx,:]
        # Construct batch
        inputs = (u, y)
        outputs = (s, w)
        return inputs, outputs
       
    @partial(jit, static_argnums=(0,))
    def __get_realizations(self, key):
        idx_train = np.arange(self.N_train_realizations)
                        
        u_temp, y_temp, s_temp, w_temp = vmap(self.__generate_one_realization_data, (0, 0, None, None, None))(key, idx_train, self.usol, self.u0_train, self.norms)
          
        u_temp = np.float32(u_temp.reshape(self.N_train_realizations * self.batch_size,-1))
        y_temp = np.float32(y_temp.reshape(self.N_train_realizations * self.batch_size,-1))
        s_temp = np.float32(s_temp.reshape(self.N_train_realizations * self.batch_size,-1))
        w_temp = np.float32(w_temp.reshape(self.N_train_realizations * self.batch_size,-1))
               
        return u_temp, y_temp, s_temp, w_temp

    def __generate_one_realization_data(self, key, idx, usol, u0_train, norms):
            
        u = usol[idx]  
        u0 = u0_train[idx]
        ww = norms[idx]
          
        s = np.swapaxes(u, 0, 1)
        s = s.reshape(self.P1*self.P2, self.dim)
        u = np.tile(u0, (self.batch_size, 1))
        w = np.tile(ww, (self.batch_size, 1)) # for dim > 1, otherwise, w = np.tile(ww, (self.batch_size))
           
        idx_keep = random.choice(key, s.shape[0], (self.batch_size,), replace=False)

        return u, self.y[idx_keep,:], s[idx_keep], w
    