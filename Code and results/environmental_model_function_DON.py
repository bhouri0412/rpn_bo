import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'

from jax import vmap, random, jit
from jax import numpy as np
import numpy as onp

from rpn_bo_utilities import uniform_prior
from rpn_bo_models import ParallelDeepOnet
from rpn_bo_dataloaders import DataGenerator_batch
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

# pollutant concentration function
def c(s,t,M,D,L,tau):
    c1 = M/np.sqrt(4*np.pi*D*t)*np.exp(-s**2/4/D/t)
    c2 = M/np.sqrt(4*np.pi*D*(t-tau))*np.exp(-(s-L)**2/4/D/(t-tau))   
    return np.where(t>tau, c1+c2, c1)
s1 = np.array([0.0, 1.0, 2.5])
t1 = np.array([15.0, 30.0, 45.0, 60.0])
ST = np.meshgrid(s1, t1)
STo = np.array(ST).T
    
# function mapping the vectorial input x to the vectorial output consisting of the concentration evaluation at 3x4 grid points
def f(x):
    res = []
    for i in range(STo.shape[0]):
        resl = []
        for j in range(STo.shape[1]):
            resl.append( c(STo[i,j,0],STo[i,j,1],x[0],x[1],x[2],x[3]) )
        res.append(np.array(resl)) 
    return np.array(res)

#### General simulation params ####
N = 5
prev = 0 # previous independent random runs
nTrSet = 10-prev # total independent random runs to perform

#### DeepONet functional evaluation points ####
m = 4
P1 = 4
P2 = 3
Ms = 2.5
Mt = 60.0
soln_dim = 1
s1 = np.array([0.0, 1.0, 2.5])/Ms
t1 = np.array([15.0, 30.0, 45.0, 60.0])/Mt
Tm, Xm = np.meshgrid(t1, s1)
y_test_sample = np.hstack([Tm.flatten()[:,None], Xm.flatten()[:,None]])

#### RPN-BO hyperparameters ####
num_restarts_acq = 500
nIter = 30
q1 = 1 
nIter_q1 = nIter//q1
N_ensemble = 128
fraction = 0.8
branch_layers = [m, 64, 64]
trunk_layers =  [2, 64, 64]
nIter_RPN = 5000
options = {'criterion': 'TS', # LCB EI TS
           'kappa': 2.0,
           'weights': None} # exact gmm None

train_key = random.PRNGKey(0)
key_TS = random.PRNGKey(123)

case = 'results/environmental_model_function_DON'
true_y = f(true_x)
true_y = np.expand_dims(true_y, axis = 2)

for j in range(nTrSet):
    
    print('Train Set:',j+1)
    
    # Initial training data
    X = np.load(case+'/X_'+str(j+prev)+'.npy')
    y = np.load(case+'/y_'+str(j+prev)+'.npy')
    X = np.array(X)
    y = np.array(y)
    y = y.reshape(y.shape[0],P2,P1)
    y = np.expand_dims(y, axis = 3)
    
    X_loc = X
    y_loc = y
    batch_size_loc = 12 # max value is P1*P2
    
    # list to contain BO results
    opt = []
    yo_loc = np.sum((y_loc-true_y)**2, axis = (1,2))
    opt.append( np.min(yo_loc) )
    
    for it in range(nIter_q1):
        
        sigma_X = X_loc.std(0)
        mu_X = X_loc.mean(0)
        sigma_y = y_loc.std(0)
        mu_y = y_loc.mean(0)
        
        # Create data set
        usol_train = (y_loc-mu_y)/sigma_y
        u0_train = (X_loc-mu_X)/sigma_X
        batch_size_all_loc = int(fraction*12*X_loc.shape[0])
        dataset = DataGenerator_batch(usol_train, u0_train, s1, t1, P1, P2, batch_size_loc, batch_size_all_loc, N_ensemble)

        # Initialize model
        model = ParallelDeepOnet(branch_layers, trunk_layers, N_ensemble, soln_dim)
        
        # Train model
        model.train(dataset, nIter=nIter_RPN)
        
        @jit
        def predict(x):
            x = (x-mu_X)/sigma_X
            u_test_sample = np.tile(x, (P1*P2, 1))
            samples = model.predict_s(u_test_sample, y_test_sample) # N_ensemble x P1*P2 x soln_dim
            samples = samples.reshape((samples.shape[0],P1,P2,samples.shape[-1])) # N_ensemble x P1 x P2 x soln_dim
            samples = np.transpose(samples, (0, 2, 1, 3)) # N_ensemble x P2 x P1 x soln_dim
            samples = sigma_y*samples+mu_y
            samples = np.sum((samples-true_y)**2, axis = (1,2))[:,:,None]
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
        new_y = np.expand_dims(new_y, axis = 3)
        
        # Augment training data
        X_loc = np.concatenate([X_loc, new_X], axis = 0) # augment the vectorial input dataset during the BO process
        y_loc = np.concatenate([y_loc, new_y], axis = 0) # augment the vectorial output dataset during the BO process
        
        yo_loc = np.sum((y_loc-true_y)**2, axis = 1)
        opt.append( np.min(yo_loc) ) # augment the objective values of the constructed dataset during the BO process
        
        print('Run %s, Nx %s, Ny %s, init %s, best %s' % (str(j+prev), X_loc.shape[0], y_loc.shape[0], opt[0], opt[-1]))
        
    np.save(case+'/opt_'+str(j+prev)+'.npy',np.array(opt)) # save the constructed objective tensor by RPN-BO
    
    