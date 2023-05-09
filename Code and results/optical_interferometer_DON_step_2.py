from jax import numpy as np
from jax.scipy.special import logsumexp
from gym_interf import InterfEnv

output_dim = (64, 64, 16) # 16 frames of 64 by 64 images
soln_dim = output_dim[2]
P1 = output_dim[0]
P2 = output_dim[1]

# function mapping the vectorial input x to the vectorial output consisting of the 16 images
def f(x):
    gym = InterfEnv()
    gym.reset(actions=(1e-4, 1e-4, 1e-4, 1e-4))
    action = x[:4]
    state = gym.step(action)
    return state[0]

xx, yy = np.meshgrid( np.arange(P1) / P1, np.arange(P2) / P2 )
# prediction function mapping vectorial output to scalar obective value
def output(new_y):
    intens = np.exp(-((xx - 0.5) ** 2 + (yy - 0.5) ** 2) / (0.95) ** 2) * new_y
    ivec = np.sum(intens, axis = (-1, -2))
    smax = logsumexp(ivec, -1)
    smin = -logsumexp(-ivec, -1)
    return - (smax - smin) / (smax + smin)
    
case = 'results/optical_interferometer_DON'
prev = 0 # change from 0 to 4 to consdier different random and independent run
X = np.load(case+'/X_loc_'+str(prev)+'.npy') # load vectorial inputs for the constructed dataset so far during the BO process
y = np.load(case+'/y_loc_'+str(prev)+'.npy') # load vectorial outputs for the constructed dataset so far during the BO process

new_y = f(X[-1,:]) # compute the vectorial output (the 16 images) of the newly acquired point
y = np.concatenate([ y, np.transpose(new_y[None, :, :, :], (0, 2, 3, 1)) ]) # augment the vectorial output dataset during the BO process
np.save(case+'/y_loc_'+str(prev)+'.npy',y) # save the constructed vectorial output dataset by RPN-BO

opt = np.load(case+'/opt_'+str(prev)+'.npy') # load best objective for the constructed dataset so far during the BO process

opt = np.concatenate( ( opt, np.minimum(opt[-1],output(new_y))[None] ) , axis=0 )
np.save(case+'/opt_'+str(prev)+'.npy',opt) # save the constructed objective tensor by RPN-BO

print('new_X: ', X[-1,:], 'new obj:', output(new_y), 'min obj: ',np.min(np.array(opt)))
