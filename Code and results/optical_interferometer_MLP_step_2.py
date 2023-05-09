from jax import numpy as np
from jax.scipy.special import logsumexp
from gym_interf import InterfEnv

# function mapping the vectorial input x to the vectorial output consisting of the 16 images
def f(x):
    gym = InterfEnv()
    gym.reset(actions=(1e-4, 1e-4, 1e-4, 1e-4))
    action = x[:4]
    state = gym.step(action)
    return state[0].flatten()

N_y = 64 # each frame is a N_y by N_y image
xx, yy = np.meshgrid( np.arange(N_y) / N_y, np.arange(N_y) / N_y )

# prediction function mapping vectorial output to scalar obective value
def output(y):
    y = y.reshape((16,N_y,N_y))
    intens = np.exp(-((xx - 0.5) ** 2 + (yy - 0.5) ** 2) / (0.95) ** 2) * y
    ivec = np.sum(intens, axis = (-1, -2))
    smax = logsumexp(ivec, -1)
    smin = -logsumexp(-ivec, -1)
    return - (smax - smin) / (smax + smin)

case = 'results/optical_interferometer_MLP'
prev = 0 # change from 0 to 4 to consdier different random and independent run
X = np.load(case+'/X_loc_'+str(prev)+'.npy') # load vectorial inputs for the constructed dataset so far during the BO process
y = np.load(case+'/y_loc_'+str(prev)+'.npy') # load vectorial outputs for the constructed dataset so far during the BO process

new_y = f(X[-1,:]) # compute the vectorial output (the 16 images) of the newly acquired point
y = np.concatenate([y, new_y[None,:]], axis = 0) # augment the vectorial output dataset during the BO process
np.save(case+'/y_loc_'+str(prev)+'.npy',y) # save the constructed vectorial output dataset by RPN-BO

opt = np.load(case+'/opt_'+str(prev)+'.npy') # load best objective for the constructed dataset so far during the BO process

opt = np.concatenate( ( opt, np.minimum(opt[-1],output(new_y))[None] ) , axis=0 ) # augment the objective values of the constructed dataset during the BO process
np.save(case+'/opt_'+str(prev)+'.npy',opt) # save the constructed objective tensor by RPN-BO

print('new_X: ', X[-1,:], 'new obj:', output(new_y), 'opt obj: ',np.min(np.array(opt))) # output the newly acquired point, its corresponding objective value, and the best objective value so far in the BO process
