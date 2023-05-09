from jax import numpy as np
from jax.scipy.special import logsumexp
from jax import vmap

N_y = 64 # each frame is an N_y by N_y image
xx, yy = np.meshgrid( np.arange(N_y) / N_y, np.arange(N_y) / N_y )

# prediction function mapping vectorial output to scalar obective value
def output(y):
    y = y.reshape((16,N_y,N_y)) # form the 16 frames, from the vectorial shaped tensors
    intens = np.exp(-((xx - 0.5) ** 2 + (yy - 0.5) ** 2) / (0.95) ** 2) * y
    ivec = np.sum(intens, axis = (-1, -2))
    smax = logsumexp(ivec, -1)
    smin = -logsumexp(-ivec, -1)
    return - (smax - smin) / (smax + smin)

case_l = ['results/optical_interferometer_MLP']

# create new files for vectorial inputs and outputs and best objective values which will be augmented by newly acquired points during BO process
for case in case_l:
    
    for prev in range(5):
        X = np.load(case+'/X_'+str(prev)+'.npy')
        y = np.load(case+'/y_'+str(prev)+'.npy')
        
        yo = vmap(output)(y)
        np.save(case+'/opt_'+str(prev)+'.npy',np.array(np.min(yo))[None])
            
        np.save(case+'/X_loc_'+str(prev)+'.npy',X)
        np.save(case+'/y_loc_'+str(prev)+'.npy',y)
    