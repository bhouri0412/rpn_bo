from jax import numpy as np
from jax.scipy.special import logsumexp

output_dim = (64, 64, 16) # 16 frames of 64 by 64 images
P1 = output_dim[0]
P2 = output_dim[1]
xx, yy = np.meshgrid( np.arange(P1) / P1, np.arange(P2) / P2 )

# prediction function mapping vectorial output to scalar obective value
def output(new_y):
    intens = np.exp(-((xx - 0.5) ** 2 + (yy - 0.5) ** 2) / (0.95) ** 2) * new_y
    ivec = np.sum(intens, axis = (-1, -2))
    smax = logsumexp(ivec, -1)
    smin = -logsumexp(-ivec, -1)
    return - (smax - smin) / (smax + smin)

case_l = ['results/optical_interferometer_DON']

# create new files for vectorial inputs and outputs and best objective values which will be augmented by newly acquired points during BO process
for case in case_l:
    
    for prev in range(5):
        X = np.load(case+'/X_'+str(prev)+'.npy') 
        y = np.load(case+'/y_'+str(prev)+'.npy')
        
        y = y.reshape((y.shape[0],output_dim[2],P1,P2))
        
        errs = [output(yi) for yi in y]
        
        np.save(case+'/opt_'+str(prev)+'.npy',np.array(np.min(np.array(errs)))[None])
        
        y = np.transpose(y, (0, 2, 3, 1))
        
        np.save(case+'/X_loc_'+str(prev)+'.npy',X)
        np.save(case+'/y_loc_'+str(prev)+'.npy',y)
    