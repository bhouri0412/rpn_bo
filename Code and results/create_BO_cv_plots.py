problem = 'comp_blades_shape' # choose from 'environment' 'brusselator' 'optical_interferometer' 'comp_blades_shape'

from matplotlib import pyplot as plt
plt.close('all')
plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams.update({'font.weight': 'bold',
                     'font.size': 28,
                     'lines.linewidth': 1.5,
                     'axes.labelsize': 36,
                     'axes.titlesize': 36,
                     'xtick.labelsize': 28,
                     'ytick.labelsize': 28,
                     'legend.fontsize': 36,
                     'axes.linewidth': 4,
                     "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
                      "text.usetex": True,                # use LaTeX to write all text
                      })
plt.rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']

import torch
import numpy as np

dat_file = np.load("./HOGP_results.npz", allow_pickle = True)
list(dat_file.keys())
dat = dat_file["obj"].reshape(-1)[0]
dat.keys()

from scipy import stats

################################################
############### environment ####################
################################################
if problem == 'environment':
    
    N = 5
    nIter = 30
    q1 = 1
    nTrSet = 10
    dispersion_scale = 0.2
    
    case = 'results/environmental_model_function_MLP_LCB'
    case2 = 'results/environmental_model_function_MLP_EI'
    case3 = 'results/environmental_model_function_MLP_TS'
    case4 = 'results/environmental_model_function_DON_LCB'
    case5 = 'results/environmental_model_function_DON_EI'
    case6 = 'results/environmental_model_function_DON_TS'
    
    opt_q1_RPN = []
    opt_q1_RPN2 = []
    opt_q1_RPN3 = []
    opt_q1_RPN4 = []
    opt_q1_RPN5 = []
    opt_q1_RPN6 = []

    for j in range(nTrSet):
        opt = np.load(case+'/opt_'+str(j)+'.npy')
        opt_q1_RPN.append(np.array(opt))   
        opt2 = np.load(case2+'/opt_'+str(j)+'.npy')
        opt_q1_RPN2.append(np.array(opt2)) 
        opt3 = np.load(case3+'/opt_'+str(j)+'.npy')
        opt_q1_RPN3.append(np.array(opt3)) 
        opt4 = np.load(case4+'/opt_'+str(j)+'.npy')
        opt_q1_RPN4.append(np.array(opt4)) 
        opt5 = np.load(case5+'/opt_'+str(j)+'.npy')
        opt_q1_RPN5.append(np.array(opt5)) 
        opt6 = np.load(case6+'/opt_'+str(j)+'.npy')
        opt_q1_RPN6.append(np.array(opt6)) 
        
    opt_q1_RPN = np.array(opt_q1_RPN)
    opt_q1_RPN2 = np.array(opt_q1_RPN2)
    opt_q1_RPN3 = np.array(opt_q1_RPN3)
    opt_q1_RPN4 = np.array(opt_q1_RPN4)
    opt_q1_RPN5 = np.array(opt_q1_RPN5)
    opt_q1_RPN6 = np.array(opt_q1_RPN6)
       
    m_q1_RPN, std_q1_RPN = np.median(opt_q1_RPN, axis = 0), stats.median_abs_deviation(opt_q1_RPN, axis = 0)
    m_q1_RPN2, std_q1_RPN2 = np.median(opt_q1_RPN2, axis = 0), stats.median_abs_deviation(opt_q1_RPN2, axis = 0)
    m_q1_RPN3, std_q1_RPN3 = np.median(opt_q1_RPN3, axis = 0), stats.median_abs_deviation(opt_q1_RPN3, axis = 0)
    m_q1_RPN4, std_q1_RPN4 = np.median(opt_q1_RPN4, axis = 0), stats.median_abs_deviation(opt_q1_RPN4, axis = 0)
    m_q1_RPN5, std_q1_RPN5 = np.median(opt_q1_RPN5, axis = 0), stats.median_abs_deviation(opt_q1_RPN5, axis = 0)
    m_q1_RPN6, std_q1_RPN6 = np.median(opt_q1_RPN6, axis = 0), stats.median_abs_deviation(opt_q1_RPN6, axis = 0)
    
    lower_q1_RPN = np.log10(np.clip(m_q1_RPN - dispersion_scale*std_q1_RPN, a_min=0., a_max = np.inf) + 1e-10)
    upper_q1_RPN = np.log10(m_q1_RPN + dispersion_scale*std_q1_RPN + 1e-10)
    lower_q1_RPN2 = np.log10(np.clip(m_q1_RPN2 - dispersion_scale*std_q1_RPN2, a_min=0., a_max = np.inf) + 1e-10)
    upper_q1_RPN2 = np.log10(m_q1_RPN2 + dispersion_scale*std_q1_RPN2 + 1e-10)
    lower_q1_RPN3 = np.log10(np.clip(m_q1_RPN3 - dispersion_scale*std_q1_RPN3, a_min=0., a_max = np.inf) + 1e-10)
    upper_q1_RPN3 = np.log10(m_q1_RPN3 + dispersion_scale*std_q1_RPN3 + 1e-10)
    lower_q1_RPN4 = np.log10(np.clip(m_q1_RPN4 - dispersion_scale*std_q1_RPN4, a_min=0., a_max = np.inf) + 1e-10)
    upper_q1_RPN4 = np.log10(m_q1_RPN4 + dispersion_scale*std_q1_RPN4 + 1e-10)
    lower_q1_RPN5 = np.log10(np.clip(m_q1_RPN5 - dispersion_scale*std_q1_RPN5, a_min=0., a_max = np.inf) + 1e-10)
    upper_q1_RPN5 = np.log10(m_q1_RPN5 + dispersion_scale*std_q1_RPN5 + 1e-10)
    lower_q1_RPN6 = np.log10(np.clip(m_q1_RPN6 - dispersion_scale*std_q1_RPN6, a_min=0., a_max = np.inf) + 1e-10)
    upper_q1_RPN6 = np.log10(m_q1_RPN6 + dispersion_scale*std_q1_RPN6 + 1e-10)
    
    fig = plt.figure(figsize=(21, 9))
    ax = plt.subplot(111)
    
    ax.plot(N+q1*np.arange(nIter+1), np.log10(m_q1_RPN)[:nIter+1], color='black', label = r'\textbf{RPN - MLP - LCB}')
    ax.fill_between(N+q1*np.arange(nIter+1), lower_q1_RPN[:nIter+1], upper_q1_RPN[:nIter+1], facecolor='black', alpha=0.3)
    ax.plot(N+q1*np.arange(nIter+1), np.log10(m_q1_RPN2)[:nIter+1], color='blue', label = r'\textbf{RPN - MLP - EI}')
    ax.fill_between(N+q1*np.arange(nIter+1), lower_q1_RPN2[:nIter+1], upper_q1_RPN2[:nIter+1], facecolor='blue', alpha=0.3)
    ax.plot(N+q1*np.arange(nIter+1), np.log10(m_q1_RPN3)[:nIter+1], color='limegreen', label = r'\textbf{RPN - MLP - TS}')
    ax.fill_between(N+q1*np.arange(nIter+1), lower_q1_RPN3[:nIter+1], upper_q1_RPN3[:nIter+1], facecolor='limegreen', alpha=0.3)
    
    ax.plot(N+q1*np.arange(nIter+1), np.log10(m_q1_RPN4), '-.', color='black', label = r'\textbf{RPN - DON - LCB}')
    ax.fill_between(N+q1*np.arange(nIter+1), lower_q1_RPN4, upper_q1_RPN4, facecolor='red', alpha=0.3)
    ax.plot(N+q1*np.arange(nIter+1), np.log10(m_q1_RPN5), '-.', color='blue', label = r'\textbf{RPN - DON - EI}')
    ax.fill_between(N+q1*np.arange(nIter+1), lower_q1_RPN5, upper_q1_RPN5, facecolor='blue', alpha=0.3)
    ax.plot(N+q1*np.arange(nIter+1), np.log10(m_q1_RPN6), '-.', color='limegreen', label = r'\textbf{RPN - DON - TS}')
    ax.fill_between(N+q1*np.arange(nIter+1), lower_q1_RPN6, upper_q1_RPN6, facecolor='limegreen', alpha=0.3)
    
    plt.xticks(np.arange(N, N+nIter+1, N))
    plt.xlim([5,35])
    ax.grid(color='Grey', linestyle='-', linewidth=0.5)
    
    sample_means = dat["env_means"]
    sample_stds = dat["env_stds"]
    keys = dat["env_keys"]
    key_dict = {"rnd": r'\textbf{Random}', "rnd_cf": r'\textbf{Random-CF}', "ei": r'\textbf{EI}', "ei_cf": r'\textbf{EI-CF}', \
                "ei_hogp_cf": r'\textbf{EI-HOGP-CF}', "ei_hogp_cf_smooth": r'\textbf{EI-HOGP-CF + GP}'}
    steps = torch.linspace(5, 35, 30)
    for i, key in enumerate(keys):
        ax.fill_between(steps, 
                         sample_means[i] - sample_stds[i] / 20**0.5, 
                         sample_means[i] + sample_stds[i] / 20**0.5, 
                         alpha = 0.1)
        ax.plot(steps, sample_means[i], '--', linewidth=3, label = key_dict[key])
    
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', frameon=False, fontsize = 36, bbox_to_anchor=(0.98, 0.5))
    
    plt.xlabel(r'\textbf{Function Evaluations}')
    plt.ylabel(r'\textbf{Log10(Regret)}')
    plt.savefig('figures/environmental_model_function.png',dpi=300,bbox_inches='tight')

################################################
################ brusselator ###################
################################################    
if problem == 'brusselator':
    
    N = 5
    nIter = 20 
    q1 = 1
    q2 = 2
    nIter_q1 = nIter//q1
    nIter_q2 = nIter//q2
    nIter_q1_DON = 30
    nTrSet = 10
    nTrSet_DON = 30
    dispersion_scale = 0.2
    
    case = 'results/brusselator_pde_MLP_LCB'
    case_EI = 'results/brusselator_pde_MLP_EI'
    case_q2 = 'results/brusselator_pde_MLP_EI_q_2'
    case_LCB_q2 = 'results/brusselator_pde_MLP_LCB_q_2'
    case2 = 'results/brusselator_pde_DON_TS'
    case3 = 'results/brusselator_pde_DON_LW_LCB'
    case4 = 'results/brusselator_pde_DON_LCB'
    
    opt_q1_RPN = []
    opt_q1_RPN_EI = []
    opt_q2_RPN = []
    opt_LCB_q2_RPN = []
    opt_q1_RPN2 = []
    opt_q1_RPN3 = []
    opt_q1_RPN4 = []
    
    for j in range(nTrSet):
        opt = np.load(case+'/opt_'+str(j)+'.npy')
        opt_q1_RPN.append(np.array(opt))    
        opt = np.load(case_q2+'/opt_'+str(j)+'.npy')
        opt_q2_RPN.append(np.array(opt))    
        opt = np.load(case_EI+'/opt_'+str(j)+'.npy')
        opt_q1_RPN_EI.append(np.array(opt)) 
        opt = np.load(case_LCB_q2+'/opt_'+str(j)+'.npy')
        opt_LCB_q2_RPN.append(np.array(opt))    
        
    opt_q1_RPN = np.array(opt_q1_RPN)
    opt_q2_RPN = np.array(opt_q2_RPN)
    opt_q1_RPN_EI = np.array(opt_q1_RPN_EI)
    opt_LCB_q2_RPN = np.array(opt_LCB_q2_RPN)
    
    for j in range(nTrSet):
        opt = np.load(case2+'/opt_'+str(j)+'.npy')
        opt_q1_RPN2.append(np.array(opt))    
        opt = np.load(case3+'/opt_'+str(j)+'.npy')
        opt_q1_RPN3.append(np.array(opt))    
        opt = np.load(case4+'/opt_'+str(j)+'.npy')
        opt_q1_RPN4.append(np.array(opt)) 
        
    m_q1_RPN, std_q1_RPN = np.median(opt_q1_RPN, axis = 0), stats.median_abs_deviation(opt_q1_RPN, axis = 0)
    m_q1_RPN2, std_q1_RPN2 = np.median(opt_q1_RPN2, axis = 0), stats.median_abs_deviation(opt_q1_RPN2, axis = 0)
    m_q1_RPN3, std_q1_RPN3 = np.median(opt_q1_RPN3, axis = 0), stats.median_abs_deviation(opt_q1_RPN3, axis = 0)
    m_q1_RPN4, std_q1_RPN4 = np.median(opt_q1_RPN4, axis = 0), stats.median_abs_deviation(opt_q1_RPN4, axis = 0)
    m_q2_RPN, std_q2_RPN = np.median(opt_q2_RPN, axis = 0), stats.median_abs_deviation(opt_q2_RPN, axis = 0)
    m_q1_RPN_EI, std_q1_RPN_EI = np.median(opt_q1_RPN_EI, axis = 0), stats.median_abs_deviation(opt_q1_RPN_EI, axis = 0)
    m_LCB_q2_RPN, std_LCB_q2_RPN = np.median(opt_LCB_q2_RPN, axis = 0), stats.median_abs_deviation(opt_LCB_q2_RPN, axis = 0)
    
    lower_q1_RPN = np.log10(np.clip(m_q1_RPN - dispersion_scale*std_q1_RPN, a_min=0., a_max = np.inf) + 1e-8)
    upper_q1_RPN = np.log10(m_q1_RPN + dispersion_scale*std_q1_RPN + 1e-8)
    lower_q1_RPN2 = np.log10(np.clip(m_q1_RPN2 - dispersion_scale*std_q1_RPN2, a_min=0., a_max = np.inf) + 1e-10)
    upper_q1_RPN2 = np.log10(m_q1_RPN2 + dispersion_scale*std_q1_RPN2 + 1e-10)
    lower_q1_RPN3 = np.log10(np.clip(m_q1_RPN3 - dispersion_scale*std_q1_RPN3, a_min=0., a_max = np.inf) + 1e-10)
    upper_q1_RPN3 = np.log10(m_q1_RPN3 + dispersion_scale*std_q1_RPN3 + 1e-10)
    lower_q1_RPN4 = np.log10(np.clip(m_q1_RPN4 - dispersion_scale*std_q1_RPN4, a_min=0., a_max = np.inf) + 1e-10)
    upper_q1_RPN4 = np.log10(m_q1_RPN4 + dispersion_scale*std_q1_RPN4 + 1e-10)
    lower_q2_RPN = np.log10(np.clip(m_q2_RPN - dispersion_scale*std_q2_RPN, a_min=0., a_max = np.inf) + 1e-8)
    upper_q2_RPN = np.log10(m_q2_RPN + dispersion_scale*std_q2_RPN + 1e-8)
    lower_q1_RPN_EI = np.log10(np.clip(m_q1_RPN_EI - dispersion_scale*std_q1_RPN_EI, a_min=0., a_max = np.inf) + 1e-8)
    upper_q1_RPN_EI = np.log10(m_q1_RPN_EI + dispersion_scale*std_q1_RPN_EI + 1e-8)
    lower_LCB_q2_RPN = np.log10(np.clip(m_LCB_q2_RPN - dispersion_scale*std_LCB_q2_RPN, a_min=0., a_max = np.inf) + 1e-8)
    upper_LCB_q2_RPN = np.log10(m_LCB_q2_RPN + dispersion_scale*std_LCB_q2_RPN + 1e-8)
    
    fig = plt.figure(figsize=(21, 9))
    ax = plt.subplot(111)
    
    ax.plot(N+q1*np.arange(nIter_q1+1), np.log10(m_q1_RPN), color='black', label = r'\textbf{RPN - MLP - LCB}')
    ax.fill_between(N+q1*np.arange(nIter_q1+1), lower_q1_RPN, upper_q1_RPN, facecolor='black', alpha=0.3)
    ax.plot(N+q2*np.arange(nIter_q2+1), np.log10(m_LCB_q2_RPN), color='slategrey', label = r'\textbf{RPN - MLP - LCB, q=2}')
    ax.fill_between(N+q2*np.arange(nIter_q2+1), lower_LCB_q2_RPN, upper_LCB_q2_RPN, facecolor='slategrey', alpha=0.3)
    
    ax.plot(N+q1*np.arange(nIter_q1+1), np.log10(m_q1_RPN_EI), color='blue', label = r'\textbf{RPN - MLP - EI}')
    ax.fill_between(N+q1*np.arange(nIter_q1+1), lower_q1_RPN_EI, upper_q1_RPN_EI, facecolor='blue', alpha=0.3)
    ax.plot(N+q2*np.arange(nIter_q2+1), np.log10(m_q2_RPN), color='lightskyblue', label = r'\textbf{RPN - MLP - EI, q=2}')
    ax.fill_between(N+q2*np.arange(nIter_q2+1), lower_q2_RPN, upper_q2_RPN, facecolor='lightskyblue', alpha=0.3)
    
    ax.plot(N+q1*np.arange(nIter_q1_DON+1), np.log10(m_q1_RPN4), '-.', color='black', label = r'\textbf{RPN - DON - LCB}')
    ax.fill_between(N+q1*np.arange(nIter_q1_DON+1), lower_q1_RPN4, upper_q1_RPN4, facecolor='black', alpha=0.3)
    ax.plot(N+q1*np.arange(nIter_q1_DON+1), np.log10(m_q1_RPN3), '-.', color='hotpink', label = r'\textbf{RPN - DON - LCB-LW}')
    ax.fill_between(N+q1*np.arange(nIter_q1_DON+1), lower_q1_RPN3, upper_q1_RPN3, facecolor='hotpink', alpha=0.3)
    ax.plot(N+q1*np.arange(nIter_q1_DON+1), np.log10(m_q1_RPN2), '-.', color='limegreen', label = r'\textbf{RPN - DON - TS}')
    ax.fill_between(N+q1*np.arange(nIter_q1_DON+1), lower_q1_RPN2, upper_q1_RPN2, facecolor='limegreen', alpha=0.3)
    
    plt.xticks(np.arange(N, N+nIter_q1_DON+1, N))
    plt.xlim([5,35])
    ax.grid(color='Grey', linestyle='-', linewidth=0.5)
    
    sample_means = dat["pde_means"]
    sample_stds = dat["pde_stds"]
    keys = dat["pde_keys"]
    key_dict = {"rnd": r'\textbf{Random}', "rnd_cf": r'\textbf{Random-CF}', "ei": r'\textbf{EI}', "ei_cf": r'\textbf{EI-CF}', \
                "ei_hogp_cf": r'\textbf{EI-HOGP-CF}', "ei_hogp_cf_smooth": r'\textbf{EI-HOGP-CF + GP}'}
    steps = torch.linspace(5, 35, 30)
    for i, key in enumerate(keys):
        ax.fill_between(steps, 
                         sample_means[i] - sample_stds[i] / 20**0.5, 
                         sample_means[i] + sample_stds[i] / 20**0.5, 
                         alpha = 0.1)
        ax.plot(steps, sample_means[i], '--', linewidth=3, label = key_dict[key])
    
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', frameon=False, fontsize = 36, bbox_to_anchor=(0.98, 0.5))
    
    plt.xlabel(r'\textbf{Function Evaluations}')
    plt.ylabel(r'\textbf{Log10(Variance)}')
    plt.savefig('figures/brusselator_pde.png', dpi=300,bbox_inches='tight')

################################################
############ optical_interferometer ############
################################################  
if problem == 'optical_interferometer':
    
    N = 15
    nIter = 85 
    q1 = 1
    nIter_q1 = nIter//q1
    nTrSet = 5
    dispersion_scale = 0.2
    
    case = 'results/optical_interferometer_MLP_LCB'
    case2 = 'results/optical_interferometer_MLP_EI'
    case3 = 'results/optical_interferometer_MLP_TS'
    case4 = 'results/optical_interferometer_DON_EI'
    
    opt_q1_RPN = []
    opt_q1_RPN2 = []
    opt_q1_RPN3 = []
    opt_q1_RPN4 = []
    for j in range(nTrSet):
        opt = -np.load(case+'/'+'opt_'+str(j)+'.npy')[:nIter_q1+1]
        opt2 = -np.load(case2+'/'+'opt_'+str(j)+'.npy')[:nIter_q1+1]
        opt3 = -np.load(case3+'/'+'opt_'+str(j)+'.npy')[:nIter_q1+1]
        opt4 = -np.load(case4+'/'+'opt_'+str(j)+'.npy')[:nIter_q1+1]
        opt_q1_RPN.append(np.array(opt))    
        opt_q1_RPN2.append(np.array(opt2))
        opt_q1_RPN3.append(np.array(opt3))
        opt_q1_RPN4.append(np.array(opt4))
        
    opt_q1_RPN = np.array(opt_q1_RPN)
    opt_q1_RPN2 = np.array(opt_q1_RPN2)
    opt_q1_RPN3 = np.array(opt_q1_RPN3)
    opt_q1_RPN4 = np.array(opt_q1_RPN4)
    m_q1_RPN, std_q1_RPN = np.median(opt_q1_RPN, axis = 0), stats.median_abs_deviation(opt_q1_RPN, axis = 0)
    m_q1_RPN2, std_q1_RPN2 = np.median(opt_q1_RPN2, axis = 0), stats.median_abs_deviation(opt_q1_RPN2, axis = 0)
    m_q1_RPN3, std_q1_RPN3 = np.median(opt_q1_RPN3, axis = 0), stats.median_abs_deviation(opt_q1_RPN3, axis = 0)
    m_q1_RPN4, std_q1_RPN4 = np.median(opt_q1_RPN4, axis = 0), stats.median_abs_deviation(opt_q1_RPN4, axis = 0)
    
    lower_q1_RPN = np.clip(m_q1_RPN - dispersion_scale*std_q1_RPN, a_min=0., a_max = np.inf) + 1e-8
    upper_q1_RPN = m_q1_RPN + dispersion_scale*std_q1_RPN + 1e-8
    lower_q1_RPN2 = np.clip(m_q1_RPN2 - dispersion_scale*std_q1_RPN2, a_min=0., a_max = np.inf) + 1e-8
    upper_q1_RPN2 = m_q1_RPN2 + dispersion_scale*std_q1_RPN2 + 1e-8
    lower_q1_RPN3 = np.clip(m_q1_RPN3 - dispersion_scale*std_q1_RPN3, a_min=0., a_max = np.inf) + 1e-8
    upper_q1_RPN3 = m_q1_RPN3 + dispersion_scale*std_q1_RPN3 + 1e-8
    lower_q1_RPN4 = np.clip(m_q1_RPN4 - dispersion_scale*std_q1_RPN4, a_min=0., a_max = np.inf) + 1e-8
    upper_q1_RPN4 = m_q1_RPN4 + dispersion_scale*std_q1_RPN4 + 1e-8
    
    fig = plt.figure(figsize=(21, 9))
    ax = plt.subplot(111)
    
    ax.plot(N+q1*np.arange(nIter_q1+1), m_q1_RPN, color='black', label = r'\textbf{RPN - MLP - LCB}')
    ax.fill_between(N+q1*np.arange(nIter_q1+1), lower_q1_RPN, upper_q1_RPN, facecolor='black', alpha=0.3)
    ax.plot(N+q1*np.arange(nIter_q1+1), m_q1_RPN2, color='blue', label = r'\textbf{RPN - MLP - EI}')
    ax.fill_between(N+q1*np.arange(nIter_q1+1), lower_q1_RPN2, upper_q1_RPN2, facecolor='blue', alpha=0.3)
    ax.plot(N+q1*np.arange(nIter_q1+1), m_q1_RPN3, color='limegreen', label = r'\textbf{RPN - MLP - TS}')
    ax.fill_between(N+q1*np.arange(nIter_q1+1), lower_q1_RPN3, upper_q1_RPN3, facecolor='limegreen', alpha=0.3)
    ax.plot(N+q1*np.arange(nIter_q1+1), m_q1_RPN4, '-.', color='blue', label = r'\textbf{RPN - DON - EI}')
    ax.fill_between(N+q1*np.arange(nIter_q1+1), lower_q1_RPN4, upper_q1_RPN4, facecolor='blue', alpha=0.3)
    ax.grid(color='Grey', linestyle='-', linewidth=0.5)
    plt.xlim([15,100])
    
    sample_means = dat["optics_means"]
    sample_stds = dat["optics_stds"]
    keys = dat["optics_keys"]
    key_dict = {"rnd": r'\textbf{Random}', "rnd_cf": r'\textbf{Random-CF}', "ei": r'\textbf{EI}', "ei_cf": r'\textbf{EI-CF}', \
                "ei_hogp_cf": r'\textbf{EI-HOGP-CF}', "ei_hogp_cf_smooth": r'\textbf{EI-HOGP-CF + GP}'}
    steps = torch.linspace(0, 100, 100)
    for i, key in enumerate(keys):
        plt.fill_between(steps, 
                         sample_means[i] - sample_stds[i] / 45**0.5, 
                         sample_means[i] + sample_stds[i] / 45**0.5, 
                         alpha = 0.1)
        plt.plot(steps, sample_means[i], '--', linewidth=3, label = key_dict[key])
    
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', frameon=False, fontsize = 36, bbox_to_anchor=(0.98, 0.5))
    
    plt.xlabel(r'\textbf{Function Evaluations}')
    plt.ylabel(r'\textbf{Visibility (V)}')
    plt.savefig('figures/optical_interferometer.png', dpi=300,bbox_inches='tight')

################################################
############### comp_blades_shape ###############
################################################
if problem == 'comp_blades_shape':
    dispersion_scale = 0.2
    
    x_MFGP = np.load('results/compressor_blades_shape_MLP/x_MFGP.npy')
    mean_MFGP = np.load('results/compressor_blades_shape_MLP/mean_MFGP.npy')
    std_MFGP = np.load('results/compressor_blades_shape_MLP/std_MFGP.npy')
    
    x_SFGP = np.load('results/compressor_blades_shape_MLP/x_SFGP.npy')
    mean_SFGP = np.load('results/compressor_blades_shape_MLP/mean_SFGP.npy')
    std_SFGP = np.load('results/compressor_blades_shape_MLP/std_SFGP.npy')
    
    x_MFRPN = np.load('results/compressor_blades_shape_MLP/x_MFRPN.npy')
    mean_MFRPN = np.load('results/compressor_blades_shape_MLP/mean_MFRPN.npy')
    std_MFRPN = np.load('results/compressor_blades_shape_MLP/std_MFRPN.npy')
    
    lower_SFGP = mean_SFGP - dispersion_scale*std_SFGP
    upper_SFGP = mean_SFGP + dispersion_scale*std_SFGP
    lower_MFGP = mean_MFGP - dispersion_scale*std_MFGP
    upper_MFGP = mean_MFGP + dispersion_scale*std_MFGP
    lower_MFRPN = mean_MFRPN - dispersion_scale*std_MFRPN
    upper_MFRPN = mean_MFRPN + dispersion_scale*std_MFRPN
    
    plt.figure(figsize = (16, 9), facecolor = "w")
    plt.plot(x_MFRPN,mean_MFRPN, linewidth=3, color='black', label = r'\textbf{MF - RPN - LCBC}')
    plt.fill_between(x_MFRPN, lower_MFRPN, upper_MFRPN, facecolor='black', alpha=0.3)
    plt.plot(x_SFGP,mean_SFGP, '--', linewidth=3, color='magenta', label = r'\textbf{SF - GP - LCBC}')
    plt.fill_between(x_SFGP, lower_SFGP, upper_SFGP, facecolor='magenta', alpha=0.3)
    plt.plot(x_MFGP,mean_MFGP, '--', linewidth=3, color='orange', label = r'\textbf{MF - GP - LCBC}')
    plt.fill_between(x_MFGP, lower_MFGP, upper_MFGP, facecolor='orange', alpha=0.3)
    plt.xlim([0,200])
    plt.grid(color='Grey', linestyle='-', linewidth=0.5)
    
    plt.legend(fontsize = 28)
    plt.xlabel(r'\textbf{Cost (unit of CFD HF evaluations) for aqcuired points}')
    plt.ylabel(r'\textbf{Optimization objective}')
    plt.show()
    plt.savefig('figures/comp_blades_shape.png',dpi=300,bbox_inches='tight')
    