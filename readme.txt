- All files and directories mentioned in this file are contained under the folder "Code and results".

- The files "rpn_bo_acquisitions.py" , "rpn_bo_architectures.py" , "rpn_bo_dataloaders.py" , "rpn_bo_models.py" , "rpn_bo_models" , "rpn_bo_optimizers.py" and "rpn_bo_utilities.py" contain the functions needed to implement the RPN-BO method. These functions are imported and used within the scripts implementing the RPN-BO method for a given problem as detailed below for the examples considered.

- The folder "results" contain the results for the different problems described in the manuscript and listed below.

- The file "create_BO_cv_plots.py" uses the saved results in the folder "results" and creates and saves plots in the folder "figures". These are the plots included in the proposed manuscript.

- The file "HOGP_results.npz" contains the results for the numerical problems using the published HOGP method of Maddox at el., 2021 that was compared to the proposed RPN-BO method.

- The code was tested using the jax version 0.3.13, the jaxlib version 0.3.10, the numpy version 1.20.1 and the scipy version 1.7.2.

- To run the Brusselator PDE example, the py-pde package is needed (https://pypi.org/project/py-pde/). The code was tested using the py-pde version 0.12.7.

- To run the Optical Interferometer example, the gym_interf package is needed (https://github.com/dmitrySorokin/gym_interf , https://bitbucket.org/dmsoroki/gym_interf/src/master/ , https://github.com/dmitrySorokin/interferobotProject).

- The scripts and results folder for the different examples considered are detailed as follows:

###################################################
########## Environmental Model Function ###########
###################################################

- The file "environmental_model_function_MLP.py" contains the implementation of the RPN-BO method for the environmental model function example using MLP. For reproducibility of the optimization results, "environmental_model_function_MLP.py" calls 10 random initial datasets that are saved in folder "results/environmental_model_function_MLP" where "X_i.npy" and "y_i.npy" for i=0,...,9 are the corresponding vectorial inputs and outputs respectively.

- The optimization results generated by running the script "environmental_model_function_MLP.py" are saved in the folder "results/environmental_model_function_MLP" as numpy arrays "opt_i.npy", i=0,...,9, containing the best objective function value for the constructed dataset by RPN-BO for the 10 random independent runs.

- The obtained results with MLP are saved in the folders "results/environmental_model_function_MLP_EI", "results/environmental_model_function_MLP_LCB" and "results/environmental_model_function_MLP_TS" for the EI, LCB and TS acquisition functions respectively.

- The file "environmental_model_function_DON.py" contains the implementation of the RPN-BO method for the environmental model function example using DON. For reproducibility of the optimization results, "environmental_model_function_DON.py" calls 10 random initial datasets that are saved in folder "results/environmental_model_function_DON" where "X_i.npy" and "y_i.npy" for i=0,...,9 are the corresponding vectorial inputs and outputs respectively.

- The optimization results generated by running the script "environmental_model_function_DON.py" are saved in the folder "results/environmental_model_function_DON" as numpy arrays "opt_i.npy", i=0,...,9, containing the best objective function value for the constructed dataset by RPN-BO for the 10 random independent runs.

- The obtained results with DON are saved in the folders "results/environmental_model_function_DON_EI", "results/environmental_model_function_DON_LCB" and "results/environmental_model_function_DON_TS" for the EI, LCB and TS acquisition functions respectively.

###################################################
################# Brusselator PDE #################
###################################################

- The file "brusselator_pde_MLP.py" contains the implementation of the RPN-BO method for the Brusselator PDE control example using MLP. For reproducibility of the optimization results, "brusselator_pde_MLP.py" calls 10 random initial datasets that are saved in folder "results/brusselator_pde_MLP" where "X_i.npy" and "y_i.npy" for i=0,...,9 are the corresponding vectorial inputs and outputs respectively.

- The optimization results generated by running the script "brusselator_pde_MLP.py" are saved in the folder "results/brusselator_pde_MLP" as numpy arrays "opt_i.npy", i=0,...,9, containing the best objective function value for the constructed dataset by RPN-BO for the 10 random independent runs.

- The obtained results with MLP are saved in the folders "results/brusselator_pde_MLP_EI", and "results/brusselator_pde_MLP_LCB" for the EI and LCB acquisition functions respectively. The obtained results with MLP for 2 acquired points at each optimization step are saved in the folders "results/brusselator_pde_MLP_EI_q_2", and "results/brusselator_pde_MLP_LCB_2" for the EI and LCB acquisition functions respectively.

- The file "brusselator_pde_DON.py" contains the implementation of the RPN-BO method for the Brusselator PDE control example using DON. For reproducibility of the optimization results, "brusselator_pde_DON.py" calls 10 random initial datasets that are saved in folder "results/brusselator_pde_DON" where "X_i.npy" and "y_i.npy" for i=0,...,9 are the corresponding vectorial inputs and outputs respectively.

- The optimization results generated by running the script "brusselator_pde_DON.py" are saved in the folder "results/brusselator_pde_DON" as numpy arrays "opt_i.npy", i=0,...,9, containing the best objective function value for the constructed dataset by RPN-BO for the 10 random independent runs.

- The obtained results with DON are saved in the folders "results/brusselator_pde_DON_LCB", "results/brusselator_pde_DON_LW_LCB" and "results/brusselator_pde_DON_TS" for the LCB, LW-LCB and TS acquisition functions respectively.

###################################################
############# Optical Interferometer ##############
###################################################

- To avoid memory accumulation throughout the Bayesian Optimization process which requires an ensemble neural net to be trained for each optimization step, the RPN-BO implementation for the optical interferometer alignment example is divided into three steps as follows:
1. Run the script "optical_interferometer_MLP_step_0.py" in order to create copies of the initial training datasets available in folder saved in folder "results/optical_interferometer_MLP". Based on X_i.npy" and "y_i.npy" for i=0,...,4 containing the vectorial inputs and outputs for the initial training dataset for 5 random independent run, copies of these numpy array will be created and saved in the folder "results/optical_interferometer_MLP" under the names "X_loc_i.npy" and "y_loc_i.npy" for i=0,...,4, which will be augmented throughout the BO optimization. "optical_interferometer_MLP_step_0.py" will also compute and save the best objective values for each of the initial training datasets under the names "opt_i.npy" for i=0,...4.
2. Run the following command from the terminal:
for i in {1..85}; do python optical_interferometer_MLP_step_1.py ; do python optical_interferometer_MLP_step_2.py ; done
where 85 corresponds to the number of BO optimization steps and can be adjusted as desired. In the considered case, the initial training dataset size is 15 and 85 more points are acquired for a final constructed dataset by BO of size 100.
** The script "optical_interferometer_MLP_step_1.py" fits an MLP-based RPN model for the datasets  "X_loc_i.npy" and "y_loc_i.npy" contained within "results/optical_interferometer_MLP", where i corresponds to the variable "prev" in the script "optical_interferometer_MLP_step_1.py". 
** The script "optical_interferometer_MLP_step_1.py" also acquires additional point(s) and saves the augmented dataset for the vectorial input within the file "X_loc_i.npy" in the folder "results/optical_interferometer_MLP".
** The script "optical_interferometer_MLP_step_2.py" computes the corresponding vectorial output(s) to the newly acquired inputs points computed in the previous step by the script "optical_interferometer_MLP_step_1.py". The variable "prev" has to be the same for these two scripts in order to consider the same dataset.
** The script "optical_interferometer_MLP_step_2.py" also saves the augmented dataset for the vectorial output within the file "y_loc_i.npy" in the folder "results/optical_interferometer_MLP". 
** The script "optical_interferometer_MLP_step_2.py" also computes the objective value(s) of the newly acquired point(s) and saves the augmented array of best objective values under the names "opt_i.npy" in the folder "results/optical_interferometer_MLP".

- After running the commands 1 and 2 detailed above for each of the independent initial datasets, the optimization results are saved in the folder "results/optical_interferometer_MLP" as numpy arrays "opt_i.npy", i=0,...,4, containing the best objective function value for the constructed dataset by RPN-BO for the 5 random independent runs.

- The obtained results with MLP are saved in the folders "results/optical_interferometer_MLP_EI", "results/optical_interferometer_MLP_LCB" and "results/optical_interferometer_MLP_TS" for the EI, LCB and TS acquisition functions respectively.

- For the DON the same procedure can be applied as follows:
1. Run the script "optical_interferometer_DON_step_0.py" in order to create copies of the initial training datasets available in folder saved in folder "results/optical_interferometer_DON".
2. Run the following command from the terminal:
for i in {1..85}; do python optical_interferometer_DON_step_1.py ; do python optical_interferometer_DON_step_2.py ; done
where 85 corresponds to the number of BO optimization steps and can be adjusted as desired.

- After running the commands 1 and 2 detailed above for each of the independent initial datasets, the optimization results are saved in the folder "results/optical_interferometer_DON" as numpy arrays "opt_i.npy", i=0,...,4, containing the best objective function value for the constructed dataset by RPN-BO for the 5 random independent runs.

- The obtained results with DON are saved in the folders "results/optical_interferometer_DON_EI" for the EI acquisition function.

- Note that this strategy of considering scripts running a single optimization steps and saving the augmented datasets constructed by BO after each optimization step, along with executing them with the for command in the terminal can be applied to any example for which a memory limitation may be faced.

- For the interested reader and if enough computational ressources are available in terms of memory capacity, a single script containing the whole implementation of the RPN-BO method for the optical interferometer alignment example using MLP can be found in the file "optical_interferometer_MLP_all_steps.py".

###################################################
######### Compressor’s Blades Shape Design ########
###################################################

- This example's simulation requires Raytheon Technologies’ in-house CFD solver UTCFD which is not publicly available and we could not share the source code.
- We share the accessible format of the MLP-Based RPN-BO multi-fidelity BO results, along with single- and multi-fidelity Gaussian Processed-based BO results in the folder "results/compressor_blades_shape_MLP".