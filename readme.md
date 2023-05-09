## RPN-BO: Scalable Bayesian optimization with high-dimensional outputs using randomized prior networks

Code and data accompanying the manuscript titled "Scalable Bayesian optimization with high-dimensional outputs using randomized prior networks", authored by Mohamed Aziz Bhouri, Michael Joly, Robert Yu, Soumalya Sarkar and Paris Perdikaris.

## Abstract

Several fundamental problems in science and engineering consist of global optimization tasks involving unknown high-dimensional (black-box) functions that map a set of controllable variables to the outcomes of an expensive experiment. Bayesian Optimization (BO) techniques are known to be effective in tackling global optimization problems using a relatively small number objective function evaluations, but their performance suffers when dealing with high-dimensional outputs. To overcome the major challenge of dimensionality, here we propose a deep learning framework for BO and sequential decision making based on bootstrapped ensembles of neural architectures with randomized priors. Using appropriate architecture choices, we show that the proposed framework can approximate functional relationships between design variables and quantities of interest, even in cases where the latter take values in high-dimensional vector spaces or even infinite-dimensional function spaces. In the context of BO, we augmented the proposed probabilistic surrogates with re-parameterized Monte Carlo approximations of multiple-point (parallel) acquisition functions, as well as methodological extensions for accommodating black-box constraints and multi-fidelity information sources. We test the proposed framework against state-of-the-art methods for BO and demonstrate superior performance across several challenging tasks with high-dimensional outputs, including a constrained optimization task involving shape optimization of rotor blades in turbo-machinery.

## Citation

    @article{Bhouri2023RPNBO,
    title = {Scalable Bayesian optimization with high-dimensional outputs using randomized prior networks},
    author = {Bhouri, Mohamed Aziz and Joly, Michael and Yu, Robert and Sarkar, Soumalya and Perdikaris, Paris },
    journal = {arXiv preprint arXiv:2302.07260},
    doi = {https://doi.org/10.48550/arXiv.2302.07260},
    year = {2023},
    }
