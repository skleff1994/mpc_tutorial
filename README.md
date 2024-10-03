# Description
This repository provides a set of python scripts solving on toy optimal control problems (simple pendulum, cartpole, etc.). The goal is to familiarize with the core concepts and methods from Optimal Control theory (continous and discrete time). 

In particular, the following methods are implemented 
- Dynamic Programming (value iteration)
- HJB PDE numerical resolution
- Pontryagin Maximum Principle (a.k.a indirect optimal control)
- Direct Optimal Control (using SQP or DDP, based on Crocoddyl API)

The repo is still under construction (do not look into `experimental/`). The scripts available in `pendulum/` and `cartpole/` should work, but feedback & contributions is welcome in case not.

# Dependencies
- [Crocoddyl](https://github.com/loco-3d/crocoddyl)
- [mim_solvers](https://github.com/machines-in-motion/mim_solvers/tree/main)
- matplotlib

# Installation
It is recommended to use conda. Install miniconda if you do not have it already by following the instructions from the anaconda webpage : https://docs.anaconda.com/miniconda/miniconda-install/

Then create a conda environment :
```
conda create -n mpc_tutorial
```

Activate the environment and install mim_solvers and matplotlib
```
conda activate mpc_tutorial
conda install -c conda-forge mim-solvers
conda install matplotlib
```

Then run the script of your choice, e.g. the pendulum SQP by running :
```
python pendulum/pendulum_sqp.py
```
