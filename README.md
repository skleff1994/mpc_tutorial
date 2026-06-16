# MPC Tutorial

This repository provides a set of Python scripts and Jupyter notebooks solving toy optimal control problems (simple pendulum, cartpole, etc.). The goal is to familiarize students with the core concepts and methods from Optimal Control theory (continuous and discrete time).

In particular, the following methods are implemented:
* **Dynamic Programming** (Value Iteration)
* **HJB PDE** numerical resolution
* **Pontryagin Maximum Principle** (a.k.a. indirect optimal control)
* **Direct Optimal Control** (using SQP or DDP, based on Crocoddyl API)

---

## 🚀 Interactive Cloud Execution

You can run the notebooks directly in your web browser without installing anything locally:

### 1. Google Colab
Open the notebooks directly in Google Colab using the badges below. 

> [!IMPORTANT]
> Google Colab uses **`condacolab`** to install compiled robotics dependencies. 
> 1. Run the **first cell** (Conda Setup) in the notebook.
> 2. The kernel will **automatically restart** (you will see a crash/restart notification — this is expected!).
> 3. Run the **second cell** to install packages and clone the repository.

* **Intro:** [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/skleff1994/mpc_tutorial/blob/master/notebooks/intro.ipynb)
* **Part 1 (Optimal Control):** [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/skleff1994/mpc_tutorial/blob/master/notebooks/part1.ipynb)
* **Part 2 (Constraints & SQP):** [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/skleff1994/mpc_tutorial/blob/master/notebooks/part2.ipynb)
* **Part 3 (MPC & DDP):** [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/skleff1994/mpc_tutorial/blob/master/notebooks/part3.ipynb)

### 2. Binder
Click the badge below to start a full JupyterLab environment in Binder with all dependencies pre-installed:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/skleff1994/mpc_tutorial/master)

---

## 💻 Local Installation

We support two ways to install dependencies locally: **Pixi** (recommended, modern and fast) and **Conda**.

### Option A: Pixi (Recommended)
[Pixi](https://pixi.sh) is a modern package creator and manager. It installs all dependencies (including Python and system libraries) locally inside a `.pixi/` folder without touching your global system environment.

1. **Install Pixi** (if you don't have it):
   ```bash
   curl -fsSL https://pixi.sh/install.sh | bash
   ```
   *(Restart your terminal after installation)*

2. **Run a script directly**:
   Pixi will automatically install everything needed on the first run:
   ```bash
   pixi run pendulum-ocp
   ```
   Other tasks: `pendulum-bellman`, `cartpole-ocp`, `kuka-ocp`, `kuka-mpc`.

3. **Start Jupyter Lab**:
   ```bash
   pixi run lab
   ```

---

### Option B: Conda
If you prefer traditional Conda/Mamba:

1. **Create the environment**:
   ```bash
   conda env create -f environment.yml
   ```

2. **Activate the environment**:
   ```bash
   conda activate mpc_tutorial
   ```

3. **Install the package and mim_robots**:
   ```bash
   pip install -e . --no-deps
   pip install git+https://github.com/machines-in-motion/mim_robots.git --no-deps
   ```

4. **Run a script**:
   ```bash
   python pendulum/pendulum_ocp.py
   ```

---

## 🛠️ Dependencies

The main dependencies installed automatically by the setup tools are:
* [Crocoddyl](https://github.com/loco-3d/crocoddyl) (Contact-Consistent Robot Co-dynamics)
* [Pinocchio](https://github.com/stack-of-tasks/pinocchio) (Rigid Body Dynamics)
* [mim_solvers](https://github.com/machines-in-motion/mim_solvers) (Fast solvers for optimal control)
* [mim_robots](https://github.com/machines-in-motion/mim_robots) (Robot descriptions and wrappers)
* `pybullet`, `osqp`, `numpy`, `scipy`, `matplotlib`, `jupyterlab`

