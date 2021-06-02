# Neural Symplectic Integrator for Astrophysical *N*-body Simulations
Wisdom-Holman integrator augmented with physics informed neural interacting Hamiltonian(NIH). The drift phase is done analytically with a Kepler solver, and the velicity kicks are done by a neural network. The nerual network replaces the function that numerically computes the interacting Hamiltonian.

# Instellation
The code in this repository requires the following packages: `abie`, `torch`, and `matplotlib`. They can be installed easily with the following command:

    pip install abie torch matplotlib 

The code generally requires Python 3.7 or newer (Python 3.8 is tested).

# Getting Started
The nerual network behind the NIH is pretrained with a small number of three-body systems, where two low-mass planets orbit around a solar-type star. Despite the NIH being trained with system of *N*=3, the resulting WH-NIH integrator can deal with arbitraray *N*.

To retrain the NIH, please generate the training data using

    python generate_training_data.py

Modify the initial conditions in `generate_training_data.py` if desired. `Then perform the training

    python train.py

After that, the WH-NIH integrator can be tested with

    python test_wh.py

Every time when `test_wh.py` is executed, a random planetary system will be created, which is subsequently integrated with a traditional WH integrator and a WH-NIH integrator. Since the initial conditions are identical, the results from both integrator can be directly compared. A few plots `compare_*.pdf` are generated to make this comparison.

