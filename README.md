# MCMC main

This is a Markov-chain Monte Carlo (MCMC) fitting routine to fit model spectra to observations of jets in evolved binary systems.
We implemented the emcee-package by Dan Foreman-Mackey
https://emcee.readthedocs.io/en/stable/


## Usage

The main script is main.py, which takes the object id and the input parameter file as command line arguments, i.e.:  

`python main.py -o BD+46_442 -dat BD+46_442_x_wind.dat`

You can create new 'jet' objects in Cone.py.

Current jet configurations are
- Simple stellar jet
- Stellar jet
- X-wind
- Strict X-wind
- Scaled disk wind
- Strict scaled disk wind

