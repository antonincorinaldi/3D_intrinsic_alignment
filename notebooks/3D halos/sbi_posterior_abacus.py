### This file allows to load the posterior used to compute summary statistics of galaxy surveys (in 3D_halos.ipynb). 
### It can be directly run on a cluster.



import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from random import *

from scipy.spatial.transform import Rotation

import torch
from torch.distributions import Uniform, Distribution

from sbi.utils import BoxUniform, RestrictedPrior

from sbi.analysis import pairplot
from sbi.inference import NPE, SNPE, simulate_for_sbi, infer
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
    prepare_for_sbi
)

import corner

from astropy.io import fits

from tqdm import tqdm




### Downloading the Abacus halos (z=0.575)





halos_table = Table.read('/n17data/corinaldi/halos/Abacus_halos_z0.575_all_mass.fits')
nb_halos=200_000

random_indices = np.random.choice(len(halos_table), size=nb_halos, replace=False)
halos_table = halos_table[random_indices]




### Some functions required for further work...

# - ...to compute ellipticities



def e_complex(a,b,r):
    abs_e = (1-(b/a)) / (1+(b/a))
    e1 = abs_e*np.cos(2*r)
    e2 = abs_e*np.sin(2*r)
    return e1, e2

def abs_e(e1,e2):
    return np.sqrt(e1*e1+e2*e2)

def a_b(e1,e2):
    e = abs_e(e1,e2)
    return 1+e,1-e 


# ...to format ellipsoidal halos to match Abacus



def format_ellipsoid(eigenvectors, eigenvalues, position = np.asarray([0,0,0])):
    '''
    Formatt ellipsoid parameters to match Abacus. 
    Eigenvectors and values must be in order of least to greatest
    '''
    el = Table()
    el['sigman_eigenvecsMin_L2com'] = eigenvectors[0]
    el['sigman_eigenvecsMid_L2com'] = eigenvectors[1]
    el['sigman_eigenvecsMaj_L2com'] = eigenvectors[2]

    el['sigman_L2com'] = np.sqrt(eigenvalues)
    el['sigma_L2com'] = position

    return el





def population_3D(mu_tau_B, mu_tau_C,
                             sigma_tau_B, sigma_tau_C,
                             r_tau,
                             el=halos_table,
                             nb_halos=nb_halos,
                             batch_size=5000,
                             max_iter=1000):
  

    halos_table2 = el.copy()
    axis_orig = np.array(halos_table2['sigman_L2com']**2) 

    valid_axes = np.zeros_like(axis_orig)
    found_mask = np.zeros(nb_halos, dtype=bool)

    n_iter = 0
    while not np.all(found_mask) and n_iter < max_iter:
        n_iter += 1

        remaining = np.sum(~found_mask)
        if remaining == 0:
            break

        taus = np.random.multivariate_normal(
            mean=[mu_tau_B, mu_tau_C],
            cov=[
                [sigma_tau_B**2, r_tau * sigma_tau_B * sigma_tau_C],
                [r_tau * sigma_tau_B * sigma_tau_C, sigma_tau_C**2],
            ],
            size=min(batch_size, remaining)
        )

        tau_B2 = np.clip(taus[:, 0], 0, 1)
        tau_C2 = np.clip(taus[:, 1], 0, 1)

        idx_remaining = np.where(~found_mask)[0][:len(tau_B2)]
        A = axis_orig[idx_remaining, 0]
        B = axis_orig[idx_remaining, 1] * tau_B2
        C = axis_orig[idx_remaining, 2] * tau_C2

        mask = (B / A <= 1) & (C / A <= 1) & (B >= C) & (B > 0) & (C > 0)

        valid_idx = idx_remaining[mask]
        valid_axes[valid_idx, 0] = A[mask]
        valid_axes[valid_idx, 1] = B[mask]
        valid_axes[valid_idx, 2] = C[mask]

        found_mask[valid_idx] = True

    if not np.all(found_mask):
        missing = np.where(~found_mask)[0]
        valid_axes[missing] = axis_orig[missing]

    eigenvecs_Min = halos_table2['sigman_eigenvecsMin_L2com']
    eigenvecs_Mid = halos_table2['sigman_eigenvecsMid_L2com']
    eigenvecs_Max = halos_table2['sigman_eigenvecsMaj_L2com']
    eigenvectors = np.stack((eigenvecs_Min, eigenvecs_Mid, eigenvecs_Max), axis=1)

    ellipsoids = np.array([
        format_ellipsoid(eigenvectors[i, :, :], valid_axes[i, :])
        for i in range(nb_halos)
    ])

    evcl = np.array([
        ellipsoids['sigman_eigenvecsMaj_L2com'],
        ellipsoids['sigman_eigenvecsMid_L2com'],
        ellipsoids['sigman_eigenvecsMin_L2com']
    ])
    evcl = np.transpose(evcl, (1, 0, 2))
    axis_lengths = ellipsoids['sigman_L2com']**2

    return evcl, axis_lengths




# Function to project the 3D ellipsoidal galaxies of the modeled population in 2D along and perpendicular to the line-of-sight (LOS)
# Returns the 2 components of the projected ellipticity of the galaxies

def projection (evcl, evls, p_axis='', e_bins=np.linspace(0,1,100)):

    # Projection 3D => 2D
    if p_axis=='x': # Projection perpendicular to the LOS
        K = np.sum(evcl[:,:,0][:,:,None]*(evcl/evls[:,None]), axis=1)
        r = evcl[:,:,2] - evcl[:,:,0] * K[:,2][:,None] / K[:,0][:,None]
        s = evcl[:,:,1] - evcl[:,:,0] * K[:,1][:,None] / K[:,0][:,None] 

    if p_axis=='y': # Projection along the LOS
        K = np.sum(evcl[:,:,1][:,:,None] * (evcl/evls[:,None]), axis=1)
        r = evcl[:,:,0] - evcl[:,:,1] * K[:,0][:,None] / K[:,1][:,None]
        s = evcl[:,:,2] - evcl[:,:,1] * K[:,2][:,None] / K[:,1][:,None]


    # Coefficients A,B,C (eq 23 of (2))
    A1 = np.sum(r**2 / evls, axis=1)
    B1 = np.sum(2*r*s / evls, axis=1)
    C1 = np.sum(s**2 / evls, axis=1)


    # Axis a_p,b_p and orientation angle r_p of the projected galaxy
    r_p = np.pi / 2 + np.arctan2(B1,A1-C1)/2
    a_p = 1/np.sqrt((A1+C1)/2 + (A1-C1)/(2*np.cos(2*r_p)))
    b_p = 1/np.sqrt(A1+C1-(1/a_p**2))


    # Projected ellipticity
    e1, e2 = e_complex(a_p, b_p, r_p)

    e = [e1,e2] ; e=np.array(e)

    # Final output = summary statistics = P(e)
    e_counts,_ = np.histogram(np.sqrt(e[0,:]**2+e[1,:]**2),bins=e_bins)

    return e_counts






### Simulator to project 3D galaxies and their host-halos in 2D



# SIMULATOR (for simulation-based inference): projection of the 3D galaxies-halos in 2D along the line of sight ('y')
# Output = summary statistics = P(e)


def simulator(theta, 
                el=halos_table,
                nb_halos=nb_halos, 
                p_axis='y', # The direction of projection (here 'y' denotes by convention the direction of the line-of-sight)
               ):

    mu_tau_B, mu_tau_C, sigma_tau_B, sigma_tau_C, r_tau = theta

    evcl, evls = population_3D (mu_tau_B, mu_tau_C, sigma_tau_B, sigma_tau_C, r_tau, el, nb_halos)

    e_counts = projection (evcl, evls, p_axis=p_axis)
    
    return torch.tensor(e_counts / float(nb_halos), dtype=torch.float32)




### Prior 


class CustomPrior(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Define uniform priors for the parameters
        self.prior_uniform = BoxUniform(
            low=torch.tensor( [0., 0., 0.,  0., -0.5]), 
            high=torch.tensor([1., 1., 1.5, 1.5, 1.])
        )
    
    def log_prob(self, x):
        # Check the constraint mu_tau_B > mu_tau_C
        mutauB_minus_mutauC = x[..., 0] - x[..., 1]  # assuming mu_tauB is at index 0 and mu_C at index 1
        mask = (mutauB_minus_mutauC > 0)
        return torch.where(mask, self.prior_uniform.log_prob(x), torch.tensor(-float('inf')))

    
    def sample(self, sample_shape=torch.Size()):
        if len(sample_shape) == 0:
            sample_shape = torch.Size([1])  # Default to a single sample if sample_shape is empty
            
        samples = []
        while len(samples) < sample_shape[0]:
            sample = self.prior_uniform.sample((1,))
            if (-sample[..., 0] + sample[..., 1]).lt(0) :  # this checks if it is less than zero, so I switched it around
                samples.append(sample)
                
        return torch.cat(samples, dim=0)

# Instantiate the custom prior
prior = CustomPrior()



# Running the simulations, training of the neural network and estimation of the posterior (NPE = Neural Posterior Estimation) 

posterior = infer(simulator, prior, method = 'NPE', num_simulations = 10000, num_workers = 64)


### Saving the estimated posterior

torch.save(posterior,'posterior_abacus.pt')

