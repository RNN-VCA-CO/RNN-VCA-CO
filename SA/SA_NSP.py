import numpy as np
import random as rd
import itertools
import math
import time
import os

"""
NSP utility functions
"""
# qubo formulation of NSP by d-wave
def dwave_nsp_qubo(n_nurses, n_days):

    # row, col size of J matrix
    size = n_nurses * n_days
    # dictionary for aapping nurse,day to indea in the solution vector
    d = {(nurse,day):nurse*n_days+day for nurse in range(n_nurses) for day in range(n_days)}

    # HNC correlation coeff
    a = 3.5
    # HSC coeff and constants
    hsc_coeff = 1.3
    workload = 1
    effort = 1
    # SNC coeff and constants
    snc_coeff = .3
    min_duty_days = int(n_days/n_nurses)
    preference = 1

    # qubo matrix
    J = np.zeros(shape=(size,size))

    # placing the HNC entries in J 
    for nurse in range(n_nurses):
        for day in range(n_days-1):
            i, j = d[(nurse,day)], d[(nurse,day+1)]
            J[i][j] = a

    # placing the HSC entries in J 
    for nurse in range(n_nurses):
        for day in range(n_days):
            index = d[(nurse, day)]
            J[index, index] += hsc_coeff * (effort**2 - (2*workload*effort))

    for day in range(n_days):
        for nurse1 in range(n_nurses):
            for nurse2 in range(nurse1 + 1, n_nurses):
                index1 = d[(nurse1, day)]
                index2 = d[(nurse2, day)]
                J[index1, index2] += 2*hsc_coeff*effort**2

    # placing the SNC entries in J 
    for nurse in range(n_nurses):
        for day in range(n_days):
            index = d[(nurse, day)]
            J[index, index] += snc_coeff * (preference**2 - (2*preference*min_duty_days))

    for nurse in range(n_nurses):
        for day1 in range(n_days):
            for day2 in range(day1 + 1, n_days):
                index1 = d[(nurse, day1)]
                index2 = d[(nurse, day2)]
                J[index1, index2] += 2*snc_coeff*preference**2

    offset = hsc_coeff*n_days*workload**2 + snc_coeff*n_nurses*min_duty_days**2
    
    return J, offset


# compute energy of sample/samples
# returns energy for an array of samples with an overall shape of (M,N) OR a single np sample of shape (N,)
def nsp_matrixelements(samples, J, offset):

    xJ = np.dot(samples, J)
    energies = np.einsum('ij,ij->i', xJ, samples)
    energies += offset
    return energies

"""
SA class
"""
class SA:

    def __init__(self, n_samples, N, n_warmup, n_anneal, n_eq, J, offset, T0=2.0):
        """
        n_samples - number of discrete, unrelated samples that are concurrently processed
        N - system size
        n_warmup - there are N metropolis moves per warmup step at temperature T=T0
        n_anneal - number of annealing steps where temperature is decreased after which the equilibrating steps follows
        n_eq - number of equilibrating steps after which there are N metropolis steps
        J - qubo matrix of NSP instance
        offset - constant that is used in energy calculations
        T0 = initial temperature
        """
        # seed for reproducibility
        seed = 111
        rd.seed(seed)
        np.random.seed(seed)

        # size of the system
        self.num_sample = n_samples
        self.size = N
        # warm up and annealing loops
        self.T0 = T0
        self.n_warmup = n_warmup
        self.n_anneal = n_anneal
        self.n_equilibrium = n_eq
        # J and offset for energy calculation
        self.J = J
        self.offset = offset

        # initialize random binary vector and its E
        self.samples = np.random.choice([0,1], size=(self.num_sample, self.size))
        self.energies = nsp_matrixelements(self.samples, self.J, self.offset)

    # perform warm up and annealing
    def run(self):

        # warm up process
        for i in range(self.n_warmup):
            # Sweep step
            for j in range(self.size):
                # metropolis step
                self.metropolis(self.T0)
        # annealing step
        for i in range(self.n_anneal):
            T = self.T0 - self.T0*i/self.n_anneal
            # equilibrating step
            for j in range(self.n_equilibrium):
                # sweep
                for k in range(self.size):
                    # metropolis step
                    self.metropolis(T)

        return self.samples, self.energies


    def metropolis(self, T):

        # generate a new, neighboring vector and its energy
        index = np.random.choice([0,self.size-1], size=self.num_sample)
        index = np.random.randint(0,self.size, size=self.num_sample)
        samples_new = self.samples.copy()
        
        # get the respective binary vlues
        values = samples_new[np.arange(self.num_sample),index]
        # flip values
        values = np.bitwise_xor(values, 1)
        # create the new samples
        samples_new[np.arange(self.num_sample),index] = values
        
        # compute E of new vector
        energies_new = nsp_matrixelements(samples_new, self.J, self.offset)
        # get the delta between E and E_new
        delta_energies = energies_new - self.energies
        
        # get probs of new samples replacing old ones
        probs = np.exp(-delta_energies/T)
        replace_index = probs > np.random.random(self.num_sample)
        # replace old samples
        self.samples[replace_index] = samples_new[replace_index]
        self.energies[replace_index] = energies_new[replace_index]

"""
Main
"""
if __name__ == '__main__':
    # NSP configuration
    D = 15
    N = 7
    system_size = N*D
    Q, offset = dwave_nsp_qubo(N, D)
    # VCA hyperparameters
    n_warmup = 1000
    n_anneal = 128
    n_eq = 5
    sa_model = SA(n_samples=50, N=system_size, n_warmup=n_warmup, n_anneal=n_anneal, n_eq=n_eq, J=Q, offset=offset, T0=2.0)
    sa_samples, sa_energies = sa_model.run()
    np.save("NSP_energies", sa_energies)
    np.save("NSP_samples", sa_samples)
