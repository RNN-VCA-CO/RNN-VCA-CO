#!/usr/bin/python
# -*- coding: utf-8 -*-

#----------------------------------------------------------------------------------------------------------
"""
This implementation is an adaptation of RNN Wave Functions' code https://github.com/mhibatallah/RNNWavefunctions
Title : Implementation of Variational Neural Annealing for the TSP
Description : TSP
"""
#-----------------------------------------------------------------------------------------------------------

import tensorflow as tf
import numpy as np
import os
import time
import random
from math import ceil
import argparse

from DilatedRNN import DilatedRNN
from DilatedRNN_WeightSharing import DilatedRNN_WeightSharing
from Helper_functions import *

#### Hyperparams
parser = argparse.ArgumentParser()
parser.add_argument('--N', type = int, default=20)
parser.add_argument('--T0', type = float, default=1.0)
parser.add_argument('--seed', type = int, default=111)
parser.add_argument('--lr', type = float, default=1e-3)
parser.add_argument('--numsamples', type = int, default=50)
parser.add_argument('--numunits', type = int, default=20)
parser.add_argument('--Nwarmup', type = int, default=5)
parser.add_argument('--WeightSharing', type = str2bool, default="True")

args = parser.parse_args()

# Note:
seed = args.seed
N = args.N #total number of spins
numunits = args.numunits #number of memory units for each RNN cell
numlayers = ceil(np.log2(N)) #number of layers
numsamples = args.numsamples #number of samples used for training
lr = args.lr #learning rate
T0 = args.T0 #initial temperature
num_warmup_steps = args.Nwarmup #number of warmup steps
num_equilibrium_steps = 5 #number of training steps after each annnealing step
activation_function = tf.nn.elu #activation function used for the Dilated RNN cell

#Defining the other parameters
units=[numunits]*numlayers #list containing the number of hidden units for each layer of the RNN

#Seeding for reproducibility purposes
random.seed(seed)  # `python` built-in pseudo-random generator
np.random.seed(seed)  # numpy pseudo-random generator
tf.compat.v1.set_random_seed(seed)  # tensorflow pseudo-random generator

coordinates = np.random.rand(N, 2) #Generate coordinates of the cities

np.savetxt("./coordinates_N"+str(N)+".txt", coordinates)

list_Nannealing = [2**i for i in range(4, 14+1)]

for num_annealing_steps in list_Nannealing:

    tf.compat.v1.reset_default_graph()

    print('\n')
    print("Number of cities =", N)
    print("Initial_temperature =", T0)
    print('Seed = ', seed)
    print("Learning rate", lr)
    print("Num units =", numunits)
    print("Weight Sharing =", args.WeightSharing)

    num_steps = num_annealing_steps*num_equilibrium_steps + num_warmup_steps

    print("\nNumber of annealing steps = {0}".format(num_annealing_steps))
    print("Number of training steps = {0}".format(num_steps))
    print("Number of layers = {0}\n".format(numlayers))

    # Intitializing the RNN-----------
    if args.WeightSharing:
        DRNN = DilatedRNN_WeightSharing(N,units=units,cell=tf.compat.v1.nn.rnn_cell.BasicRNNCell, activation = activation_function, seed = seed) #contains the graph with the RNNs
    else:
        DRNN = DilatedRNN(N,units=units,cell=tf.compat.v1.nn.rnn_cell.BasicRNNCell, activation = activation_function, seed = seed) #contains the graph with the RNNs

    ########## Checkpointing
    if not os.path.exists('./Check_Points/'):
        os.mkdir('./Check_Points/')

    if not os.path.exists('./Check_Points/Size_'+str(N)):
        os.mkdir('./Check_Points/Size_'+str(N))

    savename = '_N'+str(N)+'_WeightSharing'+str(args.WeightSharing)+'_samp'+str(numsamples)+'_numunits'+str(numunits)+'_numlayers'+str(numlayers)+'_lr'+str(lr)+"_Nanneal"+str(num_annealing_steps)+"_T0"+str(T0)
    backgroundpath = './Check_Points/Size_'+str(N)
    filename = backgroundpath+'/RNNwavefunction'+savename+'.ckpt'

    #Building the graph -------------------
    with tf.compat.v1.variable_scope(DRNN.scope,reuse=tf.compat.v1.AUTO_REUSE):
        with DRNN.graph.as_default():

            global_step = tf.Variable(0, trainable=False)
            learningrate_placeholder = tf.compat.v1.placeholder(dtype=tf.float64,shape=[])
            learningrate = tf.compat.v1.train.exponential_decay(learningrate_placeholder, global_step, 100, 1.0, staircase=True)

            #Defining the optimizer
            optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=learningrate)

            #Defining Tensorflow placeholders
            Eloc=tf.compat.v1.placeholder(dtype=tf.float64,shape=[numsamples])
            sampleplaceholder_forgrad=tf.compat.v1.placeholder(dtype=tf.int32,shape=[numsamples,N])
            log_probs_forgrad = DRNN.log_probability(sampleplaceholder_forgrad,inputdim=N)
            samplesandprobs = DRNN.sample(numsamples=numsamples,inputdim=N)

            T_placeholder = tf.compat.v1.placeholder(dtype=tf.float64,shape=())

            #Here we define a fake cost function that would allows to get the gradients of free energy using the tf.stop_gradient trick
            Floc = Eloc + T_placeholder*log_probs_forgrad
            cost = tf.reduce_mean(tf.multiply(log_probs_forgrad,tf.stop_gradient(Floc))) - tf.reduce_mean(log_probs_forgrad)*tf.reduce_mean(tf.stop_gradient(Floc))

            gradients, variables = zip(*optimizer.compute_gradients(cost))
            #Calculate Gradients---------------

            #Define the optimization step
            optstep=optimizer.apply_gradients(zip(gradients,variables), global_step = global_step)

            #Tensorflow saver to checkpoint
            saver=tf.compat.v1.train.Saver()

            #For initialization
            init=tf.compat.v1.global_variables_initializer()
    #----------------------------------------------------------------

    #Starting Session------------
    #GPU management
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    sess=tf.compat.v1.Session(graph=DRNN.graph, config=config)
    sess.run(init)

    ## Counting the number of parameters
    # with DRNN.graph.as_default():
    #     variables_names =[v.name for v in tf.compat.v1.trainable_variables()]
    #     sum = 0
    #     values = sess.run(variables_names)
    #     for k,v in zip(variables_names, values):
    #         v1 = tf.reshape(v,[-1])
    #         print(k,v1.shape)
    #         sum +=v1.shape[0]
    #     print('The sum of params is {0}'.format(sum))

    #To store data
    meanEnergy=[]
    varEnergy=[]
    varFreeEnergy = []
    meanFreeEnergy = []

    #Loading previous trainings----------
    with tf.compat.v1.variable_scope(DRNN.scope,reuse=tf.compat.v1.AUTO_REUSE):
        with DRNN.graph.as_default():

            try:
                print("Loading the model from checkpoint")
                saver.restore(sess,filename)
                print("Loading was successful!")
            except:
                print("Loading was failed!")

            try:
                print("Trying to load energies!")

                meanEnergy=np.loadtxt(backgroundpath + '/meanEnergy'+ savename +'.txt').tolist()
                varEnergy=np.loadtxt(backgroundpath + '/varEnergy' + savename +'.txt').tolist()
                meanFreeEnergy = np.savetxt(backgroundpath + '/meanFreeEnergy' + savename +'.txt').tolist()
                varFreeEnergy = np.savetxt(backgroundpath + '/varFreeEnergy' + savename +'.txt').tolist()
            except:
                print("Failed! No need to load energies if running for the first time!")
    #------------------------------------


    ## Run Variational Annealing
    with tf.compat.v1.variable_scope(DRNN.scope,reuse=tf.compat.v1.AUTO_REUSE):
        with DRNN.graph.as_default():

            samples = np.ones((numsamples, N), dtype=np.int32)

            T = T0 #initializing temperature

            start = time.time()

            for it in range(len(meanEnergy),num_steps+1):

                #Annealing
                if it>=num_warmup_steps and  it <= num_annealing_steps*num_equilibrium_steps + num_warmup_steps and it % num_equilibrium_steps == 0:
                  annealing_step = (it-num_warmup_steps)/num_equilibrium_steps
                  T = T0*(1-annealing_step/num_annealing_steps)

                #Showing current status after that the annealing starts
                if it%num_equilibrium_steps==0:
                  if it <= num_annealing_steps*num_equilibrium_steps + num_warmup_steps and it>=num_warmup_steps:
                      annealing_step = (it-num_warmup_steps)/num_equilibrium_steps
                      print("\nAnnealing step: {0}/{1}".format(annealing_step,num_annealing_steps))

                samples, log_probabilities = sess.run(samplesandprobs)

                # Estimating the local energies
                local_energies = TSP_energy_2D(coordinates,samples)

                meanE = np.mean(local_energies)
                varE = np.var(local_energies)

                #adding elements to be saved
                meanEnergy.append(meanE)
                varEnergy.append(varE)

                meanF = np.mean(local_energies+T*log_probabilities)
                varF = np.var(local_energies+T*log_probabilities)

                meanFreeEnergy.append(meanF)
                varFreeEnergy.append(varF)

                if it%num_equilibrium_steps==0:
                    print('mean(E): {0}, mean(F): {1}, var(E): {2}, var(F): {3}, #samples {4}, #Training step {5}'.format(meanE,meanF,varE,varF,numsamples, it))
                    print("Temperature: ", T)
                    print(samples[0])

                if it%500==0 or it==num_steps:
                    #Saving the performances for loading
                    saver.save(sess,filename)
                    np.savetxt(backgroundpath + '/meanEnergy' + savename +'.txt', meanEnergy)
                    np.savetxt(backgroundpath + '/varEnergy' + savename +'.txt', varEnergy)
                    np.savetxt(backgroundpath + '/meanFreeEnergy' + savename +'.txt', meanFreeEnergy)
                    np.savetxt(backgroundpath + '/varFreeEnergy' + savename +'.txt', varFreeEnergy)

                #Run gradient descent step
                sess.run(optstep,feed_dict={Eloc:local_energies, sampleplaceholder_forgrad: samples, learningrate_placeholder: lr, T_placeholder:T})

                if it%5 == 0:
                    print("Elapsed time is =", time.time()-start, " seconds")
                    print('\n\n')


            ## Save the final model
            saver.save(sess,filename)

            #Here we produce samples at the end of annealing
            Nsteps = 500
            numsamples_estimation = 10**6 #Num samples to be obtained at the end
            numsamples_perstep = numsamples_estimation//Nsteps #The number of steps taken to get "numsamples_estimation" samples (to avoid memory allocation issues)

            samplesandprobs_final = DRNN.sample(numsamples=numsamples_perstep,inputdim=N)
            energies = np.zeros((numsamples_estimation))
            solutions = np.zeros((numsamples_estimation, N), dtype = np.int32)
            log_probs_final = np.zeros((numsamples_estimation))
            print("\nSaving energy and variance before the end of annealing")

            for i in range(Nsteps):
                # print("\nsampling started")
                samples_final, log_probs_final[i*numsamples_perstep:(i+1)*numsamples_perstep] = sess.run(samplesandprobs_final)
                # print("\nsampling finished")
                energies[i*numsamples_perstep:(i+1)*numsamples_perstep] = TSP_energy_2D(coordinates,samples_final)
                solutions[i*numsamples_perstep:(i+1)*numsamples_perstep] = samples_final
                print("Sampling step:" , i+1, "/", Nsteps)

            print("meanE = ", np.mean(energies))
            print("varE = ", np.var(energies))
            print("minE = ",np.min(energies))
            print("Optimal config =", solutions[np.argmin(energies)])
            print("Entropy =", -np.mean(log_probs_final))

    #----------------------------
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 10))
    plot_tsp(coordinates, solutions[np.argmin(energies)], ax)
    fig.savefig(backgroundpath + '/TSPtour' + savename +'.png', dpi = 300)

    np.save(backgroundpath + '/Energies' + savename +'.npy', energies)
    np.save(backgroundpath + '/OptimalSolution' + savename +'.npy', solutions[np.argmin(energies)])
    np.save(backgroundpath + '/FinalEntropy' + savename +'.npy', -np.mean(log_probs_final))
