# RNN-VCA-CO
This code is used to produce the results of our paper `Supplementing RNNs with annealing to Solve Optimization Problems`. We use both Vanilla and Dilated RNNs to solve the Maximum-Cut, Nurse Scheduling and Traveling Salesman Problems.

This code was build on top of the code in this repository: https://github.com/VectorInstitute/VariationalNeuralAnnealing

## `Summary`

- The data we used for our paper can be found in the directory `Data`.

- The code we used to run Variational Classical Annealing (VCA) can be found in `VCA`.

- The code we used to run Simulated Annealing (SA) can be found in `SA`.

## `Note`

- 'VCA_Max-Cut.ipynb' and 'VCA_NSP.ipynb' files use tensorflow v2 with latest numpy version.
- The .py files in directory 'TSP_code' use tensorflow v1 with an older numpy version (v1.16) for compatibility.

## `License`
This code is under the ['Attribution-NonCommercial-ShareAlike 4.0 International'](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.
 
