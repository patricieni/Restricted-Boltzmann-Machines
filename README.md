# Restricted-Boltzmann-Machines
Train an RBM to infer Bars-As-Stripes patterns and other dummy examples. 

A few ideas that I used were taken from here, like how to iniatialize the weights as a function of the hidden/visible units. 
Biases were initialized to zero although there are better ways (see the guide)
https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf

I also used the below tutorial on RBM's. 
http://www.sciencedirect.com/science/article/pii/S0031320313002495

The contrastive divergence algorithm for learning the weights was done with k = 1 although you can change that. 
I used stochastic batch gradient descent where you shuffle the data at each iteration and plotted the error as you learn. After 3000 cycles you learn very little, after 4000 you don't really learn. 

Bars-as-stripes: You have 30 matrices (4x4) (reshape them as vectors) that can either have horizontal lines or vertical lines(whenever a line is present you have a row/column of ones). 

Once your RBM has learned the weights, the idea is to reconstruct a partial input by sampling from the joint distribution over the hidden and visible units.

TODO: Add more information on RBM theory and the proofs
TODO: Refactor the huge for loops into matrix operations. 
