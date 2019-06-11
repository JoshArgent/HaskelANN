# HaskelANN
A feed-forward Artificial Neural Network implementation in Haskell with a stochastic gradient descent backpropagation training algorithm. Gradients are calculated using an automatic differentiation implementation.

## Instructions
The ANN framework is fairly self contained, there are only 4 source files required to run the two demonstrations:
* ANN.hs - the main ANN framework
* AutomaticDifferentiation.hs - provides automatic differentiation types
* DataProcessing.hs - functions for loading and preparing data for machine learning
* Demo.hs - two classification demonstrations (requires iris.csv and breast_cancer.csv - see below)

To run the demonstrations:
1. Open GHCI
2. :load Demo
3. Call either the 'demo1' or 'demo2' function

## Data
The demonstrations use freely available data from the [UCI Machine Learning repository](https://archive.ics.uci.edu/ml/datasets.php).
