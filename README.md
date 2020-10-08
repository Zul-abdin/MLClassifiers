# MLClassifiers
Naive Bayes and Perceptron implementation in Python3.6 to classify faces and digits represented in text files.

## Running Program
The Naive Bayes tests and Perceptron tests can be run using `python bayesAlgo.py` and `python bayesAlgo.py` respectfully.
For each algorithm, the algorithm will use predifined training and validation/testing data sets to train the algorithms and then test prediction accuracy, with 5 trials each using training sample sizes from 10%-100%.

## bayesAlgo.py
Defines a class to create a bayes classifier given training/test image files, and thier metadata (size).
Interesting variables to experiment with include the Laplace smoothing constant, k.

## perceptronAlgo.py
Defines a class to create a perceptron classifier given training/test image files, and thier metadata (size/feature enumeration definitions).
Interesting variables to experiment with include the weight incrementing threshold constant.
